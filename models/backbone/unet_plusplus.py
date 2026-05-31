# 版权所有 (c) OpenMMLab。保留所有权利。
from functools import partial
from itertools import chain
from typing import Sequence
from timm.models.layers import DropPath
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmengine.model import ModuleList, Sequential
from mmengine.logging import MMLogger
from mmengine.runner.checkpoint import CheckpointLoader

from mmcv.cnn import build_activation_layer
from mmengine.model.weight_init import constant_init, trunc_normal_init

from models.builder import ROTATED_BACKBONES
from models.backbone.layers import build_LayerNorm2d_layer, FFN
from models.backbone.unified_multimodal_stem import UnifiedMultiModalStem
from mmcv.runner import BaseModule
from timm.models.layers import trunc_normal_


class UNetPPConvBlock(nn.Module):
    """
    UNet++ 简单卷积块 - 只用于中间节点
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 drop_path_rate=0.):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = self._build_norm(norm_cfg, out_channels)
        self.act = build_activation_layer(act_cfg)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = self._build_norm(norm_cfg, out_channels)

        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    def _build_norm(self, norm_cfg, channels):
        """构建归一化层"""
        if norm_cfg is None:
            return nn.BatchNorm2d(channels)

        if isinstance(norm_cfg, dict) and 'type' in norm_cfg:
            if norm_cfg['type'] == 'BN':
                return nn.BatchNorm2d(channels)
            elif norm_cfg['type'] == 'LN2d':
                return build_LayerNorm2d_layer(norm_cfg, channels)
            else:
                return nn.BatchNorm2d(channels)
        else:
            return nn.BatchNorm2d(channels)

    def forward(self, x):
        # 第一个卷积
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act(x)

        # 第二个卷积
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.act(x)

        return x


class UNetPPBlock(nn.Module):
    """
    UNet++ 块 - 用于主路径，带残差连接
    完全仿照 ConvNeXtBlock 的接口设计
    """
    def __init__(self,
                 in_channels,
                 dw_conv_cfg=dict(kernel_size=7, padding=3),  # 保持接口兼容
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 mlp_ratio=4.,  # 保持接口兼容
                 linear_pw_conv=True,  # 保持接口兼容
                 drop_path_rate=0.,
                 layer_scale_init_value=1e-6,
                 use_grn=False,  # 保持接口兼容
                 use_multiscale=True,  # 保持接口兼容
                 da_reduction=8):  # 保持接口兼容
        super().__init__()

        # UNet++ 标准双卷积结构
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.norm1 = self._build_norm(norm_cfg, in_channels)
        self.act = build_activation_layer(act_cfg)

        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.norm2 = self._build_norm(norm_cfg, in_channels)

        # Layer scale (保持与ConvNeXt一致)
        self.gamma = nn.Parameter(
            layer_scale_init_value * torch.ones((in_channels)),
            requires_grad=True) if layer_scale_init_value > 0 else None

        # Drop path
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    def _build_norm(self, norm_cfg, channels):
        """构建归一化层"""
        if norm_cfg is None:
            return nn.BatchNorm2d(channels)

        if isinstance(norm_cfg, dict) and 'type' in norm_cfg:
            if norm_cfg['type'] == 'BN':
                return nn.BatchNorm2d(channels)
            elif norm_cfg['type'] == 'LN2d':
                return build_LayerNorm2d_layer(norm_cfg, channels)
            else:
                return nn.BatchNorm2d(channels)
        else:
            return nn.BatchNorm2d(channels)

    def forward(self, x, dataset=None):
        shortcut = x

        # 第一个卷积
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act(x)

        # 第二个卷积
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.act(x)

        # Layer scale
        if self.gamma is not None:
            x = x.mul(self.gamma.view(1, -1, 1, 1))

        # Drop path + 残差连接
        x = self.drop_path(x)
        x = shortcut + x

        return x, None  # 返回None保持兼容性


@ROTATED_BACKBONES.register_module()
class BasicUNetPlusPlus(BaseModule):
    """
    基础 UNet++ Encoder - 完全仿照 ConvNeXt_moe 结构
    UNet++ 特点：密集跳跃连接和嵌套的U型结构
    """
    arch_settings = {
        'atto': {'depths': [2, 2, 6, 2], 'channels': [40, 80, 160, 320]},
        'femto': {'depths': [2, 2, 6, 2], 'channels': [48, 96, 192, 384]},
        'pico': {'depths': [2, 2, 6, 2], 'channels': [64, 128, 256, 512]},
        'nano': {'depths': [2, 2, 8, 2], 'channels': [80, 160, 320, 640]},
        'tiny': {'depths': [3, 3, 9, 3], 'channels': [96, 192, 384, 768]},
        'small': {'depths': [3, 3, 27, 3], 'channels': [96, 192, 384, 768]},
        'base': {'depths': [3, 3, 27, 3], 'channels': [128, 256, 512, 1024]},
        'swin_large': {'depths': [2, 2, 18, 2], 'channels': [192, 384, 768, 1536]},
        'large': {'depths': [3, 3, 27, 3], 'channels': [192, 384, 768, 1536]},
        'xlarge': {'depths': [3, 3, 27, 3], 'channels': [256, 512, 1024, 2048]},
        'huge': {'depths': [3, 3, 27, 3], 'channels': [352, 704, 1408, 2816]}
    }

    def __init__(self,
                 arch='pico',
                 in_channels=3,
                 stem_patch_size=4,
                 norm_cfg=dict(type='BN'),  # UNet++通常使用BN
                 act_cfg=dict(type='ReLU'),  # UNet++通常使用ReLU
                 linear_pw_conv=True,
                 use_grn=False,
                 drop_path_rate=0.,
                 layer_scale_init_value=1e-6,
                 out_indices=[0, 1, 2, 3],
                 da_reductions=[8, 8, 8],  # 保持接口兼容
                 frozen_stages=0,
                 gap_before_final_norm=False,
                 use_dense_connections=True,  # UNet++ 特有：密集连接
                 init_cfg=[
                     dict(type='TruncNormal', layer=['Conv2d', 'Linear'], std=.02, bias=0.),
                     dict(type='Constant', layer=['LayerNorm'], val=1., bias=0.),
                 ]):
        super().__init__(init_cfg=init_cfg)

        # 架构配置
        if isinstance(arch, str):
            assert arch in self.arch_settings, f'Unknown architecture: {arch}'
            arch = self.arch_settings[arch]
        elif isinstance(arch, dict):
            assert 'depths' in arch and 'channels' in arch

        self.depths = arch['depths']
        self.channels = arch['channels']
        self.num_stages = len(self.depths)
        self.use_dense_connections = use_dense_connections

        # 输出配置
        if isinstance(out_indices, int):
            out_indices = [out_indices]
        for i, index in enumerate(out_indices):
            if index < 0:
                out_indices[i] = 4 + index
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.gap_before_final_norm = gap_before_final_norm

        # DA reduction配置（保持接口兼容）
        if isinstance(da_reductions, int):
            self.da_reductions = [da_reductions] * 3
        else:
            self.da_reductions = da_reductions[:3] if len(da_reductions) >= 3 else da_reductions + [8] * (3 - len(da_reductions))

        # Drop path rates
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))]
        block_idx = 0

        # 下采样层
        self.downsample_layers = ModuleList()

        # 使用统一的多模态Stem（统计特性对齐）- 不下采样
        self.stem = UnifiedMultiModalStem(
            out_channels=self.channels[0],
        )

        # downsample_layers[0]: 只做LayerNorm，不下采样
        if norm_cfg is not None and isinstance(norm_cfg, dict) and norm_cfg.get('type') == 'LN2d':
            self.downsample_layers.append(
                nn.Sequential(build_LayerNorm2d_layer(norm_cfg, self.channels[0]))
            )
        else:
            # 使用 BatchNorm 作为默认
            self.downsample_layers.append(
                nn.Sequential(nn.BatchNorm2d(self.channels[0]))
            )

        # Stage 1-3 的下采样层（每个做2x下采样）
        for i in range(1, self.num_stages):
            if norm_cfg is not None and isinstance(norm_cfg, dict) and norm_cfg.get('type') == 'LN2d':
                downsample_layer = nn.Sequential(
                    build_LayerNorm2d_layer(norm_cfg, self.channels[i-1]),
                    nn.Conv2d(self.channels[i-1], self.channels[i], kernel_size=2, stride=2)
                )
            else:
                downsample_layer = nn.Sequential(
                    nn.BatchNorm2d(self.channels[i-1]),
                    nn.Conv2d(self.channels[i-1], self.channels[i], kernel_size=2, stride=2)
                )
            self.downsample_layers.append(downsample_layer)

        # 构建stages（编码器路径）
        self.stages = nn.ModuleList()
        for i in range(self.num_stages):
            depth = self.depths[i]
            channels = self.channels[i]

            stage = Sequential(*[
                UNetPPBlock(
                    in_channels=channels,
                    drop_path_rate=dpr[block_idx + j],
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    linear_pw_conv=linear_pw_conv,
                    layer_scale_init_value=layer_scale_init_value,
                    use_grn=use_grn,
                    da_reduction=self.da_reductions[i] if i < len(self.da_reductions) else 8
                ) for j in range(depth)
            ])
            block_idx += depth
            self.stages.append(stage)

        # UNet++ 密集连接块
        if self.use_dense_connections:
            self.dense_blocks = nn.ModuleDict()
            # 构建嵌套的密集连接块
            # X[i,j] 其中 i 是深度级别，j 是水平位置
            for i in range(self.num_stages - 1):  # 不需要最后一层的密集块
                for j in range(1, self.num_stages - i):
                    # 计算输入通道数
                    # j个左侧的特征（每个channels[i]） + 1个来自下层上采样的特征（也是channels[i]）
                    in_ch = self.channels[i] * (j + 1)

                    # 创建一个简单的卷积块来处理融合
                    self.dense_blocks[f'X_{i}_{j}'] = UNetPPConvBlock(
                        in_channels=in_ch,
                        out_channels=self.channels[i],
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg,
                        drop_path_rate=dpr[min(block_idx-1, len(dpr)-1)]
                    )

            # 上采样层
            self.upsample_layers = ModuleList()
            for i in range(self.num_stages - 1):
                self.upsample_layers.append(
                    nn.ConvTranspose2d(
                        self.channels[i+1],
                        self.channels[i],
                        kernel_size=2,
                        stride=2
                    )
                )

        # 输出归一化层
        for i in self.out_indices:
            if i < self.num_stages:
                if norm_cfg is not None and isinstance(norm_cfg, dict) and norm_cfg.get('type') == 'LN2d':
                    norm_layer = build_LayerNorm2d_layer(norm_cfg, self.channels[i])
                else:
                    norm_layer = nn.BatchNorm2d(self.channels[i])
                self.add_module(f'norm{i}', norm_layer)

        self._freeze_stages()

    def forward(self, x):
        """
        前向传播 - UNet++ 特有的密集连接结构
        x: (B, C, H, W) - C=1(SAR), C=3(RGB)
        """
        outs = []

        # 先通过stem（不下采样）
        x = self.stem(x)

        # 存储每个位置的特征
        features = {}

        # 编码器路径（左侧列）
        for i, stage in enumerate(self.stages):
            # 先通过downsample层
            x = self.downsample_layers[i](x)
            # 再通过stage中的blocks
            for each_layer in stage:
                x, _ = each_layer(x)

            features[f'X_{i}_0'] = x  # 存储编码器特征

        # 如果使用密集连接，构建UNet++的嵌套结构
        if self.use_dense_connections:
            # 逐列构建（从左到右）
            for j in range(1, self.num_stages):
                for i in range(self.num_stages - j):
                    # 上采样来自下层的特征
                    if i < len(self.upsample_layers):
                        up_feature = self.upsample_layers[i](features[f'X_{i+1}_{j-1}'])

                        # 拼接所有来自左侧的特征
                        concat_features = [features[f'X_{i}_{k}'] for k in range(j)]
                        concat_features.append(up_feature)

                        # 融合特征
                        fused = torch.cat(concat_features, dim=1)

                        # 通过密集块处理
                        if f'X_{i}_{j}' in self.dense_blocks:
                            fused = self.dense_blocks[f'X_{i}_{j}'](fused)
                            features[f'X_{i}_{j}'] = fused

        # 收集输出（保持与原ConvNeXt相同）
        for i in self.out_indices:
            if i < self.num_stages:
                # 直接使用编码器路径的特征（保持shape一致）
                feat = features.get(f'X_{i}_0')
                if feat is not None:
                    norm_layer = getattr(self, f'norm{i}')
                    if self.gap_before_final_norm:
                        gap = feat.mean([-2, -1], keepdim=True)
                        outs.append(norm_layer(gap).flatten(1))
                    else:
                        outs.append(norm_layer(feat))

        return tuple(outs)

    def _freeze_stages(self):
        for i in range(self.frozen_stages):
            downsample_layer = self.downsample_layers[i]
            stage = self.stages[i]
            downsample_layer.eval()
            stage.eval()
            for param in chain(downsample_layer.parameters(), stage.parameters()):
                param.requires_grad = False

    def train(self, mode=True):
        super(BasicUNetPlusPlus, self).train(mode)
        self._freeze_stages()

    def get_layer_depth(self, param_name: str, prefix: str = ''):
        """获取参数的层深度"""
        max_layer_id = 12 if self.depths[-2] > 9 else 6

        if not param_name.startswith(prefix):
            return max_layer_id + 1, max_layer_id + 2

        param_name = param_name[len(prefix):]

        if param_name.startswith('downsample_layers'):
            stage_id = int(param_name.split('.')[1])
            if stage_id == 0:
                layer_id = 0
            elif stage_id == 1 or stage_id == 2:
                layer_id = stage_id + 1
            else:
                layer_id = max_layer_id
        elif param_name.startswith('stages'):
            stage_id = int(param_name.split('.')[1])
            block_id = int(param_name.split('.')[2])
            if stage_id == 0 or stage_id == 1:
                layer_id = stage_id + 1
            elif stage_id == 2:
                layer_id = 3 + block_id // 3
            else:
                layer_id = max_layer_id
        else:
            layer_id = max_layer_id + 1

        return layer_id, max_layer_id + 2

    def init_weights(self):
        logger = MMLogger.get_current_instance()
        if self.init_cfg is None:
            logger.warn(f'No pretrained weights for {self.__class__.__name__}')
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_init(m, std=.02, bias=0.)
                elif isinstance(m, nn.LayerNorm):
                    constant_init(m, 1.0)
        else:
            assert 'checkpoint' in self.init_cfg
            ckpt = CheckpointLoader.load_checkpoint(
                self.init_cfg.checkpoint, logger=logger, map_location='cpu')

            if 'state_dict' in ckpt:
                _state_dict = ckpt['state_dict']
            elif 'model' in ckpt:
                _state_dict = ckpt['model']
            else:
                _state_dict = ckpt

            state_dict = OrderedDict()
            for k, v in _state_dict.items():
                if k.startswith('backbone.'):
                    k = k[9:]
                if k.startswith('module.'):
                    k = k[7:]
                state_dict[k] = v

            warnings = self.load_state_dict(state_dict, False)
            logger.info(f"Load pretrained weights: {warnings}")


@ROTATED_BACKBONES.register_module()
class UNetPlusPlus(BasicUNetPlusPlus):
    """
    多模态UNet++ - 支持模态标签
    继承BasicUNetPlusPlus并添加多模态支持
    """
    def __init__(self,
                 arch='pico',
                 in_channels=3,
                 stem_patch_size=4,
                 datasets=['sar', 'rgb', 'sdg'],
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 linear_pw_conv=True,
                 use_grn=False,
                 drop_path_rate=0.,
                 layer_scale_init_value=1e-6,
                 out_indices=[0, 1, 2, 3],
                 da_reductions=[8, 8, 16],
                 frozen_stages=0,
                 gap_before_final_norm=False,
                 use_dense_connections=True,
                 init_cfg=[
                     dict(type='TruncNormal', layer=['Conv2d', 'Linear'], std=.02, bias=0.),
                     dict(type='Constant', layer=['LayerNorm'], val=1., bias=0.),
                 ]):
        super().__init__(
            arch=arch,
            in_channels=in_channels,
            stem_patch_size=stem_patch_size,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            linear_pw_conv=linear_pw_conv,
            use_grn=use_grn,
            drop_path_rate=drop_path_rate,
            layer_scale_init_value=layer_scale_init_value,
            out_indices=out_indices,
            da_reductions=da_reductions,
            frozen_stages=frozen_stages,
            gap_before_final_norm=gap_before_final_norm,
            use_dense_connections=use_dense_connections,
            init_cfg=init_cfg
        )

        self.datasets = datasets if datasets is not None else ['single']

    def forward(self, x, datasets=None):
        """
        支持模态标签的前向传播
        """
        outs = []
        # 自动检测模态
        if datasets is None:
            if x.shape[1] == 1:
                dataset_type = 'sar'
            elif x.shape[1] == 3:
                dataset_type = 'rgb'
            elif x.shape[1] == 7:
                dataset_type = 'sdg'
            else:
                raise ValueError(f"Cannot infer modality from {x.shape[1]} channels")
        else:
            dataset_type = datasets[0] if isinstance(datasets, list) else datasets
        batch_size = x.shape[0]
        batch_datasets = [dataset_type] * batch_size

        # 先通过stem（不下采样）
        x = self.stem(x)

        # 存储每个位置的特征
        features = {}

        # 编码器路径（左侧列）
        for i, stage in enumerate(self.stages):
            # 先通过downsample层
            x = self.downsample_layers[i](x)

            # 再通过stage中的blocks
            for each_layer in stage:
                x, _ = each_layer(x, batch_datasets)

            features[f'X_{i}_0'] = x  # 存储编码器特征

        # 如果使用密集连接，构建UNet++的嵌套结构
        if self.use_dense_connections:
            # 逐列构建（从左到右）
            for j in range(1, self.num_stages):
                for i in range(self.num_stages - j):
                    # 上采样来自下层的特征
                    if i < len(self.upsample_layers):
                        up_feature = self.upsample_layers[i](features[f'X_{i+1}_{j-1}'])

                        # 拼接所有来自左侧的特征
                        concat_features = [features[f'X_{i}_{k}'] for k in range(j)]
                        concat_features.append(up_feature)

                        # 融合特征
                        fused = torch.cat(concat_features, dim=1)

                        # 通过密集块处理（注意这里不需要传 datasets）
                        if f'X_{i}_{j}' in self.dense_blocks:
                            fused = self.dense_blocks[f'X_{i}_{j}'](fused)
                            features[f'X_{i}_{j}'] = fused

        # 收集输出（保持与原ConvNeXt相同）
        for i in self.out_indices:
            if i < self.num_stages:
                # 直接使用编码器路径的特征（保持shape一致）
                feat = features.get(f'X_{i}_0')
                if feat is not None:
                    norm_layer = getattr(self, f'norm{i}')
                    if self.gap_before_final_norm:
                        gap = feat.mean([-2, -1], keepdim=True)
                        outs.append(norm_layer(gap).flatten(1))
                    else:
                        outs.append(norm_layer(feat))

        return tuple(outs)