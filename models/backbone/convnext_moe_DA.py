# 版权所有 (c) OpenMMLab。保留所有权利。
from functools import partial
from itertools import chain
from typing import Sequence
from timm.models.layers import DropPath
from collections import OrderedDict

import torch
import torch.nn as nn

from mmengine.model import ModuleList, Sequential
from mmengine.logging import MMLogger
from mmengine.runner.checkpoint import CheckpointLoader

from mmcv.cnn import build_activation_layer
from mmengine.model.weight_init import constant_init, trunc_normal_init

from models.builder import ROTATED_BACKBONES
from models.backbone.layers import build_LayerNorm2d_layer, FFN
from models.backbone.InceptionDWConv2d import InceptionDWConv2d
from models.backbone.unified_multimodal_stem import UnifiedMultiModalStem
from mmcv.runner import BaseModule
from timm.models.layers import trunc_normal_

class ConvNeXtBlock(nn.Module):
    """
    优化版ConvNeXt块
    保留所有现有功能，简化冗余部分
    """
    def __init__(self,
                 in_channels,
                 dw_conv_cfg=dict(kernel_size=7, padding=3),  # 保持接口
                 norm_cfg=dict(type='LN2d', eps=1e-6),
                 act_cfg=dict(type='GELU'),
                 mlp_ratio=4.,  # 保持接口，内部优化为1.0
                 linear_pw_conv=True,
                 drop_path_rate=0.,
                 layer_scale_init_value=1e-6,
                 use_grn=False,
                 use_multiscale=True,  # 保持接口兼容
                 da_reduction=8):  # 保持接口兼容
        super().__init__()

        # 使用 InceptionDWConv2d 替代标准 depthwise - 增强多方向条纹感知
        self.depthwise_conv = InceptionDWConv2d(
            in_channels,
            square_kernel_size=3,
            band_kernel_size=11,
            branch_ratio=0.25
        )

        # 剥离 DASC，换回标准的纯 ConvNeXt 深度卷积（用来回应审稿人做纯消融）
        # 标准 ConvNeXt 一般使用 7x7 卷积，groups=in_channels
        # self.depthwise_conv = nn.Conv2d(
        #     in_channels, 
        #     in_channels, 
        #     kernel_size=7, 
        #     padding=3, 
        #     groups=in_channels
        # )



        # LayerNorm
        self.linear_pw_conv = linear_pw_conv
        self.norm = build_LayerNorm2d_layer(norm_cfg, in_channels)

        # FFN - 优化为 mlp_ratio=1.0 节省显存
        mid_channels = int(4.0 * in_channels)  # 改为2.0，原本1.0太小

        if self.linear_pw_conv:
            pw_conv = nn.Linear
        else:
            pw_conv = partial(nn.Conv2d, kernel_size=1)

        self.ffn = FFN(in_channels, mid_channels, pw_conv, act_cfg, use_grn)

        # Layer scale & drop path
        self.gamma = nn.Parameter(
            layer_scale_init_value * torch.ones((in_channels)),
            requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    def forward(self, x, dataset=None):
        shortcut = x

        # 1. 深度卷积
        x = self.depthwise_conv(x)


        # 3. LayerNorm & FFN
        if self.linear_pw_conv:
            x = x.permute(0, 2, 3, 1)  # 注释掉permute
            x = self.norm(x, data_format='channel_last')  # 改为channel_first
            x = self.ffn(x)  # FFN暂时禁用
            x = x.permute(0, 3, 1, 2)  # 注释掉permute
        else:
            x = self.norm(x, data_format='channel_first')
            x = self.ffn(x)  # FFN暂时禁用

        if self.gamma is not None:
            x = x.mul(self.gamma.view(1, -1, 1, 1))

        x = self.drop_path(x)
        x = shortcut + x

        return x, None  # 返回None保持兼容性


@ROTATED_BACKBONES.register_module()
class ConvNeXt_moe(BaseModule):
    """
    多模态ConvNeXt - 简化版
    保留所有现有功能，整合统一的stem
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
                 norm_cfg=dict(type='LN2d', eps=1e-6),
                 act_cfg=dict(type='GELU'),
                 linear_pw_conv=True,
                 use_grn=False,
                 drop_path_rate=0.,
                 layer_scale_init_value=1e-6,
                 out_indices=[0, 1, 2, 3],
                 da_reductions=[8, 8, 8],  # 只需要3个值
                 frozen_stages=0,
                 gap_before_final_norm=False,
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

        # 输出配置
        if isinstance(out_indices, int):
            out_indices = [out_indices]
        for i, index in enumerate(out_indices):
            if index < 0:
                out_indices[i] = 4 + index
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.gap_before_final_norm = gap_before_final_norm

        # DA reduction配置简化
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
        self.downsample_layers.append(
            nn.Sequential(build_LayerNorm2d_layer(norm_cfg, self.channels[0]))
        )

        # Stage 1-3 的下采样层（每个做2x下采样）
        for i in range(1, self.num_stages):
            downsample_layer = nn.Sequential(
                build_LayerNorm2d_layer(norm_cfg, self.channels[i-1]),
                nn.Conv2d(self.channels[i-1], self.channels[i], kernel_size=2, stride=2)
            )
            self.downsample_layers.append(downsample_layer)

        # 构建stages
        self.stages = nn.ModuleList()
        for i in range(self.num_stages):
            depth = self.depths[i]
            channels = self.channels[i]

            stage = Sequential(*[
                ConvNeXtBlock(
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

            if i in self.out_indices:
                norm_layer = build_LayerNorm2d_layer(norm_cfg, channels)
                self.add_module(f'norm{i}', norm_layer)

        self._freeze_stages()

    def forward(self, x):
        """
        前向传播
        x: (B, C, H, W) - C=1(SAR), C=3(RGB)
        """
        outs = []

        # 先通过stem（不下采样）
        x = self.stem(x)

        # 然后通过各个stage
        for i, stage in enumerate(self.stages):
            # 先通过downsample层
            x = self.downsample_layers[i](x)
            # 再通过stage中的blocks
            for each_layer in stage:
                x, _ = each_layer(x)

            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                if self.gap_before_final_norm:
                    gap = x.mean([-2, -1], keepdim=True)
                    outs.append(norm_layer(gap).flatten(1))
                else:
                    outs.append(norm_layer(x))

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
        super(ConvNeXt_moe, self).train(mode)
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
class ConvNeXt_DA_MultiInput(ConvNeXt_moe):
    """
    多模态ConvNeXt with DA layers
    继承ConvNeXt_moe并添加Stage间的DA层
    """
    def __init__(self,
                 arch='pico',
                 in_channels=3,
                 stem_patch_size=4,
                 datasets=['sar', 'rgb'],
                 norm_cfg=dict(type='LN2d', eps=1e-6),
                 act_cfg=dict(type='GELU'),
                 linear_pw_conv=True,
                 use_grn=False,
                 drop_path_rate=0.,
                 layer_scale_init_value=1e-6,
                 out_indices=[0, 1, 2, 3],
                 da_reductions=[8, 8, 16],  # Stage间DA的reduction
                 frozen_stages=0,
                 gap_before_final_norm=False,
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
            else:
                raise ValueError(f"Cannot infer modality from {x.shape[1]} channels")
        else:
            dataset_type = datasets[0] if isinstance(datasets, list) else datasets
        batch_size = x.shape[0]
        batch_datasets = [dataset_type] * batch_size

        # 先通过stem（不下采样）
        x = self.stem(x)

        # 然后通过各个stage
        for i, stage in enumerate(self.stages):
            # 先通过downsample层
            x = self.downsample_layers[i](x)

            # 再通过stage中的blocks
            for each_layer in stage:
                x, _ = each_layer(x, batch_datasets)



            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                if self.gap_before_final_norm:
                    gap = x.mean([-2, -1], keepdim=True)
                    outs.append(norm_layer(gap).flatten(1))
                else:
                    outs.append(norm_layer(x))

        return tuple(outs)


# 别名，保持兼容性
MultiModalConvNeXt = ConvNeXt_moe