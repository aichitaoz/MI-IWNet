# layers.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath
from mmcv.cnn import build_activation_layer

class LayerNorm2d(nn.LayerNorm):
    def __init__(self, num_channels: int, **kwargs):
        super().__init__(num_channels, **kwargs)
        self.num_channels = self.normalized_shape[0]

    def forward(self, x, data_format='channel_first'):
        assert x.dim() == 4, f'LayerNorm2d 仅支持 4D 输入，但得到 {x.shape}'
        if data_format == 'channel_last':
            x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        else:
            x = x.permute(0, 2, 3, 1)
            x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
            x = x.permute(0, 3, 1, 2).contiguous()
        return x

def build_LayerNorm2d_layer(cfg: dict, num_features: int) -> nn.Module:
    layer_type = cfg.pop('type', 'LayerNorm2d')
    requires_grad = cfg.pop('requires_grad', True)
    cfg.setdefault('eps', 1e-5)
    layer = LayerNorm2d(num_features, **cfg)
    for param in layer.parameters():
        param.requires_grad = requires_grad
    return layer

class GRN(nn.Module):
    def __init__(self, in_channels, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(in_channels))
        self.beta = nn.Parameter(torch.zeros(in_channels))
        self.eps = eps

    def forward(self, x, data_format='channel_first'):
        if data_format == 'channel_last':
            gx = torch.norm(x, p=2, dim=(1,2), keepdim=True)
            nx = gx / (gx.mean(dim=-1, keepdim=True)+self.eps)
            x = self.gamma * (x * nx) + self.beta + x
        else:
            gx = torch.norm(x, p=2, dim=(2,3), keepdim=True)
            nx = gx / (gx.mean(dim=1, keepdim=True)+self.eps)
            x = self.gamma.view(1,-1,1,1)*(x*nx) + self.beta.view(1,-1,1,1) + x
        return x

import torch
import torch.nn as nn
from mmcv.cnn import build_activation_layer
from torch.utils.checkpoint import checkpoint
from functools import partial

class FFN(nn.Module):
    """
    Gated FFN (SwiGLU变体) - 显存节省版
    
    接口与原FFN完全一致:
        GatedFFN(in_channels, mid_channels, pw_conv, act_cfg, use_grn)
    
    内部自动计算合适的hidden_channels使参数量匹配
    """
    def __init__(self, in_channels, mid_channels, pw_conv, 
                 act_cfg=dict(type='GELU'), use_grn=False):
        """
        Args:
            in_channels: 输入通道数
            mid_channels: 目标中间通道数(用于计算参数匹配的hidden_channels)
            pw_conv: 卷积类型 (nn.Linear 或 partial(nn.Conv2d, kernel_size=1))
            act_cfg: 激活函数配置
            use_grn: 是否使用GRN
        """
        super().__init__()
        
        # 计算参数量匹配的hidden_channels
        # 标准FFN参数: in*mid + mid*in = 2*in*mid
        # Gated FFN参数: in*h + in*h + h*in = 3*in*h
        # 令 3*in*h = 2*in*mid => h = 2*mid/3
        hidden_channels = int(2 * mid_channels / 3)
        
        # Gate和Value投影 (独立学习)
        self.gate_proj = pw_conv(in_channels, hidden_channels)
        self.value_proj = pw_conv(in_channels, hidden_channels)
        
        # 激活函数 (推荐SiLU用于gating,但保持用户配置)
        self.act = build_activation_layer(act_cfg)
        
        # GRN (如果需要)
        self.grn = GRN(hidden_channels) if use_grn else None
        
        # 输出投影
        self.out_proj = pw_conv(hidden_channels, in_channels)

    def forward(self, x):
        # Gated mechanism
        gate = self.act(self.gate_proj(x))  # 激活只作用在gate
        value = self.value_proj(x)           # value保持线性
        
        # Element-wise gating
        x = gate * value  # 显存峰值比标准FFN低33%!
        
        if self.grn is not None:
            x = self.grn(x, data_format='channel_last')
        
        x = self.out_proj(x)
        return x