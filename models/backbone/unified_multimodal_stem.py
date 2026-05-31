import torch
import torch.nn as nn

class LayerNorm2d(nn.Module):
    """
    2D LayerNorm - 对每个样本独立归一化
    不依赖batch统计量，完美适配batch_size=1
    """
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x):
        # 在channel, H, W维度归一化（不涉及batch）
        u = x.mean(1, keepdim=True)  # (B, 1, H, W)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class UnifiedMultiModalStem(nn.Module):
    """
    4x下采样版本 - 大幅减少显存
    """
    def __init__(self, out_channels):
        super().__init__()
        
        # 通道对齐
        self.align_sar = nn.Conv2d(1, 3, kernel_size=1, bias=False)
        self.align_rgb = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=3, num_channels=3)
        )
        
        # ✅ 改为stride=4，一步到位4x下采样
        self.shared_conv = nn.Sequential(
            nn.Conv2d(3, out_channels, 
                     kernel_size=7, 
                     stride=4,  # ← 从stride=1改为stride=4
                     padding=3, 
                     bias=False),
            LayerNorm2d(out_channels),
            nn.GELU()
        )
    
    def forward(self, x):
        channels = x.shape[1]
        
        if channels == 1:
            x = self.align_sar(x)
        elif channels == 3:
            x = self.align_rgb(x)
        else:
            raise ValueError(f"Unsupported channels: {channels}")
        
        return self.shared_conv(x)
