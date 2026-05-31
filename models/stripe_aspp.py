import torch
import torch.nn as nn
import torch.nn.functional as F


def get_valid_num_groups(num_channels, preferred_groups=8):
    """
    找到能整除num_channels的最大num_groups
    """
    for groups in range(min(preferred_groups, num_channels), 0, -1):
        if num_channels % groups == 0:
            return groups
    return 1


class StripConvolution(nn.Module):
    """条纹卷积 - 使用GroupNorm（正确版本）"""
    def __init__(self, in_channels, out_channels, kernel_size=7):
        super().__init__()
        mid_channels = out_channels // 2

        def find_groups(in_c, out_c):
            import math
            gcd = math.gcd(in_c, out_c)
            return min(gcd, 8)

        groups = find_groups(in_channels, mid_channels)

        # 水平条纹
        self.h_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels,
                     kernel_size=(1, kernel_size),
                     padding=(0, kernel_size//2),
                     groups=groups, bias=False),
            nn.GroupNorm(
                num_groups=get_valid_num_groups(mid_channels, preferred_groups=8),
                num_channels=mid_channels
            ),
            nn.ReLU(inplace=True)
        )

        # 垂直条纹
        self.v_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels,
                     kernel_size=(kernel_size, 1),
                     padding=(kernel_size//2, 0),
                     groups=groups, bias=False),
            nn.GroupNorm(
                num_groups=get_valid_num_groups(mid_channels, preferred_groups=8),
                num_channels=mid_channels
            ),
            nn.ReLU(inplace=True)
        )

        # 融合
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(
                num_groups=get_valid_num_groups(out_channels, preferred_groups=8),
                num_channels=out_channels
            ),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        h_feat = self.h_conv(x)
        v_feat = self.v_conv(x)
        out = torch.cat([h_feat, v_feat], dim=1)
        out = self.fusion(out)
        return out


class LightweightCSAttention(nn.Module):
    """轻量级注意力 - 保持不变"""
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        ca = self.channel_attn(x)
        x = x * ca
        sa = self.spatial_attn(x)
        x = x * sa
        return x


class StripeASPP(nn.Module):
    """
    条纹ASPP - 使用GroupNorm
    """
    def __init__(self, in_channels, out_channels=64, num_classes=1):
        super().__init__()
        aspp_channels = 48

        # 分支1: 1x1卷积
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, aspp_channels, kernel_size=1, bias=False),
            nn.GroupNorm(
                num_groups=get_valid_num_groups(aspp_channels, preferred_groups=8),
                num_channels=aspp_channels
            ),
            nn.ReLU(inplace=True)
        )

        # 分支2: Strip Convolution
        self.branch2_strip = StripConvolution(in_channels, aspp_channels, kernel_size=7)

        # 分支3: 膨胀卷积
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, aspp_channels,
                     kernel_size=3, padding=6, dilation=6, bias=False),
            nn.GroupNorm(
                num_groups=get_valid_num_groups(aspp_channels, preferred_groups=8),
                num_channels=aspp_channels
            ),
            nn.ReLU(inplace=True)
        )

        # 分支4: 全局池化 (已经是GroupNorm，不用改)
        self.branch4_global = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, aspp_channels, kernel_size=1, bias=False),
            nn.GroupNorm(
                num_groups=get_valid_num_groups(aspp_channels, preferred_groups=8),
                num_channels=aspp_channels
            ),
            nn.ReLU(inplace=True)
        )

        # 融合层
        total_channels = aspp_channels * 4
        self.fusion = nn.Sequential(
            nn.Conv2d(total_channels, out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(
                num_groups=get_valid_num_groups(out_channels, preferred_groups=8),
                num_channels=out_channels
            ),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )

        # 注意力
        self.attention = LightweightCSAttention(out_channels, reduction=8)

        # 分类层
        self.classifier = nn.Conv2d(out_channels, num_classes, kernel_size=1)

    def forward(self, x):
        H, W = x.shape[2:]

        feat1 = self.branch1(x)
        feat2 = self.branch2_strip(x)
        feat3 = self.branch3(x)

        feat4 = self.branch4_global(x)
        feat4 = F.interpolate(feat4, size=(H, W), mode='bilinear', align_corners=False)

        concat_feat = torch.cat([feat1, feat2, feat3, feat4], dim=1)
        fused = self.fusion(concat_feat)
        fused = self.attention(fused)
        out = self.classifier(fused)

        return out

