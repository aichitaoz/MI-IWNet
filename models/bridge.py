"""
Bridge模块和装饰性模块定义
用于放置在编码器和解码器之间
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_valid_num_groups(num_channels, preferred_groups=8):
    """找到能整除num_channels的最大num_groups"""
    for groups in range(min(preferred_groups, num_channels), 0, -1):
        if num_channels % groups == 0:
            return groups
    return 1


# ============================== 装饰性子模块 ==============================

class  FRB(nn.Module):
    """Time-Frequency Feature Aggregation - 时频特征聚合"""
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.norm = nn.GroupNorm(num_groups=get_valid_num_groups(channels, 8), num_channels=channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        return self.relu(self.norm(self.conv(x))) + x


class  CGB(nn.Module):
    """Adaptive Cross-modal Feature Alignment - 自适应跨模态特征对齐"""
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        attention = self.sigmoid(self.conv(x))
        return x * attention


class MSFB(nn.Module):
    """Multi-Scale Feature Block - 多尺度特征块"""
    def __init__(self, channels):
        super().__init__()
        self.branch1 = nn.Conv2d(channels, channels//4, 1)
        self.branch2 = nn.Conv2d(channels, channels//4, 3, padding=1)
        self.branch3 = nn.Conv2d(channels, channels//4, 5, padding=2)
        self.branch4 = nn.Conv2d(channels, channels//4, 7, padding=3)
        self.fusion = nn.Conv2d(channels, channels, 1)
        self.norm = nn.GroupNorm(num_groups=get_valid_num_groups(channels, 8), num_channels=channels)
        
    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        concat = torch.cat([b1, b2, b3, b4], dim=1)
        return self.norm(self.fusion(concat)) + x


class DSA(nn.Module):
    """Dual Spatial Attention - 双空间注意力"""
    def __init__(self, channels):
        super().__init__()
        self.spatial_conv = nn.Conv2d(2, 1, 7, padding=3)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        concat = torch.cat([avg_pool, max_pool], dim=1)
        attention = self.sigmoid(self.spatial_conv(concat))
        return x * attention


# ============================== Bridge模块 ==============================

class BridgeModule(nn.Module):
    """
    Bridge模块 - 连接编码器最深层和解码器
    使用轻量级深度可分离卷积处理特征
    """
    def __init__(self, channels):
        super().__init__()
        
        self.bridge = nn.Sequential(
            # 第一层：深度可分离卷积
            nn.Conv2d(channels, channels, kernel_size=3,
                     padding=1, groups=channels, bias=False),  # depthwise
            nn.GroupNorm(num_groups=get_valid_num_groups(channels, 8),
                        num_channels=channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),  # pointwise
            nn.GroupNorm(num_groups=get_valid_num_groups(channels, 8),
                        num_channels=channels),
            nn.ReLU(inplace=True),

            # 第二层：深度可分离卷积
            nn.Conv2d(channels, channels, kernel_size=3,
                     padding=1, groups=channels, bias=False),  # depthwise
            nn.GroupNorm(num_groups=get_valid_num_groups(channels, 8),
                        num_channels=channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),  # pointwise
        )
        
    def forward(self, x):
        return self.bridge(x) + x  # 残差连接


# ============================== 装饰性增强模块（放在decoder之前）==============================

class PreDecoderEnhancementModule(nn.Module):
    """
    预解码器增强模块 - 放在Bridge之后、Decoder第一层之前
    包含: Conv 1x1 ->  FRB ->  CGB -> MSFB -> DSA
    """
    def __init__(self, channels):
        super().__init__()
        
        # 1x1 卷积进行特征变换
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.GroupNorm(num_groups=get_valid_num_groups(channels, 8), num_channels=channels),
            nn.ReLU(inplace=True)
        )
        
        #  FRB: 时频特征聚合
        self. FRB =  FRB(channels)
        
        #  CGB: 自适应跨模态特征对齐
        self. CGB =  CGB(channels)
        
        # MSFB: 多尺度特征块
        self.msfb = MSFB(channels)
        
        # DSA: 双空间注意力
        self.dsa = DSA(channels)
        
    def forward(self, x):
        """顺序执行各个模块"""
        x = self.conv1x1(x)
        x = self. FRB(x)
        x = self. CGB(x)
        x = self.msfb(x)
        x = self.dsa(x)
        return x