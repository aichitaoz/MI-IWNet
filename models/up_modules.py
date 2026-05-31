import torch
import torch.nn as nn
import torch.nn.functional as F


def get_valid_num_groups(num_channels, preferred_groups=8):
    """找到能整除num_channels的最大num_groups"""
    for groups in range(min(preferred_groups, num_channels), 0, -1):
        if num_channels % groups == 0:
            return groups
    return 1


class FAM(nn.Module):
    """
    Decoder Skip-connection Alignment (FAM)
    解码器跳跃连接对齐模块
    """
    def __init__(self, channels):
        super().__init__()
        
        # 空间注意力：强调重要区域
        self.spatial_att = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )
        
        # 特征校准：调整特征分布
        self.channel_refine = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=get_valid_num_groups(channels, 8), 
                        num_channels=channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # 空间注意力
        spatial_weight = self.spatial_att(x)
        x_att = x * spatial_weight
        
        # 特征校准
        x_refined = self.channel_refine(x_att)
        
        # 残差连接
        return x_refined + x


class CA(nn.Module):
    """
    Channel Attention (CA)
    通道注意力模块 - 使用gate引导（无残差）
    """
    def __init__(self, channels, reduction=16):
        super().__init__()
        
        # 全局信息聚合
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # 通道权重生成
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x, gate):
        # gate引导，融合gate信息
        if gate.shape[2:] != x.shape[2:]:
            gate = F.interpolate(gate, size=x.shape[2:], 
                                mode='bilinear', align_corners=False)
        x = x + gate * 0.5
        
        # 双路池化
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        
        # 生成通道权重（无残差）
        weight = self.sigmoid(avg_out + max_out)
        
        return x * weight


class DA(nn.Module):
    """
    Decoder Attention (DA)
    解码器注意力 - 使用gate引导skip connection（无残差）
    """
    def __init__(self, channels):
        super().__init__()
        
        # Gate处理
        self.gate_conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=get_valid_num_groups(channels, 8),
                        num_channels=channels),
            nn.Sigmoid()
        )
        
        # 特征增强
        self.enhance = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=get_valid_num_groups(channels, 8),
                        num_channels=channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x, gate):
        # 使用gate生成attention权重
        gate_weight = self.gate_conv(gate)
        
        # Gate引导的特征选择（无残差）
        x_gated = x * gate_weight
        
        # 特征增强
        x_enhanced = self.enhance(x_gated)
        
        return x_enhanced


class RSM(nn.Module):
    """
    Residual Selective Mechanism (RSM)
    残差选择机制
    """
    def __init__(self, channels):
        super().__init__()
        
        # 特征选择网络
        self.select = nn.Sequential(
            nn.Conv2d(channels, channels // 4, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        
        # 残差细化
        self.refine = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=get_valid_num_groups(channels, 8),
                        num_channels=channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        select_weight = self.select(x)
        x_selected = x * select_weight
        x_refined = self.refine(x_selected)
        return x_refined + x


class WeightedContact(nn.Module):
    """
    Weighted Contact (WC)
    加权融合模块
    """
    def __init__(self, channels):
        super().__init__()
        
        # 学习融合权重
        self.weight_net = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=get_valid_num_groups(channels, 8),
                        num_channels=channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 2, kernel_size=1),
            nn.Softmax(dim=1)
        )
        
    def forward(self, x1, x2):
        concat_feat = torch.cat([x1, x2], dim=1)
        weights = self.weight_net(concat_feat)
        
        w1 = weights[:, 0:1, :, :]
        w2 = weights[:, 1:2, :, :]
        
        out = x1 * w1 + x2 * w2
        return out

# class WeightedContact(nn.Module):
#     """
#     消融版本：退化为简单的特征相加 (Summation)
#     """
#     def __init__(self, channels):
#         super().__init__()
#         # 即使不做融合，也保留参数，保证模型结构定义的参数量一致性（可选）
#         # 或者直接留空，只保留接口
#         pass
        
#     def forward(self, x1, x2):
#         # 接口保持不变，但内部逻辑改为直接相加
#         return x1 + x2
        
# ============================== 升级版 Up 模块 ==============================
class Up(nn.Module):
    """
    升级版上采样模块
    流程: 
    - x1: Upsample → FAM → CA (使用x2作为gate)
    - x2: DA (使用x2自身作为gate) → RSM
    - 融合: WC → Final Conv
    """
    def __init__(self, in_channels, out_channels, dropout_rate=0.0, use_FAM=True):
        super().__init__()
        
        # 上采样
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )
        
        # 左路：FAM + CA
        self.use_FAM = use_FAM
        if use_FAM:
            self.FAM = FAM(channels=out_channels)
        self.ca = CA(channels=out_channels)
        
        # 右路：DA + RSM
        self.da = DA(channels=out_channels)
        self.rsm = RSM(channels=out_channels)
        
        # 加权融合
        self.wc = WeightedContact(channels=out_channels)
        
        # 最终处理
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(
                num_groups=get_valid_num_groups(out_channels, 8),
                num_channels=out_channels
            ),
            nn.ReLU(inplace=True)
        )

        # Dropout
        self.use_dropout = dropout_rate > 0
        if self.use_dropout:
            self.dropout = nn.Dropout2d(p=dropout_rate)

    def forward(self, x1, x2):
        """
        Args:
            x1: 来自deeper layer的特征 [B, C_in, H, W]
            x2: skip connection特征 [B, C_out, H*2, W*2] (也作为gate)
        
        Returns:
            out: 融合后的特征 [B, C_out, H*2, W*2]
        """
        # === 左路：x1 上采样 → FAM → CA ===
        x1 = self.up(x1)  # [B, C_out, H*2, W*2]
        
        if self.use_FAM:
            x1 = self.FAM(x1)  # FAM对齐
        
        x1 = self.ca(x1, gate=x2)  # CA通道注意力（x2作为gate）
        
        # === 右路：x2 → DA → RSM ===
        x2_processed = self.da(x2, gate=x2)  # DA（x2自身作为gate）
        x2_processed = self.rsm(x2_processed)  # RSM残差选择
        
        # 尺寸对齐（以防万一）
        diffY = x2_processed.size()[2] - x1.size()[2]
        diffX = x2_processed.size()[3] - x1.size()[3]
        if diffY != 0 or diffX != 0:
            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2])
        
        # === 加权融合 ===
        out = self.wc(x1, x2_processed)
        
        # === 最终卷积 ===
        out = self.conv(out)

        # Dropout
        if self.use_dropout:
            out = self.dropout(out)

        return out