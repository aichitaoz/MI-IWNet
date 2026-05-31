import torch
import torch.nn as nn
class InceptionDWConv2d(nn.Module):
    def __init__(self, in_channels, square_kernel_size=3, band_kernel_size=11, branch_ratio=0.125):
        super().__init__()

        # 根据branch_ratio计算出分支卷积层的通道数gc
        # 修改：增加超大核分支，所以需要调整通道分配
        # 原本是3个分支(square, band_w, band_h)，现在增加到5个分支
        gc = int(in_channels * branch_ratio * 0.6)  # 减少每个分支的通道数，给超大核腾出空间
        gc_large = int(in_channels * branch_ratio * 0.4)  # 超大核分支的通道数

        """
        kernel_size:指定了卷积核（或滤波器）的大小。它决定了每次卷积操作时覆盖输入张量的区域大小。
                    如果是一个整数，则表示方形卷积核；
                    如果是元组形式如(height, width)，则分别指定高度和宽度方向上的卷积核大小。
        padding :在输入张量边缘周围添加额外的零值，以控制输出张量的空间维度。它可以避免由于卷积操作导致的边界信息丢失。
                    如果是一个整数，则在每个边界的四周均匀地添加相同数量的零；
                    如果是元组形式如(top, bottom, left, right)，则分别指定上、下、左、右四个方向上的填充大小。
                    对于二维卷积，常见的简化形式为(height_padding, width_padding)。
        """
        # 深度可分离卷积层，处理方形区域，使用gc个分组进行卷积
        self.dwconv_hw = nn.Conv2d(gc, gc, square_kernel_size, padding=square_kernel_size // 2, groups=gc)

        # 深度可分离卷积层，处理水平 W 方向上的条形区域，使用gc个分组进行卷积
        self.dwconv_w = nn.Conv2d(gc, gc, kernel_size=(1, band_kernel_size), padding=(0, band_kernel_size // 2),groups=gc)
        # 深度可分离卷积层，处理竖直 H 方向上的条形区域，使用gc个分组进行卷积
        self.dwconv_h = nn.Conv2d(gc, gc, kernel_size=(band_kernel_size, 1), padding=(band_kernel_size // 2, 0),groups=gc)

        # 新增：超大核分支，用于捕获海洋内波的长程条纹特征
        # 使用31x1和1x31的超大核，显著增加感受野
        xlarge_kernel_size = 31  # 可以根据需要调整，建议21-41之间
        self.dwconv_w_xlarge = nn.Conv2d(gc_large, gc_large,
                                         kernel_size=(1, xlarge_kernel_size),
                                         padding=(0, xlarge_kernel_size // 2),
                                         groups=gc_large)
        self.dwconv_h_xlarge = nn.Conv2d(gc_large, gc_large,
                                         kernel_size=(xlarge_kernel_size, 1),
                                         padding=(xlarge_kernel_size // 2, 0),
                                         groups=gc_large)

        # 计算用于拆分输入张量x的索引，以便将x分配到不同的卷积路径中
        # 现在有5个分支：identity, square, band_w, band_h, xlarge_w, xlarge_h
        self.split_indexes = (in_channels - 3 * gc - 2 * gc_large, gc, gc, gc, gc_large, gc_large)

    # 前向传播函数，定义了数据流经网络的方式
    def forward(self, x):
        # 使用预先计算好的索引split_indexes来拆分输入张量x
        # 现在分成6个部分：identity, square, band_w, band_h, xlarge_w, xlarge_h
        x_id, x_hw, x_w, x_h, x_w_xl, x_h_xl = torch.split(x, self.split_indexes, dim=1)

        # 将不同路径处理后的张量沿通道维度拼接起来，并作为最终输出返回
        # 包含了原有的分支和新增的超大核分支
        return torch.cat(
            (x_id,                          # 身份映射（不处理）
             self.dwconv_hw(x_hw),          # 3x3方形卷积
             self.dwconv_w(x_w),            # 1x11水平条纹卷积
             self.dwconv_h(x_h),            # 11x1垂直条纹卷积
             self.dwconv_w_xlarge(x_w_xl),  # 1x31超大水平条纹卷积（新增）
             self.dwconv_h_xlarge(x_h_xl)), # 31x1超大垂直条纹卷积（新增）
            dim=1,
        )
#Exp1
# import torch
# import torch.nn as nn

# class InceptionDWConv2d(nn.Module):
#     """
#     Exp 1: Baseline 版本
#     将所有复杂的非对称分支全部移除，替换为一个最普通的 3x3 卷积。
#     用于向审稿人证明：常规的感受野无法有效捕捉细长的内波特征。
#     """
#     # 保持 __init__ 接口完全一致，防止外部调用报错
#     def __init__(self, in_channels, square_kernel_size=3, band_kernel_size=11, branch_ratio=0.125):
#         super().__init__()

#         # 直接使用一个标准的 3x3 卷积来处理所有通道
#         # padding=1 确保输出特征图的空间尺寸(H, W)与输入保持一致
#         self.baseline_conv = nn.Conv2d(
#             in_channels=in_channels, 
#             out_channels=in_channels, 
#             kernel_size=3, 
#             padding=1,
#             groups=in_channels # 如果你原来的网络极其轻量，可以加上 groups=in_channels 变成深度可分离卷积。这里先用普通卷积作为最强 Baseline。
#         )

#     def forward(self, x):
#         # 没有任何花哨的分支，直接过一遍 3x3 卷积
#         return self.baseline_conv(x)


# =====================================================================
# Exp 2: 消融实验 A (中核) - 去掉超大核分支
# =====================================================================
# class InceptionDWConv2d(nn.Module):
#     """
#     Exp 2: 仅保留 3x3, 1x11, 11x1 分支。
#     证明逻辑：11 的长度不足以捕捉破碎海况下内波的连续长条纹，31 的极度非对称长分支是必须的。
#     """
#     def __init__(self, in_channels, square_kernel_size=3, band_kernel_size=11, branch_ratio=0.125):
#         super().__init__()
        
#         # 恢复到没有超大核时的通道分配
#         gc = int(in_channels * branch_ratio)
        
#         self.dwconv_hw = nn.Conv2d(gc, gc, square_kernel_size, padding=square_kernel_size // 2, groups=gc)
#         self.dwconv_w = nn.Conv2d(gc, gc, kernel_size=(1, band_kernel_size), padding=(0, band_kernel_size // 2), groups=gc)
#         self.dwconv_h = nn.Conv2d(gc, gc, kernel_size=(band_kernel_size, 1), padding=(band_kernel_size // 2, 0), groups=gc)
        
#         # 仅分出 4 个部分：identity, square, band_w, band_h
#         self.split_indexes = (in_channels - 3 * gc, gc, gc, gc)

#     def forward(self, x):
#         x_id, x_hw, x_w, x_h = torch.split(x, self.split_indexes, dim=1)
        
#         return torch.cat(
#             (x_id, 
#              self.dwconv_hw(x_hw), 
#              self.dwconv_w(x_w), 
#              self.dwconv_h(x_h)),
#             dim=1,
#         )


# # =====================================================================
# # Exp 4: 同类竞品对比 (DCNv2 - Deformable Convolution)
# # =====================================================================
# class InceptionDWConv2d(nn.Module):
#     """
#     Exp 4: 将整个模块替换为可变形卷积 (DCN v2)。
#     证明逻辑：虽然 DCN 可以自适应形变，但在 SAR 或光学影像的高频噪声/海杂波干扰下，
#     无约束的形变容易发散。我们固定方向的强烈非对称卷积（条带卷积）对内波特征更稳健。
#     """
#     def __init__(self, in_channels, square_kernel_size=3, band_kernel_size=11, branch_ratio=0.125):
#         super().__init__()
#         # DCNv2 需要先通过普通卷积生成 offset (偏移量) 和 mask (权重掩码)
#         # kernel_size=3 的 DCN 需要 3*3=9 个采样点，每个点需要 x,y 两个偏移量 -> 18 个通道
#         self.offset_conv = nn.Conv2d(in_channels, 2 * 3 * 3, kernel_size=3, padding=1)
#         # mask 需要 9 个通道 (0~1之间的权重)
#         self.mask_conv = nn.Conv2d(in_channels, 3 * 3, kernel_size=3, padding=1)
        
#         # 真正的可变形卷积层
#         self.dcn = ops.DeformConv2d(
#             in_channels=in_channels, 
#             out_channels=in_channels, 
#             kernel_size=3, 
#             padding=1,
#             groups=in_channels # 保持深度可分离的轻量化特性，确保 FLOPs 对比公平
#         )
        
#         # 初始化 offset 和 mask 为 0，使其初始状态退化为普通 3x3 卷积，利于收敛
#         nn.init.constant_(self.offset_conv.weight, 0.)
#         nn.init.constant_(self.offset_conv.bias, 0.)
#         nn.init.constant_(self.mask_conv.weight, 0.)
#         nn.init.constant_(self.mask_conv.bias, 0.)

#     def forward(self, x):
#         offset = self.offset_conv(x)
#         mask = torch.sigmoid(self.mask_conv(x)) # mask 必须在 0~1 之间
#         return self.dcn(x, offset, mask)



# =====================================================================
# Exp 5: 等效大核对比 (Square Large Kernel)
# =====================================================================
# class InceptionDWConv2d(nn.Module):
#     """
#     Exp 5: 替换为等效感受野的大尺寸方形卷积 (31 x 31)。
#     证明逻辑：虽然 31x31 拥有和 1x31 同样远的端到端感受野，
#     但方形大核会引入极其庞大的冗余背景区域（包含大量无用海水信息），
#     同时参数量和 FLOPs 爆炸，导致模型优化困难且指标下降。
#     """
#     def __init__(self, in_channels, square_kernel_size=3, band_kernel_size=11, branch_ratio=0.125):
#         super().__init__()
#         large_k = 31 # 对应你代码里的超大核尺寸
#         pad = large_k // 2 # 确保 padding 后输出尺寸不变，31//2 = 15
        
#         # 采用深度可分离大核卷积 (借鉴 ConvNeXt 思想，否则普通 31x31 显存直接 OOM)
#         self.large_square_conv = nn.Conv2d(
#             in_channels=in_channels, 
#             out_channels=in_channels, 
#             kernel_size=large_k, 
#             padding=pad,
#             groups=in_channels 
#         )
        
#         # 增加一个 1x1 卷积进行通道混合，模拟 DASC 拼接后的特征融合
#         self.pointwise = nn.Conv2d(in_channels, in_channels, kernel_size=1)

#     def forward(self, x):
#         x = self.large_square_conv(x)
#         x = self.pointwise(x)
#         return x