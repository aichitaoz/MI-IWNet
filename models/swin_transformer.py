"""
SegFormer Model for Oceanic Internal Wave Segmentation in SAR Images
Based on: "SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers"
Adapted for parameter-shared multi-modal training (SAR/RGB)

Architecture:
- Input Adapter: UnifiedMultiModalStem - converts different modalities to unified channels
- Encoder: Hierarchical Mix Transformer (MiT) - shared parameters across modalities
- Decoder: Lightweight All-MLP decoder - fuses multi-scale features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial


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
    修复版 - 使用LayerNorm替代BatchNorm
    
    优势：
    1. ✅ 不依赖batch统计量
    2. ✅ 每个样本独立归一化
    3. ✅ 训练/推理完全一致
    4. ✅ 支持不同模态混合训练
    """
    def __init__(self, out_channels):
        super().__init__()
        
        # 通道对齐
        self.align_sar = nn.Conv2d(1, 3, kernel_size=1, bias=False)
        # RGB：添加一个1x1卷积进行通道混合和归一化，而不是直接Identity
        self.align_rgb = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=3, num_channels=3)  # 对每个通道单独归一化
        )
        self.align_sdg = nn.Conv2d(7, 3, kernel_size=1, bias=False)
        
        # ✅ 共享卷积 + LayerNorm
        self.shared_conv = nn.Sequential(
            nn.Conv2d(3, out_channels, kernel_size=7, stride=1, padding=3, bias=False),
            LayerNorm2d(out_channels),
            nn.GELU()
        )
    
    def forward(self, x):
        channels = x.shape[1]
        
        if channels == 1:
            x = self.align_sar(x)
        elif channels == 3:
            x = self.align_rgb(x)
        elif channels == 7:
            x = self.align_sdg(x)
        else:
            raise ValueError(f"Unsupported channels: {channels}, only support 1(SAR)/3(RGB)/7(SDG)")
        
        return self.shared_conv(x)


class DWConv(nn.Module):
    """Depthwise Convolution for Mix-FFN"""
    def __init__(self, dim=768):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class Mlp(nn.Module):
    """MLP with GELU activation"""
    def __init__(self, in_features, hidden_features=None, out_features=None, 
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    """Efficient Multi-Head Self Attention with Sequence Reduction"""
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, 
                 attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):
    """Transformer Block with Efficient Attention and Mix-FFN"""
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, 
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, 
                 norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        
        self.drop_path = nn.Identity() if drop_path <= 0. else nn.Dropout(drop_path)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, 
                       act_layer=act_layer, drop=drop)

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x


class OverlapPatchEmbed(nn.Module):
    """Overlapping Patch Embedding with stride"""
    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = (img_size, img_size) if isinstance(img_size, int) else img_size
        patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // stride, img_size[1] // stride
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, 
                              stride=stride, padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W


class MixTransformer(nn.Module):
    """Mix Transformer Encoder (Hierarchical) - Shared across all modalities"""
    def __init__(self, img_size=512, in_chans=64, num_classes=1, 
                 embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], 
                 mlp_ratios=[4, 4, 4, 4], 
                 qkv_bias=False, qk_scale=None, 
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 norm_layer=nn.LayerNorm, depths=[3, 4, 6, 3], 
                 sr_ratios=[8, 4, 2, 1]):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths

        # Patch embeddings for each stage
        # Note: Stage 1 now takes in_chans from UnifiedMultiModalStem output
        self.patch_embed1 = OverlapPatchEmbed(img_size=img_size, patch_size=7, stride=4, 
                                              in_chans=in_chans, embed_dim=embed_dims[0])
        self.patch_embed2 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, 
                                              in_chans=embed_dims[0], embed_dim=embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, 
                                              in_chans=embed_dims[1], embed_dim=embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(img_size=img_size // 16, patch_size=3, stride=2, 
                                              in_chans=embed_dims[2], embed_dim=embed_dims[3])

        # Transformer blocks for each stage
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        
        self.block1 = nn.ModuleList([Block(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], 
            qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, 
            drop_path=dpr[cur + i], norm_layer=norm_layer, sr_ratio=sr_ratios[0])
            for i in range(depths[0])])
        self.norm1 = norm_layer(embed_dims[0])

        cur += depths[0]
        self.block2 = nn.ModuleList([Block(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], 
            qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, 
            drop_path=dpr[cur + i], norm_layer=norm_layer, sr_ratio=sr_ratios[1])
            for i in range(depths[1])])
        self.norm2 = norm_layer(embed_dims[1])

        cur += depths[1]
        self.block3 = nn.ModuleList([Block(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], 
            qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, 
            drop_path=dpr[cur + i], norm_layer=norm_layer, sr_ratio=sr_ratios[2])
            for i in range(depths[2])])
        self.norm3 = norm_layer(embed_dims[2])

        cur += depths[2]
        self.block4 = nn.ModuleList([Block(
            dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], 
            qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, 
            drop_path=dpr[cur + i], norm_layer=norm_layer, sr_ratio=sr_ratios[3])
            for i in range(depths[3])])
        self.norm4 = norm_layer(embed_dims[3])

    def forward(self, x):
        B = x.shape[0]
        outs = []

        # Stage 1
        x, H, W = self.patch_embed1(x)
        for blk in self.block1:
            x = blk(x, H, W)
        x = self.norm1(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # Stage 2
        x, H, W = self.patch_embed2(x)
        for blk in self.block2:
            x = blk(x, H, W)
        x = self.norm2(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # Stage 3
        x, H, W = self.patch_embed3(x)
        for blk in self.block3:
            x = blk(x, H, W)
        x = self.norm3(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # Stage 4
        x, H, W = self.patch_embed4(x)
        for blk in self.block4:
            x = blk(x, H, W)
        x = self.norm4(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        return outs


class MLPDecoder(nn.Module):
    """All-MLP Decoder for SegFormer"""
    def __init__(self, in_channels=[64, 128, 256, 512], embedding_dim=256, num_classes=1):
        super().__init__()
        
        # Linear layers to unify channel dimensions
        self.linear_c4 = nn.Linear(in_channels[3], embedding_dim)
        self.linear_c3 = nn.Linear(in_channels[2], embedding_dim)
        self.linear_c2 = nn.Linear(in_channels[1], embedding_dim)
        self.linear_c1 = nn.Linear(in_channels[0], embedding_dim)
        
        # Fusion layer
        self.linear_fuse = nn.Sequential(
            nn.Conv2d(embedding_dim * 4, embedding_dim, 1, bias=False),
            nn.BatchNorm2d(embedding_dim),
            nn.ReLU(inplace=True),
        )
        
        # Final prediction layer
        self.linear_pred = nn.Conv2d(embedding_dim, num_classes, 1)
        self.dropout = nn.Dropout2d(0.1)

    def forward(self, inputs):
        c1, c2, c3, c4 = inputs
        
        # Get spatial dimensions
        n, _, h, w = c4.shape
        
        # Unify all features to H/4 x W/4 resolution
        _c4 = self.linear_c4(c4.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        _c4 = F.interpolate(_c4, size=c1.size()[2:], mode='bilinear', align_corners=False)
        
        _c3 = self.linear_c3(c3.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        _c3 = F.interpolate(_c3, size=c1.size()[2:], mode='bilinear', align_corners=False)
        
        _c2 = self.linear_c2(c2.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        _c2 = F.interpolate(_c2, size=c1.size()[2:], mode='bilinear', align_corners=False)
        
        _c1 = self.linear_c1(c1.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        
        # Concatenate and fuse
        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))
        _c = self.dropout(_c)
        
        # Final prediction
        x = self.linear_pred(_c)
        
        return x


class SegFormer(nn.Module):
    """
    SegFormer Model for Internal Wave Segmentation with Parameter Sharing
    
    特点：
    1. 单一编码器架构（参数共享）
    2. 通过 UnifiedMultiModalStem 统一不同模态输入
    3. 支持交替批次训练（每个batch单一模态）
    """
    def __init__(self, 
                 img_size=512,
                 num_classes=1,
                 stem_channels=64,  # UnifiedMultiModalStem输出通道数
                 embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8],
                 mlp_ratios=[4, 4, 4, 4],
                 qkv_bias=False,
                 depths=[3, 4, 6, 3],
                 sr_ratios=[8, 4, 2, 1],
                 drop_rate=0.0,
                 attn_drop_rate=0.0,
                 drop_path_rate=0.1,
                 decoder_dim=256):
        """
        Args:
            img_size: Input image size (default: 512)
            num_classes: Number of output classes (default: 1 for binary segmentation)
            stem_channels: Output channels from UnifiedMultiModalStem
            embed_dims: Embedding dimensions for each stage
            num_heads: Number of attention heads for each stage
            mlp_ratios: MLP expansion ratio for each stage
            depths: Number of transformer blocks for each stage
            sr_ratios: Sequence reduction ratios for efficient attention
            decoder_dim: Decoder embedding dimension
        """
        super().__init__()
        
        # ✅ 输入适配层：将不同模态统一到相同通道数
        self.input_adapter = UnifiedMultiModalStem(out_channels=stem_channels)
        
        # ✅ 共享编码器：所有模态共用同一个编码器
        self.encoder = MixTransformer(
            img_size=img_size,
            in_chans=stem_channels,  # 接收UnifiedMultiModalStem的输出
            num_classes=num_classes,
            embed_dims=embed_dims,
            num_heads=num_heads,
            mlp_ratios=mlp_ratios,
            qkv_bias=qkv_bias,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            depths=depths,
            sr_ratios=sr_ratios
        )
        
        # 共享解码器
        self.decoder = MLPDecoder(
            in_channels=embed_dims,
            embedding_dim=decoder_dim,
            num_classes=num_classes
        )

    def forward(self, x, datasets=None):
        """
        Forward pass with parameter-shared multi-modal support
        
        Args:
            x: Input tensor [B, C, H, W] where C=1 (SAR) or C=3 (RGB) or C=7 (SDG)
            datasets: List of dataset types (e.g., ["sar"] or ["rgb"])
                      Currently not used for routing, kept for API compatibility
        
        Returns:
            Output segmentation mask [B, num_classes, H, W]
        """
        B, C, H, W = x.shape
        
        # ✅ Step 1: 通过输入适配层统一通道数
        #    自动根据输入通道数选择正确的对齐分支
        x = self.input_adapter(x)  # [B, C, H, W] -> [B, stem_channels, H, W]
        
        # ✅ Step 2: 共享编码器提取多尺度特征
        encoder_features = self.encoder(x)  # 返回4个尺度的特征
        
        # ✅ Step 3: 解码器融合特征并生成分割掩码
        output = self.decoder(encoder_features)
        
        # ✅ Step 4: 上采样到原始分辨率
        output = F.interpolate(output, size=(H, W), mode='bilinear', align_corners=False)
        
        return output


# Model factory functions for different variants
def segformer_b0(img_size=512, num_classes=1, stem_channels=64):
    """SegFormer-B0: Lightweight variant"""
    return SegFormer(
        img_size=img_size,
        num_classes=num_classes,
        stem_channels=stem_channels,
        embed_dims=[32, 64, 160, 256],
        num_heads=[1, 2, 5, 8],
        mlp_ratios=[4, 4, 4, 4],
        depths=[2, 2, 2, 2],
        sr_ratios=[8, 4, 2, 1],
        decoder_dim=256
    )


def segformer_b1(img_size=512, num_classes=1, stem_channels=64):
    """SegFormer-B1: Balanced variant (recommended for internal waves)"""
    return SegFormer(
        img_size=img_size,
        num_classes=num_classes,
        stem_channels=stem_channels,
        embed_dims=[64, 128, 320, 512],
        num_heads=[1, 2, 5, 8],
        mlp_ratios=[4, 4, 4, 4],
        depths=[2, 2, 2, 2],
        sr_ratios=[8, 4, 2, 1],
        decoder_dim=256
    )


def segformer_b2(img_size=512, num_classes=1, stem_channels=64):
    """SegFormer-B2: Medium variant"""
    return SegFormer(
        img_size=img_size,
        num_classes=num_classes,
        stem_channels=stem_channels,
        embed_dims=[64, 128, 320, 512],
        num_heads=[1, 2, 5, 8],
        mlp_ratios=[4, 4, 4, 4],
        depths=[3, 4, 6, 3],
        sr_ratios=[8, 4, 2, 1],
        decoder_dim=768
    )


def segformer_b3(img_size=512, num_classes=1, stem_channels=64):
    """SegFormer-B3: Large variant"""
    return SegFormer(
        img_size=img_size,
        num_classes=num_classes,
        stem_channels=stem_channels,
        embed_dims=[64, 128, 320, 512],
        num_heads=[1, 2, 5, 8],
        mlp_ratios=[4, 4, 4, 4],
        depths=[3, 4, 18, 3],
        sr_ratios=[8, 4, 2, 1],
        decoder_dim=768
    )


def segformer_b4(img_size=512, num_classes=1, stem_channels=64):
    """SegFormer-B4: Very large variant"""
    return SegFormer(
        img_size=img_size,
        num_classes=num_classes,
        stem_channels=stem_channels,
        embed_dims=[64, 128, 320, 512],
        num_heads=[1, 2, 5, 8],
        mlp_ratios=[4, 4, 4, 4],
        depths=[3, 8, 27, 3],
        sr_ratios=[8, 4, 2, 1],
        decoder_dim=768
    )


def segformer_b5(img_size=512, num_classes=1, stem_channels=64):
    """SegFormer-B5: Largest variant"""
    return SegFormer(
        img_size=img_size,
        num_classes=num_classes,
        stem_channels=stem_channels,
        embed_dims=[64, 128, 320, 512],
        num_heads=[1, 2, 5, 8],
        mlp_ratios=[4, 4, 4, 4],
        depths=[3, 6, 40, 3],
        sr_ratios=[8, 4, 2, 1],
        decoder_dim=768
    )


# Example usage
if __name__ == "__main__":
    # Create model (B1 variant recommended)
    model = segformer_b5(img_size=1024, num_classes=1, stem_channels=64)
    
    print("="*60)
    print("SegFormer with Parameter Sharing (UnifiedMultiModalStem)")
    print("="*60)
    
    # Test with SAR input (1 channel)
    sar_input = torch.randn(2, 1, 1024, 1024)
    output_sar = model(sar_input, datasets=["sar"])
    print(f"✅ SAR Input: {sar_input.shape} -> Output: {output_sar.shape}")
    
    # Test with RGB input (3 channels)
    rgb_input = torch.randn(2, 3, 1024, 1024)
    output_rgb = model(rgb_input, datasets=["rgb"])
    print(f"✅ RGB Input: {rgb_input.shape} -> Output: {output_rgb.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n📊 Total parameters: {total_params:,}")
    print(f"📊 Trainable parameters: {trainable_params:,}")
    
    # Show architecture
    print(f"\n🏗️  Architecture:")
    print(f"   - Input Adapter: UnifiedMultiModalStem (1/3/7 -> 64 channels)")
    print(f"   - Encoder: Shared MixTransformer (all modalities)")
    print(f"   - Decoder: Shared MLPDecoder")
    print(f"\n✨ 训练方式: 交替批次 (SAR, RGB, RGB, ...)")