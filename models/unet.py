import torch
import torch.nn as nn
import torch.nn.functional as F
from models.backbone.convnext_moe_DA import ConvNeXt_DA_MultiInput
from models.backbone.unet import UNet
from models.backbone.unet_plusplus import UNetPlusPlus
from models.up_modules import Up
from models.stripe_aspp import StripeASPP
# 新增：引入你定义的 BridgeModule
from models.bridge import BridgeModule


class GlobalAttentionGate(nn.Module):
    """
    全局注意力门控模块 (GAG)
    用于编码器和解码器之间的特征细化
    通过全局信息调制skip connection
    """
    def __init__(self, F_g, F_l, F_int):
        super(GlobalAttentionGate, self).__init__()

        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.GroupNorm(num_groups=get_valid_num_groups(F_int, 8), num_channels=F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.GroupNorm(num_groups=get_valid_num_groups(F_int, 8), num_channels=F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.GroupNorm(num_groups=1, num_channels=1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_weight = nn.Sequential(
            nn.Conv2d(F_l, F_l // 16, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(F_l // 16, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, g, x):
        if g.shape[2:] != x.shape[2:]:
            g = F.interpolate(g, size=x.shape[2:], mode='bilinear', align_corners=False)

        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        global_context = self.global_pool(x)
        global_weight = self.global_weight(global_context)

        attention = psi * (1 + global_weight)
        return x * attention


def get_valid_num_groups(num_channels, preferred_groups=8):
    for groups in range(min(preferred_groups, num_channels), 0, -1):
        if num_channels % groups == 0:
            return groups
    return 1


# ============================== ConvNeXt-UNet (参数共享版本) ==============================
class InternalWaveUNet(nn.Module):
    def __init__(self,
                 n_channels=1,
                 n_classes=1,
                 convnext_arch='pico',
                 da_reductions=[8, 8, 16, 16],
                 dropout_rates=[0.1, 0.15, 0.2, 0.25],
                 use_attention_gate=True,
                 model_type='ConvNeXt'):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.use_attention_gate = use_attention_gate

        # ------------------- ConvNeXt 编码器 -------------------
        if model_type == 'ConvNeXt':
            self.encoder = ConvNeXt_DA_MultiInput(
                arch=convnext_arch,
                in_channels=n_channels,
                out_indices=[0,1,2,3],
            )
            # self.encoder = UNet(arch=convnext_arch, in_channels=n_channels, out_indices=[0,1,2,3])
        elif model_type == 'unet':
            self.encoder = UNet(arch=convnext_arch, in_channels=n_channels, out_indices=[0,1,2,3])
        elif model_type == 'UNetPlusPlus':
            self.encoder = UNetPlusPlus(arch=convnext_arch, in_channels=n_channels, out_indices=[0,1,2,3])
        
        # ------------------- ConvNeXt 各阶段通道 -------------------
        arch_channels = {
            'atto': [40, 80, 160, 320],
            'femto': [48, 96, 192, 384],
            'pico': [64, 128, 256, 512],
            'nano': [80, 160, 320, 640],
            'tiny': [96, 192, 384, 768],
            'small': [96, 192, 384, 768],
            'base':  [128, 256, 512, 1024],
            'swin_large': [192, 384, 768, 1536],
            'large': [192, 384, 768, 1536],
            'xlarge':  [256, 512, 1024, 2048],
            'huge': [352, 704, 1408, 2816]
        }
        encoder_channels = arch_channels.get(convnext_arch, [40, 80, 160, 320])
        decoder_channels = [encoder_channels[0], encoder_channels[1], encoder_channels[2], encoder_channels[3], encoder_channels[3]*2]

        # ------------------- Attention Gate Module -------------------
        if self.use_attention_gate:
            self.attention_gates = nn.ModuleDict({
                'sar': nn.ModuleList([
                    GlobalAttentionGate(encoder_channels[3], encoder_channels[2], encoder_channels[2]//2),
                    GlobalAttentionGate(encoder_channels[2], encoder_channels[1], encoder_channels[1]//2),
                    GlobalAttentionGate(encoder_channels[1], encoder_channels[0], encoder_channels[0]//2),
                ]),
                'rgb': nn.ModuleList([
                    GlobalAttentionGate(encoder_channels[3], encoder_channels[2], encoder_channels[2]//2),
                    GlobalAttentionGate(encoder_channels[2], encoder_channels[1], encoder_channels[1]//2),
                    GlobalAttentionGate(encoder_channels[1], encoder_channels[0], encoder_channels[0]//2),
                ])
            })

        # ------------------- Bridge Layer (修改处：使用外部引入的 BridgeModule) -------------------
        self.bridge = nn.ModuleDict({
            'sar': BridgeModule(encoder_channels[3]),
            'rgb': BridgeModule(encoder_channels[3])
        })

        # ------------------- UNet Decoder -------------------
        self.up1_dict = nn.ModuleDict({
            'sar': Up(encoder_channels[3], encoder_channels[2], dropout_rate=dropout_rates[0]),
            'rgb': Up(encoder_channels[3], encoder_channels[2], dropout_rate=dropout_rates[0])
        })

        self.up2_dict = nn.ModuleDict({
            'sar': Up(encoder_channels[2], encoder_channels[1], dropout_rate=dropout_rates[1]),
            'rgb': Up(encoder_channels[2], encoder_channels[1], dropout_rate=dropout_rates[1])
        })

        self.up3_dict = nn.ModuleDict({
            'sar': Up(encoder_channels[1], encoder_channels[0], dropout_rate=dropout_rates[2]),
            'rgb': Up(encoder_channels[1], encoder_channels[0], dropout_rate=dropout_rates[2])
        })
        
        self.up4_dict = nn.ModuleDict({
            'sar': nn.Sequential(
                nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
                nn.Conv2d(encoder_channels[0], encoder_channels[0], kernel_size=3, padding=1),
                nn.GroupNorm(num_groups=get_valid_num_groups(encoder_channels[0], 8), num_channels=encoder_channels[0]),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout_rates[3])
            ),
            'rgb': nn.Sequential(
                nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
                nn.Conv2d(encoder_channels[0], encoder_channels[0], kernel_size=3, padding=1),
                nn.GroupNorm(num_groups=get_valid_num_groups(encoder_channels[0], 8), num_channels=encoder_channels[0]),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout_rates[3])
            )
        })

        # ------------------- Output Head -------------------
        self.head_dict = nn.ModuleDict({
            'sar': StripeASPP(in_channels=encoder_channels[0], out_channels=64, num_classes=self.n_classes),
            'rgb': StripeASPP(in_channels=encoder_channels[0], out_channels=64, num_classes=self.n_classes),
            'sdg': StripeASPP(in_channels=encoder_channels[0], out_channels=64, num_classes=self.n_classes)
        })

    def forward(self, x, datasets=['sar']):
        encoder_features = self.encoder(x, datasets)
        x1, x2, x3, x4 = encoder_features[0], encoder_features[1], encoder_features[2], encoder_features[3]

        modality = datasets[0] if len(datasets) > 0 else 'sar'

        # ------------------- Bridge Layer (修改处：调用 BridgeModule) -------------------
        # 因为 BridgeModule 内部自带了残差连接 (return x_bridge + x)，所以这里不需要再写一次 + x4
        x4_bridged = self.bridge[modality](x4)

        # ------------------- Attention Gate (可选) -------------------
        if self.use_attention_gate:
            x3_gated = self.attention_gates[modality][0](x4_bridged, x3)
            x2_gated = self.attention_gates[modality][1](x3, x2)
            x1_gated = self.attention_gates[modality][2](x2, x1)
        else:
            x3_gated, x2_gated, x1_gated = x3, x2, x1

        # ------------------- UNet Decoder -------------------
        x = self.up1_dict[modality](x4_bridged, x3_gated)
        x = self.up2_dict[modality](x, x2_gated)
        x = self.up3_dict[modality](x, x1_gated)
        x = self.up4_dict[modality](x) 
        
        logits = self.head_dict[modality](x)
        return logits


# ============================== 创建模型 ==============================
def create_unet(model_type='ConvNeXt',
                num_classes=1,
                convnext_arch='atto',
                da_reductions=[8, 8, 16],
                dropout_rates=[0, 0, 0, 0],
                use_attention_gate=True):
    model = InternalWaveUNet(
        n_channels=1,
        n_classes=num_classes,
        convnext_arch=convnext_arch,
        da_reductions=da_reductions,
        dropout_rates=dropout_rates,
        use_attention_gate=use_attention_gate,
        model_type=model_type
    )
    return model