import torch
from thop import profile
from models.unet import create_unet # 确保路径正确
from configs.train_config import Config

def check_complexity():
    config = Config()

    # 1. 初始化你的模型
    if config.MODEL_TYPE == "SegFormer" :
        from models.SegFormer import segformer_b1
        model = segformer_b1(img_size=1024, num_classes=1, stem_channels=32)
    elif config.MODEL_TYPE == "TransUNet" :
        from models.TransUNet import transunet_paper_standard
        model = transunet_paper_standard(img_size=1024, num_classes=1, stem_channels=32)
    elif config.MODEL_TYPE == "IWResNet" :
        from models.IWResNet import iwresnet_base
        model = iwresnet_base(img_size=1024, num_classes=1, stem_channels=32)
    elif config.MODEL_TYPE == "SwinUNet" :
        from models.SwinUNet import swin_unet_base
        model = swin_unet_base(img_size=1024, num_classes=1, stem_channels=32)
    elif config.MODEL_TYPE == "IWENet" :
        from models.IWENet import iwenet_base
        model = iwenet_base(img_size=1024, num_classes=1, stem_channels=32)
    elif config.MODEL_TYPE == "mtu2net" :
        from models.MTU2Net import mtu2net_base
        model = mtu2net_base(num_classes=1, stem_channels=32)
    else:
        from models.unet import create_unet
        model = create_unet(
            config.MODEL_TYPE,
            num_classes=1,
            dropout_rates=[0.0, 0.0, 0.0, 0.0]  # 禁用Dropout
        )
    
    # 2. 创建一个虚拟输入 (Batch size=1, 通道数=3, 高=1024, 宽=1024)
    # 注意：这里的输入尺寸必须和你论文里写的 1024x1024 一致，因为 FLOPs 随分辨率变化
    input_data = torch.randn(1, 3, 1024, 1024)
    
    # 3. 使用 thop 计算
    flops, params = profile(model, inputs=(input_data, ))
    
    # 4. 打印结果（转换为 G 和 M，方便填表）
    print(f"Total FLOPs: {flops / 1e9:.2f} G")
    print(f"Total Parameters: {params / 1e6:.2f} M")

if __name__ == "__main__":
    check_complexity()