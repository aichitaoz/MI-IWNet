import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train.main import Config
from models.unet import create_unet

import warnings
warnings.filterwarnings('ignore')


class FeatureExtractor:
    """提取ConvNeXt指定层的特征"""
    
    def __init__(self, model, layer_name):
        self.model = model
        self.layer_name = layer_name
        self.feature = None
        self.hook_handle = None
        self._register_hook()
    
    def _register_hook(self):
        """根据层名注册钩子"""
        # 根据点分隔的层名获取目标层
        layer = self.model
        for attr in self.layer_name.split('.'):
            layer = getattr(layer, attr)
        
        def hook_fn(module, input, output):
            self.feature = output.cpu().detach()
        
        self.hook_handle = layer.register_forward_hook(hook_fn)
        print(f"✅ 已注册钩子到: {self.layer_name}")
    
    def release(self):
        """释放钩子"""
        if self.hook_handle:
            self.hook_handle.remove()


def preprocess_image(image_path, config):
    """读取并预处理RGB图像"""
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    
    if image is None:
        raise ValueError(f"Cannot load image: {image_path}")
    
    # 保存原始图像
    original_image = image.copy()
    
    # resize到模型输入尺寸
    image_resized = cv2.resize(image, (config.IMG_SIZE, config.IMG_SIZE))
    
    # 处理RGB图像
    if image_resized.ndim == 2:
        # 灰度图转RGB
        image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_GRAY2RGB)
    elif image_resized.shape[2] == 3:
        # BGR转RGB
        image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
    elif image_resized.shape[2] == 4:
        # BGRA转RGB
        image_bgr = cv2.cvtColor(image_resized, cv2.COLOR_BGRA2BGR)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    else:
        raise ValueError(f"Unsupported image format: {image_path}")
    
    # [H,W,3] -> [3,H,W]
    input_tensor = torch.from_numpy(image_rgb.transpose(2, 0, 1).astype(np.float32) / 255.0)
    
    # normalize [-1,1]
    input_tensor = (input_tensor - 0.5) / 0.5
    input_tensor = input_tensor.unsqueeze(0)
    
    return input_tensor, image_rgb, original_image


def extract_and_save_features(image_path, save_dir, model, config, device, layer_configs):
    """提取多个层的特征并保存"""
    # 预处理图像
    input_tensor, image_resized, original_image = preprocess_image(image_path, config)
    input_tensor = input_tensor.to(device)
    
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # 为每个层创建提取器
    extractors = []
    for layer_name, _ in layer_configs:
        extractor = FeatureExtractor(model, layer_name)
        extractors.append((layer_name, extractor))
    
    # 前向传播（一次前向传播提取所有层）
    with torch.no_grad():
        _ = model(input_tensor, datasets=['rgb'])
    
    # 处理每个层的特征
    for (layer_name, extractor), (_, save_name) in zip(extractors, layer_configs):
        feature = extractor.feature
        
        if feature is None:
            print(f"  ⚠️  {layer_name}: No output captured")
            continue
        
        # 处理特征图 [B, C, H, W] -> [H, W]
        feat_map = feature[0].numpy()
        
        # 如果有多个通道，对通道求平均
        if feat_map.ndim == 3:
            feat_map_mean = np.mean(feat_map, axis=0)
        else:
            feat_map_mean = feat_map
        
        # 归一化到 [0, 1]
        feat_map_norm = (feat_map_mean - feat_map_mean.min()) / (feat_map_mean.max() - feat_map_mean.min() + 1e-8)
        
        # 应用viridis配色
        cmap = plt.cm.get_cmap('viridis')
        feat_color = cmap(feat_map_norm)[:, :, :3]
        
        # 创建保存目录
        layer_save_dir = os.path.join(save_dir, save_name)
        os.makedirs(layer_save_dir, exist_ok=True)
        
        # 保存纯特征热力图（放大到原图尺寸）
        feat_resized = cv2.resize(feat_color, (original_image.shape[1], original_image.shape[0]))
        plt.imsave(
            os.path.join(layer_save_dir, f"{base_name}_heatmap.png"),
            feat_resized
        )
        
        print(f"  ✅ {save_name:20s} | Shape: {tuple(feature.shape)}")
    
    # 释放所有钩子
    for _, extractor in extractors:
        extractor.release()


def batch_extract_features(image_dir, save_dir, model, config, device, layer_configs):
    """批量处理文件夹"""
    image_files = sorted([f for f in os.listdir(image_dir) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff'))])
    
    print(f"\n📁 Found {len(image_files)} images in {image_dir}\n")
    
    for i, img_file in enumerate(image_files, 1):
        image_path = os.path.join(image_dir, img_file)
        print(f"[{i}/{len(image_files)}] Processing {img_file}...")
        
        try:
            extract_and_save_features(image_path, save_dir, model, config, device, layer_configs)
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    print(f"\n🎉 All done! Results saved to {save_dir}")


def load_model(model_path, config):
    """加载 ConvNeXt 模型"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(model_path, map_location=device)
    
    # 创建模型
    model = create_unet(
        'ConvNeXt',
        num_classes=1,
        dropout_rates=[0.0, 0.0, 0.0, 0.0]
    )
    
    # 加载权重
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
        print("✅ Model loaded successfully (strict mode)")
    except RuntimeError:
        print("⚠️ Loading with strict=False")
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    
    model.to(device).eval()
    return model, device


def main():
    print("\n" + "=" * 80)
    print("🔥 ConvNeXt RGB Multi-Layer Feature Extraction")
    print("=" * 80 + "\n")
    
    config = Config()
    
    # 模型路径
    model_path = 'checkpoints/pth_rgb/best_ConvNeXt_model.pth'
    
    # RGB数据路径
    image_dir = '/home/xiaobowen/project/internal_wave_detection_project/IW_data/images_rgb/'
    save_dir = f'/home/xiaobowen/project/internal_wave_detection_project/IW_data/features_onlyRGB/ConvNeXt_RGB/'
    
    # ============ 配置要提取的层（和SAR一样）============
    # 格式: (层的完整路径, 保存文件夹名称)
    layer_configs = [
        # Encoder 各阶段输出
        ('encoder.norm0', '1_encoder_stage1'),      # (1, 40, 256, 256)
        ('encoder.norm1', '2_encoder_stage2'),      # (1, 80, 128, 128)
        ('encoder.norm2', '3_encoder_stage3'),      # (1, 160, 64, 64)
        ('encoder.norm3', '4_encoder_stage4'),      # (1, 320, 32, 32) - 最深层
        
        # Head fusion 输出（分割头fusion后的特征）
        ('head_dict.rgb.fusion', '5_head_fusion'),  # RGB分支fusion输出
    ]
    # =====================================
    
    # 加载模型
    print("🚀 Loading ConvNeXt RGB model...\n")
    model, device = load_model(model_path, config)
    
    # 批量处理
    batch_extract_features(image_dir, save_dir, model, config, device, layer_configs)


if __name__ == '__main__':
    main()