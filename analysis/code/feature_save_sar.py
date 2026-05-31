import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import warnings

# 忽略警告
warnings.filterwarnings('ignore')

# 确保路径正确
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 尝试导入项目配置和模型创建函数
try:
    from train.main import Config
    from models.unet import create_unet
except ImportError:
    print("⚠️ 无法导入自定义模块，请确保脚本在项目根目录下运行。")

class FeatureExtractor:
    """提取模型指定层的特征"""
    def __init__(self, model, layer_name):
        self.model = model
        self.layer_name = layer_name
        self.feature = None
        self.hook_handle = None
        self._register_hook()
    
    def _register_hook(self):
        layer = self.model
        # 处理嵌套层名，如 'encoder.norm0'
        for attr in self.layer_name.split('.'):
            layer = getattr(layer, attr)
        
        def hook_fn(module, input, output):
            # output 可能是一个 tuple (取决于 forward 的返回)，这里取第一个元素
            if isinstance(output, tuple):
                self.feature = output[0].cpu().detach()
            else:
                self.feature = output.cpu().detach()
        
        self.hook_handle = layer.register_forward_hook(hook_fn)
        print(f"✅ 已注册钩子到: {self.layer_name}")
    
    def release(self):
        if self.hook_handle:
            self.hook_handle.remove()

def preprocess_image(image_path, config):
    """读取并预处理图像"""
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Cannot load image: {image_path}")
    
    # 将输入图 Resize 到模型配置的输入尺寸 (如 1024x1024)
    image_resized = cv2.resize(image, (config.IMG_SIZE, config.IMG_SIZE))
    
    # 处理通道
    if image_resized.ndim == 3:
        image_resized = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
    
    # 归一化处理
    if image_resized.dtype == np.uint16:
        image_resized = (image_resized / 65535.0 * 255.0).astype(np.uint8)
    
    # 转为 Tensor 并标准化
    input_tensor = torch.from_numpy(image_resized.astype(np.float32)/255.0).unsqueeze(0)
    input_tensor = (input_tensor - 0.5) / 0.5
    input_tensor = input_tensor.unsqueeze(0) # [1, 1, H, W]
    
    return input_tensor

def extract_and_save_features(image_path, save_dir, model, config, device, layer_configs):
    """提取特征并以真实尺寸保存"""
    input_tensor = preprocess_image(image_path, config).to(device)
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # 注册提取器
    extractors = []
    for layer_name, _ in layer_configs:
        extractor = FeatureExtractor(model, layer_name)
        extractors.append((layer_name, extractor))
    
    # 前向传播
    with torch.no_grad():
        # 这里保持接口兼容，传入 datasets 参数
        _ = model(input_tensor, datasets=['sar'])
    
    # 保存每一层的特征
    for (layer_name, extractor), (_, folder_name) in zip(extractors, layer_configs):
        feature = extractor.feature
        if feature is None:
            print(f"  ⚠️  {layer_name}: 未捕获到特征图")
            continue
            
        feat_map = feature[0].numpy()  # [C, H, W]
        layer_save_dir = os.path.join(save_dir, folder_name)
        os.makedirs(layer_save_dir, exist_ok=True)
        
        # 多通道处理
        if feat_map.ndim == 3:
            C, H, W = feat_map.shape
            print(f"  ✅ {folder_name:20s} | 真实尺寸: {H}x{W} | 通道数: {C}")
            
            # 为该图片的所有通道创建子文件夹
            img_channel_dir = os.path.join(layer_save_dir, base_name)
            os.makedirs(img_channel_dir, exist_ok=True)
            
            for c_idx in range(C):
                channel_data = feat_map[c_idx]
                
                # 归一化用于染色
                channel_norm = (channel_data - channel_data.min()) / (channel_data.max() - channel_data.min() + 1e-8)
                
                # 应用 Viridis 颜色映射
                cmap = plt.cm.get_cmap('viridis')
                channel_color = cmap(channel_norm)[:, :, :3] # 舍弃 Alpha 通道
                
                # 直接保存，不使用 cv2.resize
                save_path = os.path.join(img_channel_dir, f"channel_{c_idx:03d}.png")
                plt.imsave(save_path, channel_color)
        
        extractor.release()

def batch_process(image_dir, save_dir, model, config, device, layer_configs):
    """批量处理"""
    image_files = sorted([f for f in os.listdir(image_dir) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff'))])
    
    print(f"\n📁 开始处理目录: {image_dir}，共 {len(image_files)} 张图\n")
    
    for i, img_file in enumerate(image_files, 1):
        image_path = os.path.join(image_dir, img_file)
        print(f"[{i}/{len(image_files)}] 正在处理: {img_file}")
        try:
            extract_and_save_features(image_path, save_dir, model, config, device, layer_configs)
        except Exception as e:
            print(f"  ❌ 错误: {e}")

def main():
    # --- 配置区 ---
    model_path = 'checkpoints/pth_sar/best_ConvNeXt_model.pth'
    image_dir = '/home/xiaobowen/project/internal_wave_detection_project/IW_data/images_sar'
    save_dir = '/home/xiaobowen/project/internal_wave_detection_project/IW_data/features_onlysar/ConvNeXt_real_size/'
    
    # 配置要提取的层 (层路径, 保存文件夹名)
    layer_configs = [
        ('encoder.norm0', 'stage1_256x256'), 
        ('encoder.norm1', 'stage2_128x128'), 
        ('encoder.norm2', 'stage3_64x64'), 
        ('encoder.norm3', 'stage4_32x32'), 
    ]
    # --------------

    config = Config()
    
    # 加载模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(model_path, map_location=device)
    
    model = create_unet('ConvNeXt', num_classes=1, dropout_rates=[0.0, 0.0, 0.0, 0.0])
    
    # 兼容性加载权重
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    model.load_state_dict(state_dict, strict=False)
    model.to(device).eval()
    
    batch_process(image_dir, save_dir, model, config, device, layer_configs)
    print(f"\n🎉 任务完成！特征图已按真实尺寸保存在: {save_dir}")

if __name__ == '__main__':
    main()