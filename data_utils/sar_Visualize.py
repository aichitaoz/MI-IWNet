import os
import sys
import cv2
import torch
import numpy as np
import warnings

# 添加上级目录到 sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train.main import Config
from models.unet import create_unet
# 注意：如果你有其他外部的 import (比如 create_mobile_unet)，请在这里自行补充

warnings.filterwarnings('ignore')

def load_and_preprocess_image(image_path, target_size=(4096, 2048)):
    """加载图像，调整到统一尺寸并转换为 RGB 格式"""
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"无法读取图像: {image_path}")

    h, w = image.shape[:2]
    if h != 1024 or w != 1024:
        image = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)

    if image.ndim == 2:
        processed_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        processed_image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    else:
        processed_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return processed_image

def get_image_patches(image, crop_size=1024, stride=512):
    """通过滑窗将原图切割为多个 patch"""
    h, w = image.shape[:2]
    patches, positions = [], []

    for y in range(0, max(1, h - crop_size + 1), stride):
        for x in range(0, max(1, w - crop_size + 1), stride):
            y_end, x_end = min(y + crop_size, h), min(x + crop_size, w)
            patch_img = image[y_end - crop_size:y_end, x_end - crop_size:x_end]
            
            patches.append(patch_img)
            positions.append((y_end - crop_size, x_end - crop_size, crop_size, crop_size))

    return patches, positions

def predict_patches(model, patches, device):
    """对所有的 patch 进行批量/逐个推理"""
    patch_predictions = []
    
    for patch_img in patches:
        # [H, W, 3] -> [1, 3, H, W] 并归一化到 [-1, 1]
        input_tensor = torch.from_numpy(patch_img.transpose(2, 0, 1).astype(np.float32) / 255.0)
        input_tensor = ((input_tensor - 0.5) / 0.5).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(input_tensor, datasets=['rgb'])  
            prob = torch.sigmoid(output).cpu().numpy()[0, 0] 
            patch_predictions.append(prob)
            
    return patch_predictions

def merge_patches(patch_predictions, positions, original_size):
    """将 patch 的预测结果平滑拼接回原图尺寸"""
    h, w = original_size
    merged_prob = np.zeros((h, w), dtype=np.float32)
    overlap_count = np.zeros((h, w), dtype=np.int32)

    for pred_prob, (y, x, patch_h, patch_w) in zip(patch_predictions, positions):
        end_y, end_x = min(y + patch_h, h), min(x + patch_w, w)
        actual_h, actual_w = end_y - y, end_x - x

        merged_prob[y:end_y, x:end_x] += pred_prob[:actual_h, :actual_w]
        overlap_count[y:end_y, x:end_x] += 1

    mask = overlap_count > 0
    merged_prob[mask] /= overlap_count[mask]
    merged_binary = (merged_prob > 0.5).astype(np.uint8)

    return merged_binary

def create_overlay(image, pre_mask, alpha=0.5):
    """将预测的 Mask (红色) 叠加到图像上"""
    overlay = image.astype(np.float32) / 255.0
    mask_indices = pre_mask > 0

    overlay[mask_indices] = overlay[mask_indices] * (1 - alpha) + np.array([1.0, 0.0, 0.0]) * alpha
    return (overlay * 255).astype(np.uint8)

def load_model(model_path, config):
    """
    完整还原你原版的加载逻辑，依然依靠 Config 调度
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(model_path, map_location=device)

    if config.MODEL_TYPE == 'unet':
        model = create_unet(
            config.MODEL_TYPE,
            num_classes=1,
            da_reductions=config.DA_REDUCTIONS
        )
    elif config.MODEL_TYPE == 'mobile_unet':
        model = create_mobile_unet(config.MODEL_TYPE, num_classes=1) 
    elif config.MODEL_TYPE == 'convnext_unet':
        # ⚠️ 恢复你的原代码状态。由于你顶部并没有 import create_convnext_unet，
        # 如果你运行到这里报错 NameError，你需要按照你实际情况把它改成统一的接口：
        # 例如改为：model = create_unet(config.MODEL_TYPE, num_classes=1)
        model = create_convnext_unet(config.MODEL_TYPE, num_classes=1)
    else:
        raise ValueError(f"Unsupported model type: {config.MODEL_TYPE}")

    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device).eval()
    return model, device

def process_single_image(image_path, save_dir, model, device):
    """处理单张图片的完整流水线 (仅保存 pre 和 overlay)"""
    processed_image = load_and_preprocess_image(image_path)
    original_size = processed_image.shape[:2]
    patches, positions = get_image_patches(processed_image, crop_size=1024, stride=512)

    patch_predictions = predict_patches(model, patches, device)
    pre_mask = merge_patches(patch_predictions, positions, original_size)
    overlay_img = create_overlay(processed_image, pre_mask)

    os.makedirs(save_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    
    pre_save_path = os.path.join(save_dir, f"{base_name}_pre.png")
    overlay_save_path = os.path.join(save_dir, f"{base_name}_overlay.png")
    
    cv2.imwrite(pre_save_path, pre_mask * 255)
    cv2.imwrite(overlay_save_path, cv2.cvtColor(overlay_img, cv2.COLOR_RGB2BGR))

def main():
    config = Config()

    # 模型路径选择机制恢复
    if config.MODEL_TYPE == 'unet':
        model_path = 'checkpoints/pth/best_unet_model.pth'
    elif config.MODEL_TYPE == 'mobile_unet':
        model_path = 'checkpoints/pth/best_mobile_unet_model.pth'
    elif config.MODEL_TYPE == 'convnext_unet':
        model_path = 'checkpoints/pth/best_convnext_unet_model.pth'

    image_dir = '/home/xiaobowen/project/internal_wave_detection_project/IW_data/inference/rgb'
    save_dir = '/home/xiaobowen/project/internal_wave_detection_project/IW_data/results/infer'

    print(f"正在基于 Config 加载 {config.MODEL_TYPE} 模型: {model_path}")

    # 使用原版的 load_model 加载
    model, device = load_model(model_path, config)

    image_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff'))])
    print(f"找到 {len(image_files)} 张图像，开始推理...")

    for i, img_file in enumerate(image_files, 1):
        image_path = os.path.join(image_dir, img_file)
        try:
            process_single_image(image_path, save_dir, model, device)
            print(f"[{i}/{len(image_files)}] ✅ 成功处理: {img_file}")
        except Exception as e:
            print(f"[{i}/{len(image_files)}] ❌ 处理失败 {img_file}: {str(e)}")

    print(f"\n🎉 所有推理完成！结果已保存在: {save_dir}")

if __name__ == '__main__':
    main()