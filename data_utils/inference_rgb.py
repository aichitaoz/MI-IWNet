import os
import cv2
import torch
import numpy as np
import sys

# 添加上级目录到 sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train.main import Config
from models.unet import create_unet

import warnings
warnings.filterwarnings('ignore')


def load_and_cut_patches(image_path, crop_size=1024, stride=None):
    """
    加载RGB图像并切成patches，用于推理
    返回：patches列表，原始图像尺寸，patch位置信息
    """
    if stride is None:
        stride = crop_size

    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    if image is None:
        raise ValueError(f"Cannot load image: {image_path}")

    # 与 InterWaveDataset 保持一致：先resize到4096×2048
    h, w = image.shape[:2]
    if h != 1024 or w != 1024:
        image = cv2.resize(image, (4096, 2048), interpolation=cv2.INTER_LINEAR)

    # 处理图像格式
    if image.ndim == 2:
        processed_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 3:
        if np.allclose(image[..., 0], image[..., 1]) and np.allclose(image[..., 1], image[..., 2]):
            processed_image = cv2.cvtColor(image[..., 0], cv2.COLOR_GRAY2RGB)
        else:
            processed_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif image.shape[2] == 4:
        image_bgr = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        if np.allclose(image_bgr[..., 0], image_bgr[..., 1]) and np.allclose(image_bgr[..., 1], image_bgr[..., 2]):
            processed_image = cv2.cvtColor(image_bgr[..., 0], cv2.COLOR_GRAY2RGB)
        else:
            processed_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    else:
        raise ValueError(f"Unsupported image format: {image_path}")

    h, w = processed_image.shape[:2]
    patches = []
    patch_positions = []

    # 滑窗切割
    for y in range(0, max(1, h - crop_size + 1), stride):
        for x in range(0, max(1, w - crop_size + 1), stride):
            y2, x2 = min(y + crop_size, h), min(x + crop_size, w)
            patch_img = processed_image[y2 - crop_size:y2, x2 - crop_size:x2]
            patches.append(patch_img)
            patch_positions.append((y2 - crop_size, x2 - crop_size, crop_size, crop_size))

    return patches, (h, w), patch_positions, processed_image


def preprocess_patch(patch_img):
    """
    将单个RGB patch转换为模型输入tensor
    """
    # [H,W,3] -> [3,H,W]
    input_tensor = torch.from_numpy(patch_img.transpose(2,0,1).astype(np.float32)/255.0)
    # normalize [-1,1]
    input_tensor = (input_tensor - 0.5) / 0.5
    # 增加 batch 维度 [1,C,H,W]
    input_tensor = input_tensor.unsqueeze(0)
    return input_tensor


def merge_patch_predictions(patch_predictions, patch_positions, original_size, crop_size=1024):
    """
    将patch预测结果合并成完整图像
    """
    h, w = original_size
    merged_prob = np.zeros((h, w), dtype=np.float32)
    overlap_count = np.zeros((h, w), dtype=np.int32)

    for pred_prob, (y, x, patch_h, patch_w) in zip(patch_predictions, patch_positions):
        end_y = min(y + patch_h, h)
        end_x = min(x + patch_w, w)
        actual_h = end_y - y
        actual_w = end_x - x

        patch_prob_crop = pred_prob[:actual_h, :actual_w]
        merged_prob[y:end_y, x:end_x] += patch_prob_crop
        overlap_count[y:end_y, x:end_x] += 1

    # 求平均（处理重叠区域）
    mask = overlap_count > 0
    merged_prob[mask] /= overlap_count[mask]
    merged_binary = (merged_prob > 0.5).astype(np.uint8)

    return merged_prob, merged_binary


def process_patches_with_model(patches, model, device):
    """
    对所有RGB patches进行模型推理
    """
    patch_predictions = []

    for patch_img in patches:
        input_tensor = preprocess_patch(patch_img)
        pred_prob = predict_image(model, input_tensor, device)
        patch_predictions.append(pred_prob)

    return patch_predictions


def predict_image(model, input_tensor, device):
    """
    输入 [1,C,H,W] tensor，输出概率图
    """
    input_tensor = input_tensor.to(device)
    with torch.no_grad():
        output = model(input_tensor, datasets=['rgb'])  # [1,1,H,W]
        prob = torch.sigmoid(output).cpu().numpy()[0,0]  # [H,W]
    return prob


def load_model(model_path, config):
    """
    加载模型
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
        model = create_convnext_unet(config.MODEL_TYPE, num_classes=1)
    else:
        raise ValueError(f"Unsupported model type: {config.MODEL_TYPE}")

    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device).eval()
    return model, device


def create_overlay_image(image, pred_binary, alpha=0.5):
    """
    将预测mask覆盖在原图上（红色显示预测区域）
    """
    overlay = image.astype(np.float32)/255.0

    pred_mask = pred_binary > 0

    # 如果是灰度单通道，将灰度复制成RGB
    if overlay.ndim == 2:
        overlay = np.stack([overlay]*3, axis=-1)

    # 红色显示预测区域
    overlay[pred_mask] = overlay[pred_mask] * (1 - alpha) + np.array([1.0, 0.0, 0.0]) * alpha

    return overlay


def infer_and_save(image_path, save_dir, model, config, device):
    """
    推理单张图像，保存mask和overlay两张图
    """
    # 1. 切割图像为patches
    patches, original_size, patch_positions, processed_image = load_and_cut_patches(
        image_path, crop_size=1024, stride=512
    )

    print(f"Processing {len(patches)} patches for {os.path.basename(image_path)}")

    # 2. 对每个patch进行模型推理
    patch_predictions = process_patches_with_model(patches, model, device)

    # 3. 合并patch预测结果
    merged_prob, merged_binary = merge_patch_predictions(
        patch_predictions, patch_positions, original_size
    )

    # 4. 创建overlay图像
    overlay_image = create_overlay_image(processed_image, merged_binary)

    # 5. 保存结果
    os.makedirs(save_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # 保存mask图（二值图，0-255）
    mask_save_path = os.path.join(save_dir, f"{base_name}_mask.png")
    cv2.imwrite(mask_save_path, merged_binary * 255)
    
    # 保存overlay图（RGB格式）
    overlay_save_path = os.path.join(save_dir, f"{base_name}_overlay.png")
    overlay_bgr = cv2.cvtColor((overlay_image * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.imwrite(overlay_save_path, overlay_bgr)

    return mask_save_path, overlay_save_path


def batch_infer(image_dir, save_dir, model, config, device):
    """
    批量推理图像
    """
    image_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(('.png','.jpg','.jpeg','.tiff'))])
    print(f"Found {len(image_files)} images to process")

    for i, img_file in enumerate(image_files):
        print(f"\n[{i+1}/{len(image_files)}] Processing {img_file}")
        image_path = os.path.join(image_dir, img_file)

        try:
            mask_path, overlay_path = infer_and_save(image_path, save_dir, model, config, device)
            print(f"✅ Saved mask to: {mask_path}")
            print(f"✅ Saved overlay to: {overlay_path}")
        except Exception as e:
            print(f"❌ Error processing {img_file}: {str(e)}")
            continue


def main():
    """
    主函数：推理并保存结果
    """
    config = Config()

    # 模型路径选择
    if config.MODEL_TYPE == 'unet':
        model_path = 'checkpoints/pth/best_unet_model.pth'
    elif config.MODEL_TYPE == 'mobile_unet':
        model_path = 'checkpoints/pth/best_mobile_unet_model.pth'
    elif config.MODEL_TYPE == 'convnext_unet':
        model_path = 'checkpoints/pth/best_convnext_unet_model.pth'

    # 输入输出路径
    image_dir = '/home/xiaobowen/project/internal_wave_detection_project/IW_data/inference/rgb'
    save_dir = f'/home/xiaobowen/project/internal_wave_detection_project/IW_data/results/infer'

    print(f"Loading model: {model_path}")
    print(f"Processing RGB images from: {image_dir}")
    print(f"Results will be saved to: {save_dir}")

    # 加载模型
    model, device = load_model(model_path, config)

    # 批量推理
    batch_infer(image_dir, save_dir, model, config, device)

    print(f"\n✅ All inference completed! Results saved in: {save_dir}")


if __name__ == '__main__':
    main()