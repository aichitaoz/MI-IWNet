import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import sys

# 添加上级目录到 sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train.main import Config
from models.unet import create_unet

import warnings
warnings.filterwarnings('ignore')


def load_and_cut_patches(image_path, mask_path, crop_size=1024, stride=None):
    """
    加载RGB图像并切成patches
    """
    if stride is None:
        stride = crop_size

    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if image is None or mask is None:
        raise ValueError(f"Cannot load image or mask: {image_path}")

    # resize到4096×2048
    h, w = image.shape[:2]
    if h != 1024 or w != 1024:
        image = cv2.resize(image, (4096, 2048), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (4096, 2048), interpolation=cv2.INTER_LINEAR)

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

    # mask 二值化
    mask_binary = (mask > 127).astype(np.uint8)

    h, w = processed_image.shape[:2]
    image_patches = []
    patch_positions = []

    # 滑窗切割
    for y in range(0, max(1, h - crop_size + 1), stride):
        for x in range(0, max(1, w - crop_size + 1), stride):
            y2, x2 = min(y + crop_size, h), min(x + crop_size, w)
            patch_img = processed_image[y2 - crop_size:y2, x2 - crop_size:x2]
            patch_mask = mask_binary[y2 - crop_size:y2, x2 - crop_size:x2]
            image_patches.append((patch_img, patch_mask))
            patch_positions.append((y2 - crop_size, x2 - crop_size, crop_size, crop_size))

    return image_patches, (h, w), patch_positions, processed_image, mask_binary


def preprocess_patch(patch_img, patch_mask):
    """将单个RGB patch转换为模型输入tensor"""
    input_tensor = torch.from_numpy(patch_img.transpose(2,0,1).astype(np.float32)/255.0)
    input_tensor = (input_tensor - 0.5) / 0.5
    input_tensor = input_tensor.unsqueeze(0)
    return input_tensor


def merge_patch_predictions(patch_predictions, patch_positions, original_size, crop_size=1024):
    """将patch预测结果合并成完整图像"""
    h, w = original_size
    merged_prob = np.zeros((h, w), dtype=np.float32)
    merged_binary = np.zeros((h, w), dtype=np.uint8)
    overlap_count = np.zeros((h, w), dtype=np.int32)

    for (pred_prob, pred_binary), (y, x, patch_h, patch_w) in zip(patch_predictions, patch_positions):
        end_y = min(y + patch_h, h)
        end_x = min(x + patch_w, w)
        actual_h = end_y - y
        actual_w = end_x - x

        patch_prob_crop = pred_prob[:actual_h, :actual_w]
        patch_binary_crop = pred_binary[:actual_h, :actual_w]

        merged_prob[y:end_y, x:end_x] += patch_prob_crop
        merged_binary[y:end_y, x:end_x] += patch_binary_crop
        overlap_count[y:end_y, x:end_x] += 1

    mask = overlap_count > 0
    merged_prob[mask] /= overlap_count[mask]
    merged_binary[mask] = (merged_binary[mask] / overlap_count[mask] > 0.5).astype(np.uint8)

    return merged_prob, merged_binary


def process_patches_with_model(image_patches, model, device, threshold=0.5):
    """对所有RGB patches进行模型推理"""
    patch_predictions = []
    for patch_img, patch_mask in image_patches:
        input_tensor = preprocess_patch(patch_img, patch_mask)
        pred_prob, pred_binary = predict_image(model, input_tensor, device, threshold)
        patch_predictions.append((pred_prob, pred_binary))
    return patch_predictions


def predict_image(model, input_tensor, device, threshold=0.5):
    """输入 [1,C,H,W] tensor，输出概率图和二值预测"""
    input_tensor = input_tensor.to(device)
    with torch.no_grad():
        output = model(input_tensor, datasets=['rgb'])
        prob = torch.sigmoid(output).cpu().numpy()[0,0]
        binary = (prob > threshold).astype(np.uint8)
    return prob, binary


def load_model(model_path, config):
    """加载模型"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(model_path, map_location=device)
    model = create_unet(config.MODEL_TYPE, num_classes=1, dropout_rates=[0.0, 0.0, 0.0, 0.0])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device).eval()
    return model, device


def pixel_to_latlon(y, x, img_shape, lat_range, lon_range):
    """将像素坐标转换为经纬度"""
    h, w = img_shape
    lat_min, lat_max = lat_range
    lon_min, lon_max = lon_range
    
    lat = lat_max - (y / h) * (lat_max - lat_min)
    lon = lon_min + (x / w) * (lon_max - lon_min)
    return lat, lon


def create_full_overlay(image, pred_binary, mask_binary, alpha=0.5):
    """
    创建完整的overlay图（红色预测、绿色GT、黄色重叠）
    """
    # 归一化处理
    if image.dtype == np.uint8:
        overlay = image.astype(np.float32) / 255.0
    elif image.dtype == np.uint16:
        overlay = image.astype(np.float32) / 65535.0
    else:
        img_min, img_max = image.min(), image.max()
        if img_max > img_min:
            overlay = (image.astype(np.float32) - img_min) / (img_max - img_min)
        else:
            overlay = image.astype(np.float32)

    pred_mask = pred_binary > 0
    gt_mask = mask_binary > 0
    overlap_mask = pred_mask & gt_mask

    # 如果是灰度单通道，转成RGB
    if overlay.ndim == 2:
        overlay = np.stack([overlay]*3, axis=-1)

    # 只在预测中出现（红色）
    pred_only = pred_mask & (~gt_mask)
    # 只在GT中出现（绿色）
    gt_only = gt_mask & (~pred_mask)
    
    overlay[pred_only] = overlay[pred_only] * (1 - alpha) + np.array([1.0, 0.0, 0.0]) * alpha  # 红色
    overlay[gt_only] = overlay[gt_only] * (1 - alpha) + np.array([0.0, 1.0, 0.0]) * alpha      # 绿色
    overlay[overlap_mask] = overlay[overlap_mask] * (1 - alpha) + np.array([1.0, 1.0, 0.0]) * alpha  # 黄色

    # 裁剪到 [0, 1] 范围
    overlay = np.clip(overlay, 0.0, 1.0)
    
    return overlay


def create_paper_style_visualization(image_path, mask_path, model, config, device, 
                                     save_path, roi_coords, lat_range, lon_range, threshold=0.5):
    """
    创建论文风格的可视化图
    新布局：
    - 左上 (a): 完整 overlay（带矩形框）
    - 左下 (b): Ground Truth（带矩形框）
    - 中间 (c): Prediction Mask 的 ROI 放大
    - 右边 (d): Overlay 的 ROI 放大
    """
    # 1. 加载并处理图像
    image_patches, original_size, patch_positions, processed_image, mask_binary = load_and_cut_patches(
        image_path, mask_path, crop_size=1024, stride=512
    )
    
    print(f"Processing {len(image_patches)} patches...")
    print(f"Image size: {original_size[0]} x {original_size[1]} (H x W)")
    
    # 2. 模型预测
    patch_predictions = process_patches_with_model(image_patches, model, device, threshold)
    merged_prob, merged_binary = merge_patch_predictions(patch_predictions, patch_positions, original_size)
    
    # 3. 创建完整的overlay图
    full_overlay = create_full_overlay(processed_image, merged_binary, mask_binary, alpha=0.5)
    
    # 4. 提取ROI区域
    y1, x1, y2, x2 = roi_coords
    print(f"ROI coordinates: y1={y1}, x1={x1}, y2={y2}, x2={x2}")
    
    roi_image = processed_image[y1:y2, x1:x2]
    roi_mask = mask_binary[y1:y2, x1:x2]
    roi_pred = merged_binary[y1:y2, x1:x2]
    roi_overlay = full_overlay[y1:y2, x1:x2]
    
    roi_h, roi_w = roi_image.shape[:2]
    print(f"ROI size: {roi_h} x {roi_w} (H x W)")
    
    # 5. 创建图形布局 - 2行3列
    fig = plt.figure(figsize=(22, 8))
    gs = GridSpec(2, 3, figure=fig, width_ratios=[1, 1.2, 1.2], 
                  height_ratios=[1, 1], wspace=0.25, hspace=0.02)  # 直接在 GridSpec 中设置很小的 hspace
    
    # 经纬度标签设置
    h, w = processed_image.shape[:2]
    lat_ticks = np.linspace(0, h, 6)
    lon_ticks = np.linspace(0, w, 6)
    lat_labels = [f"{lat:.1f}°N" for lat in np.linspace(lat_range[1], lat_range[0], 6)]
    lon_labels = [f"{lon:.1f}°E" for lon in np.linspace(lon_range[0], lon_range[1], 6)]
    
    # === 左上 (a)：完整 overlay + 矩形框 ===
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(full_overlay)
    
    # 绘制白色矩形框
    rect_width = x2 - x1
    rect_height = y2 - y1
    rect = mpatches.Rectangle((x1, y1), rect_width, rect_height, 
                             linewidth=1, edgecolor='white', facecolor='none')
    ax1.add_patch(rect)
    
    ax1.set_yticks(lat_ticks)
    ax1.set_yticklabels(lat_labels, fontsize=8)
    ax1.set_xticks(lon_ticks)
    ax1.set_xticklabels(lon_labels, fontsize=8, rotation=45)
    ax1.set_title('(a) Full Overlay (Red: Pred, Green: GT, Yellow: Overlap)', 
                  fontsize=10, fontweight='bold')
    
    # === 左下 (b)：GT mask + 矩形框 ===
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.imshow(mask_binary, cmap='gray')
    
    # 绘制白色矩形框
    rect2 = mpatches.Rectangle((x1, y1), rect_width, rect_height, 
                              linewidth=1, edgecolor='white', facecolor='none')
    ax2.add_patch(rect2)
    
    ax2.set_yticks(lat_ticks)
    ax2.set_yticklabels(lat_labels, fontsize=8)
    ax2.set_xticks(lon_ticks)
    ax2.set_xticklabels(lon_labels, fontsize=8, rotation=45)
    ax2.set_title('(b) Ground Truth', fontsize=10, fontweight='bold')
    
    # ROI区域的经纬度范围
    roi_lat_min, _ = pixel_to_latlon(y2, x1, processed_image.shape[:2], lat_range, lon_range)
    roi_lat_max, _ = pixel_to_latlon(y1, x1, processed_image.shape[:2], lat_range, lon_range)
    _, roi_lon_min = pixel_to_latlon(y1, x1, processed_image.shape[:2], lat_range, lon_range)
    _, roi_lon_max = pixel_to_latlon(y1, x2, processed_image.shape[:2], lat_range, lon_range)
    
    roi_lat_ticks = np.linspace(0, roi_h, 5)
    roi_lon_ticks = np.linspace(0, roi_w, 5)
    roi_lat_labels = [f"{lat:.1f}°N" for lat in np.linspace(roi_lat_max, roi_lat_min, 5)]
    roi_lon_labels = [f"{lon:.1f}°E" for lon in np.linspace(roi_lon_min, roi_lon_max, 5)]
    
    # === 中间 (c)：Prediction Mask 的 ROI 放大 ===
    ax3 = fig.add_subplot(gs[:, 1])  # 占据两行
    ax3.imshow(roi_pred, cmap='gray')
    ax3.set_yticks(roi_lat_ticks)
    ax3.set_yticklabels(roi_lat_labels, fontsize=8)
    ax3.set_xticks(roi_lon_ticks)
    ax3.set_xticklabels(roi_lon_labels, fontsize=8, rotation=45)
    ax3.set_title('(c) Prediction Mask (Enlarged)', fontsize=10, fontweight='bold')
    ax3.set_aspect('equal')
    
    # === 右边 (d)：Overlay 的 ROI 放大 ===
    ax4 = fig.add_subplot(gs[:, 2])  # 占据两行
    ax4.imshow(roi_overlay)
    ax4.set_yticks(roi_lat_ticks)
    ax4.set_yticklabels(roi_lat_labels, fontsize=8)
    ax4.set_xticks(roi_lon_ticks)
    ax4.set_xticklabels(roi_lon_labels, fontsize=8, rotation=45)
    ax4.set_title('(d) Overlay (Enlarged)', fontsize=10, fontweight='bold')
    ax4.set_aspect('equal')
    
    # 保存时使用 bbox_inches='tight' 去除白边，但用 pad_inches 控制边距
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    
    print(f"✅ Paper-style visualization saved to: {save_path}")


def main():
    """主函数"""
    config = Config()
    
    # 模型路径
    if config.MODEL_TYPE == 'ConvNeXt':
        model_path = 'checkpoints/pth/best_ConvNeXt_model.pth'
    elif config.MODEL_TYPE == 'mobile_unet':
        model_path = 'checkpoints/pth/best_mobile_unet_model.pth'
    elif config.MODEL_TYPE == 'convnext_unet':
        model_path = 'checkpoints/pth/best_convnext_unet_model.pth'
    
    # 图像路径
    image_path = '/home/xiaobowen/project/internal_wave_detection_project/IW_data/RGB/test/images/MODIS_TrueColor_2002-06-23_1Terra.tiff'
    mask_path = '/home/xiaobowen/project/internal_wave_detection_project/IW_data/RGB/test/segmentation_masks/MODIS_TrueColor_2002-06-23_1Terra.png'
    save_dir = f'/home/xiaobowen/project/internal_wave_detection_project/IW_data/results/{config.MODEL_TYPE}/RGB/paper_style/'
    
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'MODIS_2002-06-23_paper_style.png')
    
    # 经纬度范围
    lat_range = (18.32, 23.19)  # (lat_min, lat_max)
    lon_range = (112.4, 121.32)  # (lon_min, lon_max)
    
    # ROI坐标设置
    roi_coords = (500, 1600, 1400, 2500)  # (y1, x1, y2, x2)
    
    print(f"Loading model: {model_path}")
    model, device = load_model(model_path, config)
    
    print(f"Processing image: {image_path}")
    create_paper_style_visualization(
        image_path, mask_path, model, config, device,
        save_path, roi_coords, lat_range, lon_range, threshold=0.8
    )
    
    print(f"\n✅ Visualization completed!")


if __name__ == '__main__':
    main()