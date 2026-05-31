import os
import sys
import cv2
import torch
import numpy as np
import warnings

# 添加上级目录到 sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.unet import create_unet
warnings.filterwarnings('ignore')

def load_and_preprocess_image(image_path, target_size=(4096, 2048)):
    """加载图像，调整到统一尺寸并转换为 RGB 格式"""
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"无法读取图像: {image_path}")

    h, w = image.shape[:2]
    if h != 1024 or w != 1024:
        image = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)

    # 统一转换为 RGB 三通道
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

def remove_small_components(binary_mask, min_area=50):
    """
    通过连通域分析，去除面积较小的噪点斑块。
    min_area: 最小保留面积 (像素个数)，4096*2048 图建议设为 1000~5000
    """
    # 查找连通域 (8邻接)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    
    clean_mask = np.zeros_like(binary_mask)
    # 标签 0 是背景，所以从 1 开始遍历
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            clean_mask[labels == i] = 1
            
    return clean_mask

def create_gt_pre_overlay(image, pre_mask, gt_mask, alpha=0.5):
    """
    将 GT(绿) 和 Pre(红) 叠加到原图上。
    TP(预测正确的区域): 会呈现黄色 (红+绿)
    FN(漏检的GT区域): 会呈现纯绿色
    FP(误检的Pre区域): 会呈现纯红色
    """
    overlay = image.astype(np.float32) / 255.0

    # 1. 先把 GT 染成绿色
    gt_indices = gt_mask > 0
    overlay[gt_indices] = overlay[gt_indices] * (1 - alpha) + np.array([0.0, 1.0, 0.0]) * alpha
    
    # 2. 把 Pre 染成红色
    pre_indices = pre_mask > 0
    overlay[pre_indices] = overlay[pre_indices] * (1 - alpha) + np.array([1.0, 0.0, 0.0]) * alpha

    # 3. 如果重叠(TP)，为了更清晰，我们显式将其设置为黄色
    both_indices = gt_indices & pre_indices
    overlay[both_indices] = overlay[both_indices] * (1 - alpha) + np.array([1.0, 1.0, 0.0]) * alpha

    return (overlay * 255).astype(np.uint8)

def process_single_image(image_path, save_dir_pre, save_dir_overlay, gt_dir, model, device):
    """处理单张图片的完整流水线"""
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # 1. 预处理与切图
    processed_image = load_and_preprocess_image(image_path)
    original_size = processed_image.shape[:2]
    patches, positions = get_image_patches(processed_image, crop_size=1024, stride=512)

    # 2. 模型推理与拼接
    patch_predictions = predict_patches(model, patches, device)
    pre_mask = merge_patches(patch_predictions, positions, original_size)

    # --- 核心改动 3: 连通域过滤 ---
    # 剔除面积小于 2000 的极小区域
    pre_mask = remove_small_components(pre_mask, min_area=200)

    # --- 核心改动 1: 读取对应的 GT 图像 ---
    # 尝试读取同名的 png 或与原图同后缀的文件
    gt_path_png = os.path.join(gt_dir, f"{base_name}.png")
    gt_path_orig = os.path.join(gt_dir, os.path.basename(image_path))
    
    if os.path.exists(gt_path_png):
        gt_path = gt_path_png
    elif os.path.exists(gt_path_orig):
        gt_path = gt_path_orig
    else:
        gt_path = None

    if gt_path:
        gt_img = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        # 如果 GT 和原图 resize 后的尺寸不一致，强制对其尺寸
        if gt_img.shape[:2] != original_size:
            gt_img = cv2.resize(gt_img, (original_size[1], original_size[0]), interpolation=cv2.INTER_NEAREST)
        gt_mask = (gt_img > 127).astype(np.uint8)
    else:
        print(f"  [提示] 未找到 {base_name} 的GT标签，Overlay将仅显示预测(红)")
        gt_mask = np.zeros_like(pre_mask)

    # 4. 生成叠加图 (Img + GT(绿) + Pre(红))
    overlay_img = create_gt_pre_overlay(processed_image, pre_mask, gt_mask)

    # --- 核心改动 2: 分别保存到两个文件夹 ---
    pre_save_path = os.path.join(save_dir_pre, f"{base_name}_pre.png")
    overlay_save_path = os.path.join(save_dir_overlay, f"{base_name}_overlay.png")
    
    # 保存纯预测结果
    cv2.imwrite(pre_save_path, pre_mask * 255)
    # 保存彩色叠加结果 (转 BGR 保存)
    cv2.imwrite(overlay_save_path, cv2.cvtColor(overlay_img, cv2.COLOR_RGB2BGR))

def main():
    # ---------- 配置区域 ----------
    model_path = '/home/xiaobowen/project/internal_wave_detection_project/checkpoints/revise/pth/best_ConvNeXt_model.pth'
    
    # 输入与真实标签路径
    image_dir = '/home/xiaobowen/project/internal_wave_detection_project/IW_data/images_rgb'
    gt_dir = '/home/xiaobowen/project/internal_wave_detection_project/IW_data/masks_rgb'
    
    # 输出根目录
    save_root_dir = '/home/xiaobowen/project/internal_wave_detection_project/IW_data/results/infer'
    # ------------------------------

    # 创建两个独立的保存文件夹
    save_dir_pre = os.path.join(save_root_dir, 'pre')
    save_dir_overlay = os.path.join(save_root_dir, 'overlay')
    os.makedirs(save_dir_pre, exist_ok=True)
    os.makedirs(save_dir_overlay, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"正在加载 ConvNeXt 模型: {model_path} (Device: {device})")

    # 初始化并加载 ConvNeXt 模型
    model = create_unet('ConvNeXt', num_classes=1)
    model.load_state_dict(torch.load(model_path, map_location=device)['model_state_dict'])
    model.to(device).eval()

    image_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff'))])
    print(f"找到 {len(image_files)} 张图像，开始推理...")

    for i, img_file in enumerate(image_files, 1):
        image_path = os.path.join(image_dir, img_file)
        try:
            process_single_image(image_path, save_dir_pre, save_dir_overlay, gt_dir, model, device)
            print(f"[{i}/{len(image_files)}] ✅ 成功处理: {img_file}")
        except Exception as e:
            print(f"[{i}/{len(image_files)}] ❌ 处理失败 {img_file}: {str(e)}")

    print(f"\n🎉 所有推理完成！")
    print(f"👉 纯预测图 (Pre) 保存在: {save_dir_pre}")
    print(f"👉 对比叠加图 (Overlay) 保存在: {save_dir_overlay}")

if __name__ == '__main__':
    main()