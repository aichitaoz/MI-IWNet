import os
import cv2
import torch
import numpy as np
import sys
import warnings

warnings.filterwarnings('ignore')

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.unet import create_unet
from configs.train_config import Config
from utils.get_transforms import get_transforms

# =========================
# 连通区域过滤
# =========================
def remove_small_regions(binary_mask, min_area=100):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    filtered = np.zeros_like(binary_mask)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            filtered[labels == i] = 255
    return filtered

# =========================
# 图像预处理 tile (同步可视化脚本的归一化逻辑)
# =========================
def preprocess_image_tile(tile, config):
    # 同步可视化脚本：转换为 float32 -> /255.0 -> 归一化 [-1, 1]
    tile_float = tile.astype(np.float32) / 255.0
    input_tensor = torch.from_numpy(tile_float).unsqueeze(0) # [1, H, W]
    
    # 关键归一化逻辑
    input_tensor = (input_tensor - 0.5) / 0.5
    
    return input_tensor.unsqueeze(0) # [1, 1, H, W]

# =========================
# 推理单 tile (指定 sar 分支)
# =========================
def predict_tile(model, tile_tensor, device, threshold=0.5):
    tile_tensor = tile_tensor.to(device)
    with torch.no_grad():
        # ✅ 关键：传入 datasets 参数，确保模态对齐
        output = model(tile_tensor, datasets=['sar'])
        prob = torch.sigmoid(output).cpu().numpy()[0, 0]
        binary = (prob > threshold).astype(np.uint8) * 255
    return prob, binary

# =========================
# tile 推理大图 (处理边缘 Padding 以防尺寸报错)
# =========================
def tile_inference(image, model, config, device, tile_size=1024, stride=None, threshold=0.5, min_area=100):
    if stride is None:
        stride = tile_size

    H, W = image.shape
    pred_prob = np.zeros((H, W), dtype=np.float32)
    pred_count = np.zeros((H, W), dtype=np.uint8)

    for y in range(0, H, stride):
        for x in range(0, W, stride):
            y1, x1 = y, x
            y2, x2 = min(y + tile_size, H), min(x + tile_size, W)
            
            tile = image[y1:y2, x1:x2]
            
            # 如果边缘 tile 小于 1024，进行补齐，防止模型 forward 报错
            h, w = tile.shape
            if h < tile_size or w < tile_size:
                pad_tile = np.zeros((tile_size, tile_size), dtype=tile.dtype)
                pad_tile[:h, :w] = tile
                tile_tensor = preprocess_image_tile(pad_tile, config)
                prob_full, _ = predict_tile(model, tile_tensor, device, threshold)
                prob = prob_full[:h, :w] # 裁切回原始大小
            else:
                tile_tensor = preprocess_image_tile(tile, config)
                prob, _ = predict_tile(model, tile_tensor, device, threshold)
            
            pred_prob[y1:y2, x1:x2] += prob
            pred_count[y1:y2, x1:x2] += 1

    pred_prob = pred_prob / np.maximum(pred_count, 1)
    pred_binary = (pred_prob > threshold).astype(np.uint8) * 255
    pred_binary = remove_small_regions(pred_binary, min_area=min_area)
    return pred_prob, pred_binary

# =========================
# 保存叠加图 (Overlay)
# =========================
def save_overlay(image_gray, mask, save_path, alpha=0.4):
    image_bgr = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR)
    red_mask = np.zeros_like(image_bgr)
    red_mask[mask == 255] = [0, 0, 255] # 红色
    overlay = cv2.addWeighted(image_bgr, 1.0, red_mask, alpha, 0)
    cv2.imwrite(save_path, overlay)

# =========================
# 批量推理
# =========================
def batch_inference_and_save(image_dir, save_dir, model, config, device,
                             tile_size=1024, threshold=0.5, min_area=100):
    mask_dir = os.path.join(save_dir, "masks")
    overlay_dir = os.path.join(save_dir, "overlays")
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(overlay_dir, exist_ok=True)
    
    image_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(('.png','.jpg','.jpeg','.tif','.tiff'))])
    
    for img_file in image_files:
        image_path = os.path.join(image_dir, img_file)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None: continue
        
        _, pred_binary = tile_inference(image, model, config, device,
                                        tile_size=tile_size, threshold=threshold,
                                        min_area=min_area)
        
        base_name = os.path.splitext(img_file)[0]
        # 保存 Mask
        cv2.imwrite(os.path.join(mask_dir, base_name + ".png"), pred_binary)
        # 保存 Overlay
        save_overlay(image, pred_binary, os.path.join(overlay_dir, base_name + "_overlay.jpg"))
        
        print(f"✅ Finished: {img_file}")

# =========================
# 加载模型
# =========================
def load_model(model_path, config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(model_path, map_location=device)
    model_type = config.MODEL_TYPE

    # 接口选择
    if model_type == "SegFormer":
        from models.SegFormer import segformer_b1
        model = segformer_b1(img_size=1024, num_classes=1, stem_channels=32)
    elif model_type == "TransUNet":
        from models.TransUNet import transunet_paper_standard
        model = transunet_paper_standard(img_size=1024, num_classes=1, stem_channels=32)
    elif model_type == "SwinUNet":
        from models.SwinUNet import swin_unet_base
        model = swin_unet_base(img_size=1024, num_classes=1, stem_channels=32)
    elif model_type == "ConvNeXt":
        from models.unet import create_unet
        model = create_unet("ConvNeXt", num_classes=1)
    else:
        from models.unet import create_unet
        model = create_unet(model_type, num_classes=1, dropout_rates=[0.0]*4)
    
    # ✅ 必须加载权重
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    model.load_state_dict(state_dict)
    model.to(device).eval()
    return model, device

def main():
    config = Config()
    model_path = f'checkpoints/pth/best_{config.MODEL_TYPE}_model.pth'
    image_dir = '/home/xiaobowen/project/internal_wave_detection_project/output/new_data'
    save_dir = f'/home/xiaobowen/project/internal_wave_detection_project/output/results'

    model, device = load_model(model_path, config)
    # threshold 调回 0.5 增加检出率，min_area 过滤噪声
    batch_inference_and_save(image_dir, save_dir, model, config, device,
                             tile_size=1024, threshold=0.5, min_area=300)

if __name__ == '__main__':
    main()