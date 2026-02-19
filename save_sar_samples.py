"""
保存所有SAR数据集样本为图片格式（png）
"""
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

from utils.InterWaveDataset import InterWaveDataset
from utils.get_transforms import get_transforms
from utils.prepare_data import prepare_data


def tensor_to_image(tensor):
    """
    将tensor转换为可保存的图像格式
    tensor: [C, H, W] 或 [1, H, W]
    返回: numpy array [H, W] 或 [H, W, C], uint8格式
    """
    # 转为numpy
    img_np = tensor.cpu().numpy()
    
    # 处理通道
    if img_np.shape[0] == 1:  # 单通道 [1, H, W]
        img_np = img_np.squeeze(0)  # [H, W]
    else:  # 多通道 [C, H, W]
        img_np = img_np.transpose(1, 2, 0)  # [H, W, C]
    
    # 归一化到 [0, 255]
    img_min, img_max = img_np.min(), img_np.max()
    img_np = (img_np - img_min) / (img_max - img_min + 1e-8)
    img_np = (img_np * 255).astype(np.uint8)
    
    return img_np


def save_all_sar_dataset(config):
    """
    保存所有SAR数据集样本为图片
    """
    print("\n" + "="*60)
    print("🚀 开始保存所有 SAR 数据集为图片")
    print("="*60 + "\n")
    
    # ===== 1. 准备数据 =====
    print("📂 准备数据路径...")
    all_data = prepare_data(config)
    (sar_train_imgs, sar_train_masks), (sar_val_imgs, sar_val_masks), _ = all_data["SAR"]
    print(f"   训练集: {len(sar_train_imgs)} 张")
    print(f"   验证集: {len(sar_val_imgs)} 张\n")
    
    # ===== 2. 创建 Dataset =====
    print("📦 创建 InterWaveDataset...")
    sar_train_transform = get_transforms(config.IMG_SIZE, is_train=True, is_sar=True)
    sar_val_transform = get_transforms(config.IMG_SIZE, is_train=False, is_sar=True)
    
    sar_train_dataset = InterWaveDataset(
        sar_train_imgs, sar_train_masks, 
        sar_train_transform, 
        is_train=True
    )
    sar_val_dataset = InterWaveDataset(
        sar_val_imgs, sar_val_masks, 
        sar_val_transform, 
        is_train=False
    )
    print(f"   训练集: {len(sar_train_dataset)} 个样本")
    print(f"   验证集: {len(sar_val_dataset)} 个样本\n")
    
    # ===== 3. 保存训练集 =====
    print(f"💾 保存训练集样本 ({len(sar_train_dataset)} 个)...")
    train_save_dir = Path("./sar_dataset_images/train")
    train_save_dir.mkdir(parents=True, exist_ok=True)
    
    success_count = 0
    for idx in tqdm(range(len(sar_train_dataset)), desc="训练集"):
        try:
            image, mask = sar_train_dataset[idx]
            
            # 转换为图像格式
            image_np = tensor_to_image(image)
            mask_np = (mask.cpu().numpy().squeeze(0) * 255).astype(np.uint8)
            
            # 保存图像
            cv2.imwrite(str(train_save_dir / f"train_{idx:04d}_image.png"), image_np)
            cv2.imwrite(str(train_save_dir / f"train_{idx:04d}_mask.png"), mask_np)
            
            success_count += 1
            
        except Exception as e:
            print(f"\n   ❌ 训练集样本 {idx} 失败: {e}")
    
    print(f"   ✅ 训练集保存成功: {success_count}/{len(sar_train_dataset)}\n")
    
    # ===== 4. 保存验证集 =====
    print(f"💾 保存验证集样本 ({len(sar_val_dataset)} 个)...")
    val_save_dir = Path("./sar_dataset_images/val")
    val_save_dir.mkdir(parents=True, exist_ok=True)
    
    success_count = 0
    for idx in tqdm(range(len(sar_val_dataset)), desc="验证集"):
        try:
            image, mask = sar_val_dataset[idx]
            
            # 转换为图像格式
            image_np = tensor_to_image(image)
            mask_np = (mask.cpu().numpy().squeeze(0) * 255).astype(np.uint8)
            
            # 保存图像
            cv2.imwrite(str(val_save_dir / f"val_{idx:04d}_image.png"), image_np)
            cv2.imwrite(str(val_save_dir / f"val_{idx:04d}_mask.png"), mask_np)
            
            success_count += 1
            
        except Exception as e:
            print(f"\n   ❌ 验证集样本 {idx} 失败: {e}")
    
    print(f"   ✅ 验证集保存成功: {success_count}/{len(sar_val_dataset)}\n")
    
    # ===== 5. 完成 =====
    print("="*60)
    print("✅ 保存完成！")
    print(f"📁 保存路径: ./sar_dataset_images/")
    print(f"   - train/: {len(sar_train_dataset)} 个样本（图片格式）")
    print(f"   - val/: {len(sar_val_dataset)} 个样本（图片格式）")
    print("="*60 + "\n")


if __name__ == "__main__":
    from configs.train_config import Config
    
    config = Config()
    
    # 保存所有样本为图片
    save_all_sar_dataset(config)