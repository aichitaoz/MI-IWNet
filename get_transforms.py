import albumentations as A
import cv2

def get_transforms(img_size, is_train=True, is_sar=False, is_rgb=False):
    if is_train:
        if is_sar:
            return A.Compose([
                A.RandomRotate90(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.GaussNoise(var_limit=(5, 25), p=0.15),  # ✅ 降低噪声强度和概率
            ])
        elif is_rgb:
            return A.Compose([
                A.RandomRotate90(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05, p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
                A.CLAHE(clip_limit=2.0, p=0.3),
                A.GaussNoise(var_limit=(5, 25), p=0.2),
            ])
    else:
        # 验证集不需要增强
        return A.Compose([])
