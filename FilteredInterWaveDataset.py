"""
增强版InterWaveDataset，支持训练时动态过滤稀疏patches
"""
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
import os
from utils.InterWaveDataset import InterWaveDataset

class FilteredInterWaveDataset(InterWaveDataset):
    """
    带动态过滤的海洋内波数据集
    训练时会跳过正样本比例过低的patches
    """
    def __init__(self, image_paths, mask_paths, transform=None,
                 is_train=True, crop_size=1024, stride=None,
                 min_positive_ratio=0.0, max_empty_retries=50):
        """
        Args:
            min_positive_ratio: 最小正样本比例阈值（0-1）
            max_empty_retries: 最多重试次数
        """
        super().__init__(image_paths, mask_paths, transform, is_train, crop_size, stride)
        self.min_positive_ratio = min_positive_ratio
        self.max_empty_retries = max_empty_retries

        # 统计信息
        self.filtered_count = 0
        self.accepted_count = 0

    def __getitem__(self, idx):
        if self.is_train:
            # 训练模式：动态随机切片并过滤
            for retry_count in range(self.max_empty_retries):
                # 随机选择一张图像
                current_idx = idx if retry_count == 0 else np.random.randint(0, len(self.image_paths))
                img_path = self.image_paths[current_idx]
                mask_path = self.mask_paths[current_idx]
                img_np, mask_np = self.load_image_mask(img_path, mask_path)

                h, w = img_np.shape[:2]

                # 如果图像大于crop_size，需要切片
                if h > self.crop_size and w > self.crop_size:
                    # 尝试找到满足条件的切片
                    for patch_try in range(20):  # 每张图最多尝试20个位置
                        y = np.random.randint(0, h - self.crop_size)
                        x = np.random.randint(0, w - self.crop_size)
                        patch_mask = mask_np[y:y+self.crop_size, x:x+self.crop_size]

                        # 计算正样本比例
                        positive_ratio = np.mean(patch_mask > 0.5)

                        # 如果满足最小正样本比例要求，使用这个patch
                        if positive_ratio >= self.min_positive_ratio:
                            img_np = img_np[y:y+self.crop_size, x:x+self.crop_size]
                            mask_np = patch_mask
                            break
                    else:
                        # 20次都没找到合适的patch，继续下一张图
                        self.filtered_count += 1
                        continue
                else:
                    # 图像小于crop_size，需要padding
                    pad_h = max(0, self.crop_size - h)
                    pad_w = max(0, self.crop_size - w)
                    img_np = cv2.copyMakeBorder(img_np, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
                    mask_np = cv2.copyMakeBorder(mask_np, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)

                    # 检查正样本比例
                    positive_ratio = np.mean(mask_np > 0.5)
                    if positive_ratio < self.min_positive_ratio:
                        self.filtered_count += 1
                        continue

                # 应用数据增强
                if self.transform is not None:
                    augmented = self.transform(image=img_np, mask=mask_np)
                    img_np = augmented["image"]
                    mask_np = augmented["mask"]

                # 最终检查
                mask_sum = mask_np.sum().item() if isinstance(mask_np, torch.Tensor) else np.sum(mask_np)
                if mask_sum > 0:
                    self.accepted_count += 1
                    break

            else:
                # 所有重试都失败，返回最后一个（即使不满足条件）
                print(f"⚠️ Warning: After {self.max_empty_retries} retries, still couldn't find good patch for idx {idx}")

        else:
            # 验证模式：使用预定义的patches
            return super().__getitem__(idx)

        # 转换为tensor
        if isinstance(img_np, torch.Tensor):
            image = img_np.float()
            mask = mask_np if isinstance(mask_np, torch.Tensor) else torch.from_numpy(np.asarray(mask_np)).float()
        else:
            img_f = img_np.astype(np.float32) / 255.0
            img_f = (img_f - 0.5) / 0.5
            if img_f.ndim == 2:
                image = torch.from_numpy(img_f).unsqueeze(0)
            else:
                image = torch.from_numpy(img_f).permute(2, 0, 1)
            mask = torch.from_numpy(mask_np.astype(np.float32)).unsqueeze(0)

        return image, mask

    def get_statistics(self):
        """获取过滤统计信息"""
        total_attempts = self.filtered_count + self.accepted_count
        if total_attempts > 0:
            filter_rate = self.filtered_count / total_attempts
            print(f"📊 动态过滤统计：")
            print(f"   接受: {self.accepted_count} ({self.accepted_count/total_attempts*100:.1f}%)")
            print(f"   过滤: {self.filtered_count} ({filter_rate*100:.1f}%)")
            print(f"   阈值: {self.min_positive_ratio*100:.2f}%")
        return {
            'accepted': self.accepted_count,
            'filtered': self.filtered_count,
            'threshold': self.min_positive_ratio
        }