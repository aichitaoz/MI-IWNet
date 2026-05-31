"""
交替训练 DataLoader - 为方案A设计

策略：每个batch只包含一张图（SAR或RGB交替）
显存占用 = 原来的一半
支持梯度累积：ACCUMULATION_STEPS=4 时等效 batch_size=4（2 SAR + 2 RGB）

示例：
    Batch 0: 1张 SAR
    Batch 1: 1张 RGB
    Batch 2: 1张 SAR
    Batch 3: 1张 RGB
    ...
"""

import torch
import random



class AlternatingModalityLoaderBalanced:
    """
    平衡版交替加载器 - 数量少的模态随机采样补齐

    策略：
    - 数量多的全部使用
    - 数量少的随机采样补齐（避免重复使用同一批数据）
    - ✅ 新策略：SAR:RGB = 1:2 交替训练
    - ✅ 已优化：依赖Subset预过滤，不再需要运行时检测空mask

    使用场景：当 SAR、RGB 数量差异很大时，确保训练平衡
    """
    def __init__(self, loader_sar, loader_rgb, is_train=True, sar_rgb_ratio=(1, 2)):
        """
        Args:
            loader_sar: SAR DataLoader
            loader_rgb: RGB DataLoader
            is_train: 是否为训练模式
            sar_rgb_ratio: SAR:RGB的比例，默认(1,2)表示1个SAR、2个RGB
        """
        self.loader_sar = loader_sar
        self.loader_rgb = loader_rgb
        self.is_train = is_train  # ✅ 区分训练/验证模式
        self.sar_ratio, self.rgb_ratio = sar_rgb_ratio  # ✅ 2个比例

        # ✅ 保存 dataset 引用，用于随机采样
        self.dataset_sar = loader_sar.dataset
        self.dataset_rgb = loader_rgb.dataset

        # ✅ 已移除：不再需要扫描非空mask，因为数据集已经通过Subset预过滤

        # ✅ 新策略：总长度 = (sar_ratio + rgb_ratio) × min
        # 使用最小数据集的长度，避免某个模态过度训练，减少梯度冲突
        # 例如：ratio=(1,2)时，每轮产生1个SAR + 2个RGB = 3个batch
        self.max_len = min(len(loader_sar), len(loader_rgb))  # ✅ 改为min（最小数据集）
        self.total_len = self.max_len * (self.sar_ratio + self.rgb_ratio)


    def __len__(self):
        return self.total_len

    def __iter__(self):
        """按照 SAR:RGB 比例交替产生batch"""
        iter_sar = iter(self.loader_sar)
        iter_rgb = iter(self.loader_rgb)

        for i in range(self.max_len):
            # ===== 产出 sar_ratio 个 SAR batch =====
            for _ in range(self.sar_ratio):
                sar_images, sar_masks = None, None
                try:
                    sar_images, sar_masks = next(iter_sar)
                    # ✅ 已移除：不再需要检测空mask，数据集已过滤
                except StopIteration:
                    # ✅ SAR 用完，从数据集中随机采样
                    idx = torch.randint(0, len(self.dataset_sar), (1,)).item()
                    sar_images, sar_masks = self.dataset_sar[idx]
                    # 确保是4D tensor (batch维度)
                    if sar_images.dim() == 3:
                        sar_images = sar_images.unsqueeze(0)
                    if sar_masks.dim() == 3:
                        sar_masks = sar_masks.unsqueeze(0)
                yield [sar_images], sar_masks, ['sar']

            # ===== 产出 rgb_ratio 个 RGB batch =====
            for _ in range(self.rgb_ratio):
                rgb_images, rgb_masks = None, None
                try:
                    rgb_images, rgb_masks = next(iter_rgb)
                    # ✅ 已移除：不再需要检测空mask，数据集已过滤
                except StopIteration:
                    # ✅ RGB 用完，从数据集中随机采样
                    idx = torch.randint(0, len(self.dataset_rgb), (1,)).item()
                    rgb_images, rgb_masks = self.dataset_rgb[idx]
                    # 确保是4D tensor (batch维度)
                    if rgb_images.dim() == 3:
                        rgb_images = rgb_images.unsqueeze(0)
                    if rgb_masks.dim() == 3:
                        rgb_masks = rgb_masks.unsqueeze(0)
                yield [rgb_images], rgb_masks, ['rgb']
