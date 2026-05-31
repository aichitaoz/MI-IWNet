"""
内波检测损失函数 - 简化版（已删除Uncertainty Weighting）

两层结构：
1. Modality Aware Loss - 处理三模态类别不平衡
2. Base Loss (HD + BCE) - 核心损失计算
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt, binary_erosion
import numpy as np


class HausdorffDistanceLoss(nn.Module):
    """Hausdorff Distance Loss"""
    def __init__(self, alpha=2.0, smooth=1e-6, use_gpu=True, max_iterations=50):
        super().__init__()
        self.alpha = alpha
        self.smooth = smooth
        self.use_gpu = use_gpu
        self.max_iterations = max_iterations

    def compute_distance_map_gpu(self, mask):
        B, C, H, W = mask.shape
        device, dtype = mask.device, mask.dtype
        kernel = torch.ones(1, 1, 3, 3, device=device, dtype=dtype)
        eroded = -F.max_pool2d(-mask, kernel_size=3, stride=1, padding=1)
        boundary = mask - eroded
        dist_map = torch.zeros_like(mask)
        current_region = boundary.clone()

        for i in range(1, self.max_iterations):
            dilated = F.max_pool2d(current_region, kernel_size=3, stride=1, padding=1)
            new_region = dilated * (1 - current_region) * mask
            if new_region.sum() == 0:
                break
            dist_map = dist_map + new_region * i
            current_region = current_region + new_region

        max_dist = dist_map.view(B, -1).max(dim=1, keepdim=True)[0].view(B, 1, 1, 1).clamp(min=1.0)
        return dist_map / max_dist

    def compute_distance_map(self, mask):
        batch_size = mask.shape[0]
        distance_maps = []
        for b in range(batch_size):
            mask_np = mask[b, 0].cpu().numpy()
            boundary = mask_np - binary_erosion(mask_np)
            if boundary.sum() == 0:
                dist_map = np.zeros_like(mask_np, dtype=np.float32)
            else:
                dist_map = distance_transform_edt(1 - boundary)
                max_dist = dist_map.max()
                if max_dist > 0:
                    dist_map = dist_map / max_dist
            distance_maps.append(dist_map)
        distance_maps = np.stack(distance_maps, axis=0)[:, np.newaxis, :, :]
        return torch.from_numpy(distance_maps).to(mask.device)

    def forward(self, pred, target):
        original_dtype = pred.dtype
        with torch.no_grad():
            dist_map = self.compute_distance_map_gpu(target) if self.use_gpu else self.compute_distance_map(target.float()).to(dtype=original_dtype)
        pred_prob = torch.sigmoid(pred)
        fp_loss = (pred_prob * (1 - target) * torch.pow(1.0 + dist_map, self.alpha)).sum()
        fn_loss = ((1 - pred_prob) * target * torch.pow(1.0 + dist_map, self.alpha)).sum()
        return (fp_loss + fn_loss) / (target.size(0) * target[0].numel() + self.smooth)


class ModernInternalWaveLoss(nn.Module):
    """基础损失：HD + BCE"""
    def __init__(self, weight_hd=0.5, weight_bce=0.5, hd_alpha=2.0, hd_use_gpu=True, pos_weight=3.0):
        super().__init__()
        self.weight_hd, self.weight_bce, self.pos_weight = weight_hd, weight_bce, pos_weight
        self.hd_loss = HausdorffDistanceLoss(alpha=hd_alpha, use_gpu=hd_use_gpu)

    def bce_loss(self, pred, target):
        return F.binary_cross_entropy_with_logits(pred, target, pos_weight=torch.tensor(self.pos_weight, device=pred.device))

    def forward(self, pred, target):
        loss_hd, loss_bce = self.hd_loss(pred, target), self.bce_loss(pred, target)
        return self.weight_hd * loss_hd + self.weight_bce * loss_bce, loss_hd, loss_bce


class ModalityAwareLoss(nn.Module):
    """模态感知损失"""
    def __init__(self, base_loss, sar_pos_weight=5.0, rgb_pos_weight=100.0):
        super().__init__()
        self.base_loss = base_loss
        self.sar_pos_weight, self.rgb_pos_weight = sar_pos_weight, rgb_pos_weight
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        print(f"✅ ModalityAwareLoss: SAR={sar_pos_weight}, RGB={rgb_pos_weight}")

    def forward(self, outputs, masks, datasets=None):
        if datasets is None or len(set(datasets)) == 1:
            return self.base_loss(outputs, masks)

        modality_aware_bce, total_samples = 0.0, 0
        for modality, pos_weight in [('sar', self.sar_pos_weight), ('rgb', self.rgb_pos_weight), ('sdg', self.sdg_pos_weight)]:
            indices = [i for i, d in enumerate(datasets) if d == modality]
            if len(indices) > 0:
                mod_outputs, mod_masks = outputs[indices].squeeze(1), masks[indices].squeeze(1)
                bce_raw = self.bce(mod_outputs, mod_masks)
                weights = torch.where(mod_masks == 1, torch.tensor([pos_weight], device=outputs.device), torch.ones_like(mod_masks))
                modality_aware_bce += (bce_raw * weights).mean() * len(indices)
                total_samples += len(indices)

        modality_aware_bce /= (total_samples + 1e-8)
        _, loss_hd, _ = self.base_loss(outputs, masks)
        return self.base_loss.weight_bce * modality_aware_bce + self.base_loss.weight_hd * loss_hd, loss_hd, modality_aware_bce


def create_internal_wave_loss(sar_pos_weight=50.0, rgb_pos_weight=300.0,
                               weight_hd=0.2, weight_bce=0.8, hd_alpha=1.5, hd_use_gpu=True):
    """创建损失函数"""
    base_loss = ModernInternalWaveLoss(weight_hd, weight_bce, hd_alpha, hd_use_gpu, sar_pos_weight)
    return ModalityAwareLoss(base_loss, sar_pos_weight, rgb_pos_weight)