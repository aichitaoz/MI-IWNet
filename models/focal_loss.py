"""
Focal Loss implementation for extreme class imbalance
专门针对RGB极低正样本率设计
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss - 专门解决极端类别不平衡
    通过降低易分类样本的权重，让模型关注难分类的样本
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        """
        Args:
            alpha: 平衡因子，通常设为少数类的比例
            gamma: 聚焦参数，越大越关注难分类样本（推荐2.0）
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs: 模型输出的logits (B, H, W) or (B, 1, H, W)
            targets: 真实标签 (B, H, W) or (B, 1, H, W)
        """
        # 确保维度匹配
        if inputs.dim() == 4:
            inputs = inputs.squeeze(1)
        if targets.dim() == 4:
            targets = targets.squeeze(1)

        # 计算概率
        p = torch.sigmoid(inputs)

        # 计算CE loss
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')

        # 计算p_t
        p_t = p * targets + (1 - p) * (1 - targets)

        # 计算focal weight: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma

        # 计算alpha weight
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        # 应用focal weight和alpha
        focal_loss = alpha_t * focal_weight * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class CombinedFocalHDLoss(nn.Module):
    """
    组合Focal Loss和HD Loss
    Focal处理类别不平衡，HD处理边界
    """
    def __init__(self, focal_alpha=0.25, focal_gamma=2.0,
                 weight_focal=0.8, weight_hd=0.2,
                 hd_alpha=1.5, hd_use_gpu=True):
        super().__init__()
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.weight_focal = weight_focal
        self.weight_hd = weight_hd

        # 复用现有的HD Loss
        from models.unified_loss import HausdorffDistanceLoss
        self.hd_loss = HausdorffDistanceLoss(alpha=hd_alpha, use_gpu=hd_use_gpu)

    def forward(self, pred, target):
        # Focal Loss
        loss_focal = self.focal_loss(pred, target)

        # HD Loss
        loss_hd = self.hd_loss(pred, target)

        # 组合
        total_loss = self.weight_focal * loss_focal + self.weight_hd * loss_hd

        return total_loss, loss_hd, loss_focal


class ModalityAwareFocalLoss(nn.Module):
    """
    模态感知的Focal Loss
    为SAR和RGB使用不同的alpha值
    """
    def __init__(self, sar_alpha=0.25, rgb_alpha=0.01, gamma=2.0,
                 weight_focal=0.8, weight_hd=0.2):
        super().__init__()
        self.sar_alpha = sar_alpha
        self.rgb_alpha = rgb_alpha
        self.gamma = gamma
        self.weight_focal = weight_focal
        self.weight_hd = weight_hd

        # HD Loss
        from models.unified_loss import HausdorffDistanceLoss
        self.hd_loss = HausdorffDistanceLoss(alpha=1.5, use_gpu=True)

    def forward(self, outputs, masks, datasets=None):
        if datasets is None:
            datasets = ['sar'] * len(outputs)

        total_focal = 0
        total_hd = 0
        total_samples = 0

        for i, dataset in enumerate(datasets):
            # 选择合适的alpha
            if dataset == 'rgb':
                alpha = self.rgb_alpha  # RGB用更小的alpha，因为正样本极少
            else:
                alpha = self.sar_alpha

            # 创建Focal Loss
            focal = FocalLoss(alpha=alpha, gamma=self.gamma)

            # 计算损失
            pred_i = outputs[i:i+1]
            mask_i = masks[i:i+1]

            loss_focal = focal(pred_i, mask_i)
            loss_hd = self.hd_loss(pred_i, mask_i)

            total_focal += loss_focal
            total_hd += loss_hd
            total_samples += 1

        # 平均
        avg_focal = total_focal / (total_samples + 1e-8)
        avg_hd = total_hd / (total_samples + 1e-8)

        total_loss = self.weight_focal * avg_focal + self.weight_hd * avg_hd

        return total_loss, avg_hd, avg_focal