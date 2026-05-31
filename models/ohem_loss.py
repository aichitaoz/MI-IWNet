"""
Online Hard Example Mining (OHEM) for RGB
在线困难样本挖掘 - 只在最需要学习的像素上计算损失
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class OHEMCrossEntropyLoss(nn.Module):
    """
    Online Hard Example Mining
    只选择损失最大的像素来反向传播，避免被大量简单负样本淹没
    """
    def __init__(self, thresh=0.7, min_kept=100000, ignore_index=-1):
        super().__init__()
        self.thresh = thresh  # 损失阈值
        self.min_kept = min_kept  # 最少保留的像素数
        self.ignore_index = ignore_index

    def forward(self, logits, labels):
        """
        logits: (B, 1, H, W)
        labels: (B, 1, H, W)
        """
        B, _, H, W = logits.shape
        logits = logits.squeeze(1)  # (B, H, W)
        labels = labels.squeeze(1)  # (B, H, W)

        # 计算每个像素的损失
        losses = F.binary_cross_entropy_with_logits(logits, labels, reduction='none')
        losses = losses.view(-1)  # (B*H*W,)

        # 排序找到困难样本
        sorted_losses, _ = torch.sort(losses, descending=True)

        # 确定保留多少像素
        # 方案1：保留所有正样本 + top K 负样本
        labels_flat = labels.view(-1)
        num_pos = (labels_flat > 0.5).sum().item()

        # 至少保留 min_kept 个像素，或所有正样本+等量负样本
        keep_num = max(self.min_kept, num_pos * 2)
        keep_num = min(keep_num, losses.numel())

        # 找到阈值
        if keep_num < losses.numel():
            threshold = sorted_losses[keep_num]
        else:
            threshold = 0

        # 创建mask：保留所有正样本 + 高损失负样本
        pos_mask = labels_flat > 0.5
        hard_neg_mask = (labels_flat <= 0.5) & (losses > threshold)
        keep_mask = pos_mask | hard_neg_mask

        # 确保至少保留min_kept个像素
        if keep_mask.sum() < self.min_kept:
            # 如果保留的太少，添加更多高损失样本
            _, indices = torch.topk(losses, min(self.min_kept, losses.numel()))
            keep_mask[indices] = True

        # 只在选中的像素上计算损失
        selected_losses = losses[keep_mask]

        if selected_losses.numel() > 0:
            return selected_losses.mean()
        else:
            return losses.mean()  # fallback


class BalancedFocalLoss(nn.Module):
    """
    平衡的Focal Loss - 自动平衡正负样本
    """
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits, labels):
        """
        logits: (B, 1, H, W)
        labels: (B, 1, H, W)
        """
        logits = logits.squeeze(1)
        labels = labels.squeeze(1)

        # 计算概率
        probs = torch.sigmoid(logits)

        # 计算正负样本数量
        num_pos = (labels > 0.5).float().sum()
        num_neg = (labels <= 0.5).float().sum()

        # 自动计算alpha（平衡因子）
        total = num_pos + num_neg
        alpha_pos = (num_neg / total).clamp(min=0.01, max=0.99)
        alpha_neg = (num_pos / total).clamp(min=0.01, max=0.99)

        # Focal Loss计算
        ce_loss = F.binary_cross_entropy_with_logits(logits, labels, reduction='none')
        p_t = probs * labels + (1 - probs) * (1 - labels)
        focal_weight = (1 - p_t) ** self.gamma

        # 应用平衡权重
        alpha_t = alpha_pos * labels + alpha_neg * (1 - labels)
        focal_loss = alpha_t * focal_weight * ce_loss

        return focal_loss.mean()


class RGBAdaptiveLoss(nn.Module):
    """
    RGB自适应损失 - 根据batch中正样本数量自动调整策略
    """
    def __init__(self):
        super().__init__()
        self.ohem = OHEMCrossEntropyLoss(min_kept=10000)  # 至少保留1%的像素
        self.focal = BalancedFocalLoss(gamma=2.0)
        self.dice_loss = self.dice

    def dice(self, logits, labels):
        """Dice Loss"""
        probs = torch.sigmoid(logits.squeeze(1))
        labels = labels.squeeze(1)

        smooth = 1.0
        intersection = (probs * labels).sum()
        union = probs.sum() + labels.sum()
        dice = (2 * intersection + smooth) / (union + smooth)
        return 1 - dice

    def forward(self, logits, labels, datasets):
        """
        动态选择损失策略
        返回: (total_loss, focal_loss, ohem_loss, dice_loss)
        """
        # 分离SAR和RGB
        sar_indices = [i for i, d in enumerate(datasets) if d == 'sar']
        rgb_indices = [i for i, d in enumerate(datasets) if d == 'rgb']

        total_loss = 0
        total_focal = 0  # Focal损失累计
        total_ohem = 0   # OHEM损失累计
        total_dice = 0   # Dice损失累计
        total_samples = 0

        # SAR使用标准Focal + Dice
        if len(sar_indices) > 0:
            sar_logits = logits[sar_indices]
            sar_labels = labels[sar_indices]
            sar_focal = self.focal(sar_logits, sar_labels)
            sar_dice = self.dice_loss(sar_logits, sar_labels)
            sar_loss = 0.6 * sar_focal + 0.4 * sar_dice

            total_loss += sar_loss * len(sar_indices)
            total_focal += sar_focal * len(sar_indices)  # SAR贡献到Focal
            total_dice += sar_dice * len(sar_indices)
            total_samples += len(sar_indices)

        # RGB使用OHEM + Dice 或 Focal + Dice
        if len(rgb_indices) > 0:
            rgb_logits = logits[rgb_indices]
            rgb_labels = labels[rgb_indices]

            # 检查正样本比例
            pos_ratio = (rgb_labels > 0.5).float().mean()

            rgb_dice = self.dice_loss(rgb_logits, rgb_labels)

            if pos_ratio < 0.001:  # 极度稀疏，使用OHEM
                rgb_ohem = self.ohem(rgb_logits, rgb_labels)
                rgb_loss = 0.3 * rgb_ohem + 0.7 * rgb_dice
                total_ohem += rgb_ohem * len(rgb_indices)  # RGB贡献到OHEM
            else:  # 相对正常，使用Focal
                rgb_focal = self.focal(rgb_logits, rgb_labels)
                rgb_loss = 0.5 * rgb_focal + 0.5 * rgb_dice
                total_focal += rgb_focal * len(rgb_indices)  # RGB贡献到Focal

            total_loss += rgb_loss * len(rgb_indices)
            total_dice += rgb_dice * len(rgb_indices)
            total_samples += len(rgb_indices)

        # 避免除以零
        if total_samples == 0:
            return torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0)

        # 返回平均损失
        avg_total = total_loss / total_samples
        avg_focal = total_focal / total_samples if total_focal > 0 else torch.tensor(0.0)
        avg_ohem = total_ohem / total_samples if total_ohem > 0 else torch.tensor(0.0)
        avg_dice = total_dice / total_samples

        return avg_total, avg_focal, avg_ohem, avg_dice