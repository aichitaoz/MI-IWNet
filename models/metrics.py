import torch
import numpy as np
import cv2

def boundary_f1_score(pred, target, dilation_ratio=0.02):
    """
    Boundary F1 Score (BF Score)
    Args:
        pred: 预测二值图 (numpy array)
        target: 真实标签二值图 (numpy array)
        dilation_ratio: 边界膨胀比例 (相对于图像对角线长度)
    Returns:
        float: BF Score
    """
    h, w = pred.shape
    diag_len = np.sqrt(h**2 + w**2)
    dilation = max(1, int(round(dilation_ratio * diag_len)))

    # 提取边界
    pred_boundary = pred - cv2.erode(pred, np.ones((3, 3), np.uint8))
    target_boundary = target - cv2.erode(target, np.ones((3, 3), np.uint8))

    # 边界膨胀
    pred_dil = cv2.dilate(pred_boundary, np.ones((dilation, dilation), np.uint8))
    target_dil = cv2.dilate(target_boundary, np.ones((dilation, dilation), np.uint8))

    # 计算 Precision 和 Recall
    precision = (pred_boundary & target_dil).sum() / (pred_boundary.sum() + 1e-8)
    recall = (target_boundary & pred_dil).sum() / (target_boundary.sum() + 1e-8)

    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def calculate_metrics(pred, target, threshold=0.9):
    """
    计算分割指标（Relaxed IoU, Relaxed Dice, Precision, Recall, Accuracy, BF Score）

    ✅ 修复：在整个batch上计算，而不是只用第一个样本
    ✅ 使用 Relaxed IoU/Dice：对预测和真值进行5像素膨胀后计算，容忍边界偏移

    Args:
        pred: 模型预测输出 (torch.Tensor) [B, C, H, W]
        target: 真实标签 (torch.Tensor) [B, C, H, W]
        threshold: 二值化阈值
    Returns:
        dict: 包含各种指标的字典（batch平均值）
              IoU/Dice 为 Relaxed 版本（5像素tolerance）
    """
    pred_prob = torch.sigmoid(pred).detach().cpu().numpy()
    target_np = target.cpu().numpy()

    # ✅ 处理维度：保持batch维度
    if pred_prob.ndim == 4:
        # [B, C, H, W] -> [B, H, W]
        pred_prob = pred_prob[:, 0, :, :]
    elif pred_prob.ndim == 3:
        # [B, H, W] 已经是正确格式
        pass
    elif pred_prob.ndim == 2:
        # [H, W] -> [1, H, W]
        pred_prob = pred_prob[np.newaxis, ...]

    if target_np.ndim == 4:
        # [B, C, H, W] -> [B, H, W]
        target_np = target_np[:, 0, :, :]
    elif target_np.ndim == 3:
        # [B, H, W] 已经是正确格式
        pass
    elif target_np.ndim == 2:
        # [H, W] -> [1, H, W]
        target_np = target_np[np.newaxis, ...]

    pred_binary = (pred_prob > threshold).astype(np.uint8)
    mask_binary = target_np.astype(np.uint8)

    batch_size = pred_binary.shape[0]

    # ✅ Relaxed IoU/Dice: 对预测和真值进行膨胀，容忍5像素偏移
    kernel = np.ones((5, 5), np.uint8)

    # ✅ 在整个batch上累积混淆矩阵
    tp_total = 0
    fp_total = 0
    fn_total = 0
    tn_total = 0
    bf_scores = []

    for i in range(batch_size):
        pred_i = pred_binary[i]
        mask_i = mask_binary[i]

        # ✅ Relaxed版本：膨胀后计算混淆矩阵
        pred_relaxed = cv2.dilate(pred_i, kernel)
        mask_relaxed = cv2.dilate(mask_i, kernel)

        # 混淆矩阵（使用relaxed版本）
        tp = np.logical_and(pred_relaxed == 1, mask_relaxed == 1).sum()
        fp = np.logical_and(pred_relaxed == 1, mask_relaxed == 0).sum()
        fn = np.logical_and(pred_relaxed == 0, mask_relaxed == 1).sum()
        tn = np.logical_and(pred_relaxed == 0, mask_relaxed == 0).sum()

        tp_total += tp
        fp_total += fp
        fn_total += fn
        tn_total += tn

        # BF Score需要逐样本计算（涉及形态学操作）
        bf_i = boundary_f1_score(pred_i, mask_i)
        bf_scores.append(bf_i)

    # ✅ 基于累积的混淆矩阵计算整体指标
    metrics = {}

    # IoU
    iou = tp_total / (tp_total + fp_total + fn_total + 1e-8)
    metrics['IoU'] = iou

    # Dice
    dice = 2 * tp_total / (2 * tp_total + fp_total + fn_total + 1e-8)
    metrics['Dice'] = dice

    # Precision
    precision = tp_total / (tp_total + fp_total + 1e-8)
    metrics['Precision'] = precision

    # Recall
    recall = tp_total / (tp_total + fn_total + 1e-8)
    metrics['Recall'] = recall

    # Accuracy
    accuracy = (tp_total + tn_total) / (tp_total + tn_total + fp_total + fn_total + 1e-8)
    metrics['Accuracy'] = accuracy

    # BF Score (batch平均)
    bf = np.mean(bf_scores) if len(bf_scores) > 0 else 0.0
    metrics['BF_Score'] = bf

    return metrics
