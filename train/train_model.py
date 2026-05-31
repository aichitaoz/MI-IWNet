import os
import torch
from torch.utils.data import Subset
from utils.InterWaveDataset import InterWaveDataset
from utils.FilteredInterWaveDataset import FilteredInterWaveDataset  # ✅ 新增：动态过滤的Dataset
from utils.prepare_data import prepare_data
from utils.get_transforms import get_transforms

from train.train_loop import train_epoch, validate_epoch
from models.unified_loss import create_internal_wave_loss  # ✅ 统一损失函数
import matplotlib.pyplot as plt
from lion_pytorch import Lion
from itertools import zip_longest
from train.alternating_loader import AlternatingModalityLoaderBalanced  # ✅ 方案A：交替训练


def get_non_empty_indices(dataset, modality_name='Dataset'):
    """
    扫描数据集,返回所有非空mask的样本索引
    ✅ 支持 patch_index 索引
    """
    import cv2
    import numpy as np

    valid_indices = []
    print(f"🔍 扫描{modality_name}非空mask样本（检查原始mask文件）...")

    if hasattr(dataset, 'dataset'):  # Subset
        original_dataset = dataset.dataset
        indices_map = dataset.indices
    else:
        original_dataset = dataset
        indices_map = range(len(dataset))

    if not dataset.is_train:
        # 验证集有 patch_index
        patch_index = original_dataset.patch_index
        for idx_in_patch, (img_idx, y, x) in enumerate(patch_index):
            mask_path = original_dataset.mask_paths[img_idx]
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                print(f"⚠️  {modality_name} patch {idx_in_patch} mask文件读取失败: {mask_path}")
                continue
            patch_mask = mask[y:y+original_dataset.crop_size, x:x+original_dataset.crop_size]
            if np.sum(patch_mask > 127) > 0:
                valid_indices.append(idx_in_patch)
    else:
        # 训练集按原图扫描
        for idx in indices_map:
            mask_path = original_dataset.mask_paths[idx]
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                print(f"⚠️  {modality_name}样本 {idx} mask文件读取失败: {mask_path}")
                continue
            if np.sum(mask > 127) > 0:
                valid_indices.append(idx)

    total = len(original_dataset.patch_index) if not dataset.is_train else len(original_dataset)
    valid = len(valid_indices)
    empty = total - valid
    print(f"✅ {modality_name}: {valid}/{total} 个非空样本 (过滤掉 {empty} 个原始mask全黑的样本)")

    return valid_indices



def plot_training_curves(config, num_epochs,
                         train_losses, val_losses,
                         train_loss1, val_loss1,
                         train_loss2, val_loss2,
                         train_loss3, val_loss3,  # 新增Dice损失
                         train_metrics_list, val_metrics_list):
    os.makedirs(config.FIG_SAVE_DIR, exist_ok=True)

    # ✅ 使用实际训练的轮数，而不是配置的总轮数（处理Early Stopping情况）
    actual_epochs = len(train_losses)
    epochs = range(1, actual_epochs + 1)

    # ------------------ 绘制总 loss ------------------
    plt.figure(figsize=(8,6))
    plt.plot(epochs, train_losses, label='Train Total Loss', marker='o')
    plt.plot(epochs, val_losses, label='Val Total Loss', marker='o')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('Total Loss')
    plt.legend(); plt.grid(True)
    plt.savefig(os.path.join(config.FIG_SAVE_DIR, f'{config.MODEL_TYPE}_total_loss.png'))
    plt.close()

    # ------------------ 绘制分解 loss ------------------
    plt.figure(figsize=(12,8))

    # 子图1: Focal Loss
    plt.subplot(2, 2, 1)
    plt.plot(epochs, train_loss1, label='Train Focal', marker='o', color='blue')
    plt.plot(epochs, val_loss1, label='Val Focal', marker='s', color='lightblue')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('Focal Loss')
    plt.legend(); plt.grid(True)

    # 子图2: OHEM Loss
    plt.subplot(2, 2, 2)
    plt.plot(epochs, train_loss2, label='Train OHEM', marker='o', color='red')
    plt.plot(epochs, val_loss2, label='Val OHEM', marker='s', color='lightcoral')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('OHEM Loss')
    plt.legend(); plt.grid(True)

    # 子图3: Dice Loss
    plt.subplot(2, 2, 3)
    plt.plot(epochs, train_loss3, label='Train Dice', marker='o', color='green')
    plt.plot(epochs, val_loss3, label='Val Dice', marker='s', color='lightgreen')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('Dice Loss')
    plt.legend(); plt.grid(True)

    # 子图4: 所有损失对比
    plt.subplot(2, 2, 4)
    plt.plot(epochs, train_loss1, label='Focal', marker='o', alpha=0.7)
    plt.plot(epochs, train_loss2, label='OHEM', marker='s', alpha=0.7)
    plt.plot(epochs, train_loss3, label='Dice', marker='^', alpha=0.7)
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('All Loss Components (Val)')
    plt.legend(); plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(config.FIG_SAVE_DIR, f'{config.MODEL_TYPE}_loss_components.png'))
    plt.close()

    # ------------------ 绘制指标 ------------------
    metric_names = train_metrics_list[0].keys()
    for metric in metric_names:
        train_vals = [m[metric] for m in train_metrics_list]
        val_vals = [m[metric] for m in val_metrics_list]

        plt.figure(figsize=(8,6))
        plt.plot(epochs, train_vals, label=f'Train {metric}', marker='o')
        plt.plot(epochs, val_vals, label=f'Val {metric}', marker='o')
        plt.xlabel('Epoch'); plt.ylabel(metric); plt.title(f'{metric} Curve')
        plt.legend(); plt.grid(True)
        plt.savefig(os.path.join(config.FIG_SAVE_DIR, f'{config.MODEL_TYPE}_{metric}_curve.png'))
        plt.close()
        
def train_model(config, feature_save_dir=None):
    """
    训练模型的主函数
    """
    # =================== 1️⃣ 创建保存目录 ===================
    os.makedirs(config.FEATURE_SAVE_DIR, exist_ok=True)
    os.makedirs(config.FIG_SAVE_DIR, exist_ok=True)
    os.makedirs(config.PTH_SAVE_DIR, exist_ok=True)
    
    # =================== 2️⃣ 准备数据集 ===================
    # 分别接收 SAR 数据
    # =================== 2️⃣ 准备数据集（只调用一次prepare_data）===================
    print("准备数据集...")
    all_data = prepare_data(config)  # ✅ 只调用一次，避免重复扫描
    
    # 分别接收 SAR 数据
    (sar_train_imgs, sar_train_masks), (sar_val_imgs, sar_val_masks), _ = all_data["SAR"]

    # 分别接收 RGB 数据
    (rgb_train_imgs, rgb_train_masks), (rgb_val_imgs, rgb_val_masks), _ = all_data["RGB"]

    # SAR数据增强和数据集
    sar_train_transform = get_transforms(config.IMG_SIZE, is_train=True, is_sar=True)
    sar_val_transform = get_transforms(config.IMG_SIZE, is_train=False, is_sar=True)

    sar_train_dataset = InterWaveDataset(sar_train_imgs, sar_train_masks, sar_train_transform, is_train=True)
    sar_val_dataset = InterWaveDataset(sar_val_imgs, sar_val_masks, sar_val_transform, is_train=False)

    # SAR只需要过滤验证集的空patches
    sar_val_indices = get_non_empty_indices(sar_val_dataset, modality_name="SAR验证集")
    sar_val_dataset = Subset(sar_val_dataset, sar_val_indices)

    sar_train_loader = torch.utils.data.DataLoader(
        sar_train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=6, pin_memory=True  # ✅ Windows稳定性：num_workers=6避免多进程死锁
    )
    sar_val_loader = torch.utils.data.DataLoader(
        sar_val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=6, pin_memory=True  # ✅ Windows稳定性：num_workers=6避免多进程死锁
    )

    # RGB数据增强和数据集
    rgb_train_transform = get_transforms(config.IMG_SIZE, is_train=True, is_rgb=True)
    rgb_val_transform = get_transforms(config.IMG_SIZE, is_train=False, is_rgb=True)

    # ✅ RGB训练集使用动态过滤Dataset
    rgb_min_ratio = getattr(config, 'RGB_MIN_POSITIVE_RATIO', 0.001)
    print(f"\n🎯 RGB训练集使用动态过滤，最小正样本比例: {rgb_min_ratio*100:.2f}%")
    rgb_train_dataset = FilteredInterWaveDataset(
        rgb_train_imgs, rgb_train_masks, rgb_train_transform,
        is_train=True,
        min_positive_ratio=rgb_min_ratio  # 动态过滤阈值
    )

    # RGB验证集仍使用普通Dataset
    rgb_val_dataset = InterWaveDataset(rgb_val_imgs, rgb_val_masks, rgb_val_transform, is_train=False)

    # RGB验证集只过滤空patches
    rgb_val_indices = get_non_empty_indices(rgb_val_dataset, modality_name="RGB验证集")
    rgb_val_dataset = Subset(rgb_val_dataset, rgb_val_indices)

    rgb_train_loader = torch.utils.data.DataLoader(
        rgb_train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=6, pin_memory=True  # ✅ Windows稳定性：num_workers=6避免多进程死锁
    )
    rgb_val_loader = torch.utils.data.DataLoader(
        rgb_val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=6, pin_memory=True  # ✅ Windows稳定性：num_workers=6避免多进程死锁
    )

    # =================== 🆕 方案A：交替训练 DataLoader ===================
    # ✅ 使用 AlternatingModalityLoaderBalanced 实现交替训练
    # 原理：每个batch只包含1张图（SAR或RGB交替）
    # 显存占用：原来的一半
    # 配合梯度累积：ACCUMULATION_STEPS=6 等效 batch_size=6
    train_loader = AlternatingModalityLoaderBalanced(
        sar_train_loader, rgb_train_loader,  # ✅ 只使用SAR和RGB
        is_train=True,
        sar_rgb_ratio=(0, 1)  # ✅ 修复：SAR:RGB = 1:2 (之前是1:0，导致只训练SAR)
    )
    val_loader = AlternatingModalityLoaderBalanced(
        sar_val_loader, rgb_val_loader,  # ✅ 只使用SAR和RGB
        is_train=False,
        sar_rgb_ratio=(0, 1)  # ✅ 修复：保持验证集比例一致
    )

    # =================== 3️⃣ 创建模型 ===================
    if config.MODEL_TYPE == "SegFormer" :
        from models.SegFormer import segformer_b1
        model = segformer_b1(img_size=1024, num_classes=1, stem_channels=32)
    elif config.MODEL_TYPE == "TransUNet" :
        from models.TransUNet import transunet_paper_standard
        model = transunet_paper_standard(img_size=1024, num_classes=1, stem_channels=32)
    elif config.MODEL_TYPE == "IWResNet" :
        from models.IWResNet import iwresnet_base
        model = iwresnet_base(img_size=1024, num_classes=1, stem_channels=32)
    elif config.MODEL_TYPE == "SwinUNet" :
        from models.SwinUNet import swin_unet_base
        model = swin_unet_base(img_size=1024, num_classes=1, stem_channels=32)
    elif config.MODEL_TYPE == "IWENet" :
        from models.IWENet import iwenet_base
        model = iwenet_base(img_size=1024, num_classes=1, stem_channels=32)
    elif config.MODEL_TYPE == "mtu2net" :
        from models.MTU2Net import mtu2net_base
        model = mtu2net_base(num_classes=1, stem_channels=32)
    else:
        from models.unet import create_unet
        model = create_unet(
            config.MODEL_TYPE,
            num_classes=1,
            dropout_rates=[0.0, 0.0, 0.0, 0.0]  # 禁用Dropout
        )

    model = model.to(config.DEVICE)
    def init_weights(m):
        if isinstance(m, torch.nn.Conv2d):
            torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                # 输出层特殊处理
                if m.out_channels == 1:
                    torch.nn.init.constant_(m.bias, -1.0)  # 初始偏向负（大部分是背景）
                else:
                    torch.nn.init.constant_(m.bias, 0.01)  # 其他层小正偏置
        elif isinstance(m, torch.nn.GroupNorm):
            torch.nn.init.constant_(m.weight, 1)
            torch.nn.init.constant_(m.bias, 0)

    model.apply(init_weights)
    print("✅ 模型初始化: Kaiming Normal + 调整偏置")

    # 打印参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # =================== 4️⃣ 定义损失函数 ===================
    # ✅ 使用OHEM解决RGB极端稀疏问题
    USE_OHEM = True  # 使用在线困难样本挖掘

    if USE_OHEM:
        from models.ohem_loss import RGBAdaptiveLoss
        criterion = RGBAdaptiveLoss()
        print("✅ Using OHEM (Online Hard Example Mining) for RGB extreme sparsity")
        print("   - SAR: Balanced Focal + Dice Loss")
        print("   - RGB: OHEM + Dice Loss (only compute loss on hard pixels)")
    else:
        criterion = create_internal_wave_loss(
            # 类别权重（pos_weight） - 大幅提高来强制模型关注稀疏内波
            sar_pos_weight=50.0,      # ~2%正样本
            rgb_pos_weight=500.0,    # 0.06%正样本！需要极高的权重（调高到500）
            # Base Loss参数 - 调整权重偏向BCE
            weight_hd=0.2,            # HD权重降低（全黑时惩罚不够）
            weight_bce=0.8,           # BCE权重提高（让BCE主导）
            hd_alpha=1.5,
            hd_use_gpu=True,
        )


    # =================== 5️⃣ 优化器 & 学习率调度 ===================
    # ✅ 使用Lion优化器 (Google 2023) - 适合中小型模型 + 复杂损失
    # 优势：显存效率高(比AdamW省40%)、收敛稳定、泛化能力强

    # 提高学习率来强制模型学习稀疏特征
    optimizer = Lion(
        model.parameters(),
        lr=config.LEARNING_RATE * 2,  # 提高学习率（2e-4 → 4e-4）
        weight_decay=config.WEIGHT_DECAY ,  # ✅ 增强正则化：10倍weight_decay防止过拟合
        betas=(0.9, 0.99)  # Lion默认值（比Adam的(0.9, 0.999)略小）
    )
    print(f"✅ 优化器配置:")
    print(f"   Weight Decay: {config.WEIGHT_DECAY * 10:.4f}")
    print(f"   学习率: {config.LEARNING_RATE * 2:.6f} (提高2倍)")

    # 学习率调度：OneCycleLR效果更好（搭配Lion）
    from torch.optim.lr_scheduler import OneCycleLR

    scheduler = OneCycleLR(
        optimizer,
        max_lr=config.LEARNING_RATE * 2,  # 与初始lr一致（提高后的值）
        epochs=config.NUM_EPOCHS,
        steps_per_epoch=len(train_loader),  # ⚠️ 需要知道每epoch步数
        pct_start=0.2,  # ✅ 前20% epochs warmup (从0.1改为0.2)
        anneal_strategy='cos',  # 余弦退火
        div_factor=25.0,  # 初始lr = max_lr / 25
        final_div_factor=1000  # ✅ 最终lr = max_lr / 1000 (从1e4改为1000)
    )

    # 降低学习率避免发散
    config.LEARNING_RATE = config.LEARNING_RATE * 0.5
    print(f"✅ 学习率降低50%: {config.LEARNING_RATE:.6f}")

    # =================== 6️⃣ 训练历史 ===================
    train_losses, val_losses = [], []
    train_loss1, val_loss1 = [], []  # Focal
    train_loss2, val_loss2 = [], []  # OHEM
    train_loss3, val_loss3 = [], []  # Dice
    train_metrics_list, val_metrics_list = [], []
    best_val_bf = -1.0

    # ✅ 添加Early Stopping（监控验证loss）
    from train.early_stopping import EarlyStopping
    early_stopping = EarlyStopping(
        patience=500,        # 容忍10个epoch没有改善
        min_delta=0.001,    # 最小改善量
        mode='min',         # 监控loss（越小越好）
        verbose=True        # 打印详细信息
    )
    print(f"✅ Early Stopping enabled: patience={early_stopping.patience}, min_delta={early_stopping.min_delta}")

    # =================== 7️⃣ 训练循环 ===================
    for epoch in range(config.NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{config.NUM_EPOCHS}")

        # ----------- 训练阶段 -----------
        train_loss, train_focal, train_ohem, train_dice, train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, config.DEVICE, config.MODEL_TYPE,
            feature_save_dir=feature_save_dir, epoch=epoch,
            accumulation_steps=config.ACCUMULATION_STEPS,  # ✅ 梯度累积
            scheduler=scheduler  # ✅ 传入scheduler（OneCycleLR需要batch-wise更新）
        )

        # ----------- 验证阶段 -----------
        val_loss, val_focal, val_ohem, val_dice, val_metrics = validate_epoch(
            model, val_loader, criterion, config.DEVICE, config.MODEL_TYPE
        )

        # ----------- 记录指标 -----------
        train_losses.append(train_loss); val_losses.append(val_loss)
        train_loss1.append(train_focal); val_loss1.append(val_focal)
        train_loss2.append(train_ohem); val_loss2.append(val_ohem)
        train_loss3.append(train_dice); val_loss3.append(val_dice)  # 新增Dice记录
        train_metrics_list.append(train_metrics); val_metrics_list.append(val_metrics)

        # ----------- 打印 -----------
        print("---------- Training Metrics ----------")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Focal: {train_focal:.4f} | OHEM: {train_ohem:.4f} | Dice: {train_dice:.4f}")
        for k, v in train_metrics.items(): print(f"{k}: {v:.4f}")

        print("---------- Validation Metrics ----------")
        print(f"Valid Loss: {val_loss:.4f}")
        print(f"Focal: {val_focal:.4f} | OHEM: {val_ohem:.4f} | Dice: {val_dice:.4f}")
        for k, v in val_metrics.items(): print(f"{k}: {v:.4f}")

        # ----------- 保存最优模型 -----------
        if val_metrics['BF_Score'] > best_val_bf:
            best_val_bf = val_metrics['BF_Score']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_bf': best_val_bf,
                'config': config,
                # ✅ 保存训练历史，便于下次断点重训练
                'train_losses': train_losses,
                'val_losses': val_losses,
                'train_loss1': train_loss1,  # Focal
                'val_loss1': val_loss1,
                'train_loss2': train_loss2,  # OHEM
                'val_loss2': val_loss2,
                'train_loss3': train_loss3,  # Dice
                'val_loss3': val_loss3,
                'train_metrics_list': train_metrics_list,
                'val_metrics_list': val_metrics_list,
            }, os.path.join(config.PTH_SAVE_DIR, f'best_{config.MODEL_TYPE}_model.pth'))
            print(f"New best model saved! Val BF Score: {best_val_bf:.4f}")

        # ✅ Early Stopping检查
        if early_stopping(val_loss, epoch=epoch+1):
            print(f"\n🛑 Early Stopping triggered at epoch {epoch+1}")
            print(f"   Best val_loss: {early_stopping.best_score:.4f} (Epoch {early_stopping.best_epoch})")
            print(f"   Stopping training to prevent overfitting...")
            break

        # ⚠️ OneCycleLR在train_epoch内部已经step，这里不需要再step
        # scheduler.step()  # 删除这行
    
    # =================== 绘制曲线 ===================
    plot_training_curves(
        config, config.NUM_EPOCHS,
        train_losses, val_losses,
        train_loss1, val_loss1,
        train_loss2, val_loss2,
        train_loss3, val_loss3,
        train_metrics_list, val_metrics_list
    )

    return model
