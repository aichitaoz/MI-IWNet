from tqdm import tqdm
from models.metrics import calculate_metrics
import torch


def train_epoch(model, dataloader, criterion, optimizer, device, model_type,
                epoch=None, feature_save_dir=None,
                accumulation_steps=1, scheduler=None):
    model.train()
    total_loss, focal_loss_total, ohem_loss_total, dice_loss_total = 0.0, 0.0, 0.0, 0.0
    metrics = {'IoU':0.0, 'Dice':0.0, 'Precision':0.0, 'Recall':0.0, 'Accuracy':0.0, 'BF_Score':0.0}

    optimizer.zero_grad()
    pbar = tqdm(dataloader, desc='Training', leave=False)

    for batch_idx, batch_data in enumerate(pbar):
        # ✅ 兼容新旧数据格式
        if len(batch_data) == 3:  # 新格式：单模态交替batch
            images, masks, datasets = batch_data
            # ✅ 提取单个tensor（AlternatingModalityLoaderBalanced返回[tensor]格式）
            if isinstance(images, list):
                images = images[0].to(device)  # Extract single tensor from list
            else:
                images = images.to(device)
            masks = masks.to(device)
        else:  # 旧格式：单模态
            images, masks = batch_data
            images, masks = images.to(device), masks.to(device)
            # 自动推断模态
            if images.shape[1] == 1:
                datasets = ["sar"]
            elif images.shape[1] == 3:
                datasets = ["rgb"]
            else:
                raise ValueError(f"无法推断通道数为{images.shape[1]}的模态，仅支持1(SAR)或3(RGB)通道")

        try:
            outputs = model(images, datasets=datasets)
            loss, focal_loss, ohem_loss, dice_loss = criterion(outputs, masks, datasets)

            loss = loss / accumulation_steps

            # ✅ NaN/Inf 检测
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"\n❌ NaN/Inf detected at batch {batch_idx}! Skipping...")
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                continue

            # 反向传播
            loss.backward()

            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(dataloader):
                # ✅ 梯度更新
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if scheduler is not None:
                    scheduler.step()

            # ✅ 累计统计量（Focal、OHEM和Dice损失）
            total_loss += loss.item() * accumulation_steps
            focal_loss_total += focal_loss.item()
            ohem_loss_total += ohem_loss.item()
            dice_loss_total += dice_loss.item()

            batch_metrics = calculate_metrics(outputs, masks)
            for key in metrics:
                metrics[key] += batch_metrics[key]

            pbar.set_postfix({
                'Loss': f'{loss.item() * accumulation_steps:.4f}',
                'IoU': f'{batch_metrics["IoU"]:.4f}',
                'Dice': f'{batch_metrics["Dice"]:.4f}'
            })

            # 删除了频繁的empty_cache，让PyTorch自动管理显存

        except RuntimeError as e:
            # ✅ 捕获 OOM 并安全跳过
            if "out of memory" in str(e).lower():
                print(f"\n💥 OOM detected at batch {batch_idx}, skipping...")
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                continue
            else:
                raise e

    n = len(dataloader)
    for key in metrics:
        metrics[key] /= max(1, n)

    # ✅ 返回3个值（HD和BCE）
    return (
        total_loss / max(1, n),
        focal_loss_total / max(1, n),
        ohem_loss_total / max(1, n),
        dice_loss_total / max(1, n),
        metrics
    )


def validate_epoch(model, dataloader, criterion, device, model_type):
    model.eval()
    total_loss, focal_loss_total, ohem_loss_total, dice_loss_total = 0.0, 0.0, 0.0, 0.0
    metrics = {'IoU':0.0, 'Dice':0.0, 'Precision':0.0, 'Recall':0.0, 'Accuracy':0.0, 'BF_Score':0.0}

    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Validation', leave=False)
        batch_idx = 0  # ✅ 添加batch计数器
        for batch_data in pbar:
            try:
                # ✅ 兼容新旧数据格式
                if len(batch_data) == 3:  # 新格式：单模态交替batch
                    images, masks, datasets = batch_data
                    # ✅ 提取单个tensor（AlternatingModalityLoaderBalanced返回[tensor]格式）
                    if isinstance(images, list):
                        images = images[0].to(device)  # Extract single tensor from list
                    else:
                        images = images.to(device)
                    masks = masks.to(device)
                else:  # 旧格式：单模态
                    images, masks = batch_data
                    images, masks = images.to(device), masks.to(device)
                    if images.shape[1] == 1:
                        datasets = ["sar"]
                    elif images.shape[1] == 3:
                        datasets = ["rgb"]
                    else:
                        raise ValueError(f"无法推断通道数为{images.shape[1]}的模态，仅支持1(SAR)或3(RGB)通道")

                outputs = model(images, datasets=datasets)
                loss, focal_loss, ohem_loss, dice_loss = criterion(outputs, masks, datasets)

                # ✅ NaN/Inf 检测
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"\n⚠️  Validation: NaN/Inf at batch {batch_idx}, skipping...")
                    batch_idx += 1
                    continue

                total_loss += loss.item()
                focal_loss_total += focal_loss.item()
                ohem_loss_total += ohem_loss.item()
                dice_loss_total += dice_loss.item()

                batch_metrics = calculate_metrics(outputs, masks)
                for key in metrics:
                    metrics[key] += batch_metrics[key]
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'IoU': f'{batch_metrics["IoU"]:.4f}',
                    'Dice': f'{batch_metrics["Dice"]:.4f}'
                })

                batch_idx += 1

            except Exception as e:
                print(f"\n❌ Validation error at batch {batch_idx}: {e}")
                import traceback
                traceback.print_exc()
                batch_idx += 1
                continue

    n = len(dataloader)
    for key in metrics:
        metrics[key] /= max(1, n)

    # ✅ 返回3个值（HD和BCE）
    return (
        total_loss / max(1, n),
        focal_loss_total / max(1, n),
        ohem_loss_total / max(1, n),
        dice_loss_total / max(1, n),
        metrics
    )
