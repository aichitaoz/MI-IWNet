import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.train_config import Config
from models.unet import create_unet

import warnings
warnings.filterwarnings('ignore')


def load_and_cut_patches(image_path, crop_size=1024, stride=512):
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Cannot load image: {image_path}")

    h, w = image.shape[:2]
    if h != 1024 or w != 1024:
        image = cv2.resize(image, (4096, 2048), interpolation=cv2.INTER_LINEAR)

    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    h, w = image.shape[:2]
    patches, positions = [], []
    for y in range(0, max(1, h - crop_size + 1), stride):
        for x in range(0, max(1, w - crop_size + 1), stride):
            y2, x2 = min(y + crop_size, h), min(x + crop_size, w)
            patches.append(image[y2 - crop_size:y2, x2 - crop_size:x2])
            positions.append((y2 - crop_size, x2 - crop_size, crop_size, crop_size))

    return patches, (h, w), positions, image


def predict(model, patch_img, device, threshold=0.5):
    tensor = torch.from_numpy(patch_img.transpose(2, 0, 1).astype(np.float32) / 255.0)
    tensor = ((tensor - 0.5) / 0.5).unsqueeze(0).to(device)
    with torch.no_grad():
        prob = torch.sigmoid(model(tensor, datasets=['rgb'])).cpu().numpy()[0, 0]
    return prob, (prob > threshold).astype(np.uint8)


def merge_patches(patch_preds, positions, original_size):
    h, w = original_size
    merged_prob   = np.zeros((h, w), dtype=np.float32)
    merged_binary = np.zeros((h, w), dtype=np.float32)
    count         = np.zeros((h, w), dtype=np.int32)
    for (prob, binary), (y, x, ph, pw) in zip(patch_preds, positions):
        ey, ex = min(y + ph, h), min(x + pw, w)
        merged_prob[y:ey, x:ex]   += prob[:ey-y, :ex-x]
        merged_binary[y:ey, x:ex] += binary[:ey-y, :ex-x]
        count[y:ey, x:ex]         += 1
    mask = count > 0
    merged_prob[mask]   /= count[mask]
    merged_binary = (merged_binary[mask] / count[mask] > 0.5).astype(np.uint8) if mask.any() else merged_binary.astype(np.uint8)
    # 重新生成完整 binary
    merged_binary_full = (merged_prob > 0.5).astype(np.uint8)
    return merged_binary_full


def to_rgb(image):
    img = image.astype(np.float32) / 255.0
    return img if img.ndim == 3 else np.stack([img] * 3, axis=-1)


def save_pred_overlay(image, pred_binary, save_path, alpha=0.5):
    overlay = to_rgb(image).copy()
    overlay[pred_binary > 0] = (1 - alpha) * overlay[pred_binary > 0] + alpha * np.array([1.0, 1.0, 0.15])
    plt.imsave(save_path, np.clip(overlay, 0, 1))


def save_gt_pred_overlay(image, pred_binary, mask_binary, save_path, alpha=0.6):
    overlay = to_rgb(image).copy()
    pred_only = (pred_binary > 0) & (mask_binary == 0)
    gt_only   = (mask_binary > 0) & (pred_binary == 0)
    overlap   = (pred_binary > 0) & (mask_binary > 0)
    overlay[pred_only] = (1 - alpha) * overlay[pred_only] + alpha * np.array([1.0, 0.2, 0.2])
    overlay[gt_only]   = (1 - alpha) * overlay[gt_only]   + alpha * np.array([0.2, 1.0, 0.2])
    overlay[overlap]   = (1 - alpha) * overlay[overlap]   + alpha * np.array([1.0, 1.0, 0.2])
    plt.imsave(save_path, np.clip(overlay, 0, 1))


def main():
    config = Config()
    model_path = 'checkpoints/revise/pth/best_ConvNeXt_model.pth'

    image_dir = '/home/xiaobowen/project/internal_wave_detection_project/IW_data/images_rgb'
    mask_dir  = '/home/xiaobowen/project/internal_wave_detection_project/IW_data/masks_rgb'
    save_dir  = '/home/xiaobowen/project/internal_wave_detection_project/IW_data/results/ConvNeXt/RGB'

    os.makedirs(os.path.join(save_dir, 'pred_overlay'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'gt_pred_overlay'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'pred_mask'), exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(model_path, map_location=device)
    model = create_unet('ConvNeXt', num_classes=1, dropout_rates=[0.0] * 4)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device).eval()

    for img_file in sorted(f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff'))):
        mask_path = os.path.join(mask_dir, os.path.splitext(img_file)[0] + '.png')
        if not os.path.exists(mask_path):
            print(f"⚠️ Skipped (no mask): {img_file}")
            continue

        try:
            patches, original_size, positions, image = load_and_cut_patches(os.path.join(image_dir, img_file))
            patch_preds  = [predict(model, p, device) for p in patches]
            pred_binary  = merge_patches(patch_preds, positions, original_size)

            mask_binary  = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask_binary  = cv2.resize(mask_binary, (original_size[1], original_size[0]), interpolation=cv2.INTER_NEAREST)
            mask_binary  = (mask_binary > 127).astype(np.uint8)

            base = os.path.splitext(img_file)[0]
            save_pred_overlay(image, pred_binary, os.path.join(save_dir, 'pred_overlay', f'{base}.png'))
            save_gt_pred_overlay(image, pred_binary, mask_binary, os.path.join(save_dir, 'gt_pred_overlay', f'{base}.png'))
            cv2.imwrite(os.path.join(save_dir, 'pred_mask', f'{base}.png'), pred_binary * 255)
            print(f"✅ {img_file}")
        except Exception as e:
            print(f"❌ {img_file}: {e}")


if __name__ == '__main__':
    main()