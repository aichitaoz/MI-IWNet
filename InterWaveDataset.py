import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
import os

try:
    import tifffile
    HAS_TIFFFILE = True
except ImportError:
    HAS_TIFFFILE = False
    print("Warning: tifffile not installed. Multi-channel TIFF support disabled.")

class InterWaveDataset(Dataset):
    """海洋内波数据集（支持在线切块）"""
    def __init__(self, image_paths, mask_paths, transform=None, is_train=True, crop_size=1024, stride=None):
        assert len(image_paths) == len(mask_paths), "image_paths 与 mask_paths 长度必须相同"
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.is_train = is_train
        self.crop_size = crop_size
        self.stride = stride if stride is not None else crop_size

        self.patch_index = []
        if not self.is_train:
            for i, (img_path, mask_path) in enumerate(zip(self.image_paths, self.mask_paths)):
                img, mask = self.load_image_mask(img_path, mask_path)
                if img is None:
                    raise ValueError(f"Cannot load image: {img_path}")
                h, w = img.shape[:2]
                if h <= crop_size and w <= crop_size:
                    # 验证时跳过空mask样本
                    if np.sum(mask) > 0:
                        self.patch_index.append((i, 0, 0))
                else:
                    for y in range(0, max(1, h - crop_size + 1), self.stride):
                        for x in range(0, max(1, w - crop_size + 1), self.stride):
                            patch_mask = mask[y:min(y+crop_size, h), x:min(x+crop_size, w)]
                            # 验证时跳过空mask的patch
                            if np.sum(patch_mask) > 0:
                                self.patch_index.append((i, y, x))
            print(f"Validation mode: {len(self.patch_index)} patches (empty masks filtered)")

    def __len__(self):
        return len(self.image_paths) if self.is_train else len(self.patch_index)

    def __getitem__(self, idx):
        if self.is_train:
            max_retry = 50
            for retry_count in range(max_retry):
                current_idx = idx if retry_count == 0 else np.random.randint(0, len(self.image_paths))
                img_path = self.image_paths[current_idx]
                mask_path = self.mask_paths[current_idx]
                img_np, mask_np = self.load_image_mask(img_path, mask_path)

                h, w = img_np.shape[:2]
                if h > self.crop_size and w > self.crop_size:
                    max_try = 20
                    for _ in range(max_try):
                        y = np.random.randint(0, h - self.crop_size)
                        x = np.random.randint(0, w - self.crop_size)
                        patch_mask = mask_np[y:y+self.crop_size, x:x+self.crop_size]
                        if np.mean(patch_mask) > 0.001:
                            break
                    img_np = img_np[y:y+self.crop_size, x:x+self.crop_size]
                    mask_np = mask_np[y:y+self.crop_size, x:x+self.crop_size]
                else:
                    pad_h = max(0, self.crop_size - h)
                    pad_w = max(0, self.crop_size - w)
                    img_np = cv2.copyMakeBorder(img_np, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
                    mask_np = cv2.copyMakeBorder(mask_np, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)

                if self.transform is not None:
                    augmented = self.transform(image=img_np, mask=mask_np)
                    img_np = augmented["image"]
                    mask_np = augmented["mask"]

                mask_sum = mask_np.sum().item() if isinstance(mask_np, torch.Tensor) else np.sum(mask_np)
                if mask_sum > 0:
                    break

                if retry_count == max_retry - 1:
                    print(f"Warning: After {max_retry} retries, still got empty mask for idx {idx}")
        else:
            img_idx, y, x = self.patch_index[idx]
            img_path = self.image_paths[img_idx]
            mask_path = self.mask_paths[img_idx]
            img_np, mask_np = self.load_image_mask(img_path, mask_path)

            h, w = img_np.shape[:2]
            y2, x2 = min(y + self.crop_size, h), min(x + self.crop_size, w)
            img_np = img_np[y2 - self.crop_size:y2, x2 - self.crop_size:x2]
            mask_np = mask_np[y2 - self.crop_size:y2, x2 - self.crop_size:x2]

            if self.transform is not None:
                augmented = self.transform(image=img_np, mask=mask_np)
                img_np = augmented["image"]
                mask_np = augmented["mask"]

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

    def load_image_mask(self, img_path, mask_path):
        """
        加载图像和mask
        ✅ 统一归一化到 [0, 255] uint8
        ✅ 统一返回类型避免Albumentations报错
        """
        # ===== 1. 加载mask =====
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Cannot load mask: {mask_path}")
        mask = (mask > 127).astype(np.uint8)

        # ===== 2. 加载图像 =====
        ext = os.path.splitext(img_path)[1].lower()

        if ext in ['.tif', '.tiff']:
            if not HAS_TIFFFILE:
                raise ImportError(f"tifffile is required to read {img_path}. Install: pip install tifffile")

            import signal
            def timeout_handler(signum, frame):
                raise TimeoutError(f"Timeout loading TIFF: {img_path}")

            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(30)

            try:
                img = tifffile.imread(img_path)
                signal.alarm(0)
            except TimeoutError as e:
                signal.alarm(0)
                print(f"{e}")
                raise ValueError(f"Cannot load TIFF (timeout): {img_path}")

            if img is None:
                raise ValueError(f"Cannot load TIFF image: {img_path}")
        else:
            # PNG, JPG等格式
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            if img is None:
                raise ValueError(f"Cannot load image: {img_path}")

        # ===== 3. 统一归一化到 [0, 255] uint8 =====
        if img.dtype == np.uint16:
            # uint16 [0, 65535] → [0, 255]
            img = (img / 65535.0 * 255).astype(np.uint8)
        elif img.dtype in [np.float32, np.float64]:
            # float类型归一化
            img_min, img_max = img.min(), img.max()
            if img_max > img_min:
                if img_max > 1.0:
                    # 大于1的float，归一化到[0,1]再缩放
                    img = ((img - img_min) / (img_max - img_min) * 255).astype(np.uint8)
                else:
                    # [0,1]范围的float，直接缩放
                    img = (img * 255).astype(np.uint8)
            else:
                img = np.zeros_like(img, dtype=np.uint8)
        elif img.dtype != np.uint8:
            # 其他类型统一归一化
            img_min, img_max = img.min(), img.max()
            if img_max > img_min:
                img = ((img - img_min) / (img_max - img_min) * 255).astype(np.uint8)
            else:
                img = np.zeros_like(img, dtype=np.uint8)
        # 现在 img 一定是 uint8 类型

        # ===== 4. 处理通道 =====
        if img.ndim == 2:
            # 单通道灰度图
            pass
        elif img.ndim == 3:
            if img.shape[2] == 1:
                # [H, W, 1] → [H, W]
                img = img.squeeze(-1)
            elif img.shape[2] == 3:
                # 检查是否是伪三通道（SAR图像保存为RGB格式）
                if np.allclose(img[..., 0], img[..., 1]) and np.allclose(img[..., 1], img[..., 2]):
                    # 伪三通道，取第一个通道
                    img = img[..., 0]
                else:
                    # 真RGB图像，转换颜色空间
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    # 保持3通道
            elif img.shape[2] == 7:
                # 7通道图像（多光谱），保持原样
                pass
            else:
                raise ValueError(f"Unsupported TIFF channels: {img.shape[2]} for {img_path}")
        else:
            raise ValueError(f"Unsupported image dimensions: {img.shape} for {img_path}")

        # ===== 5. 返回（确保都是uint8）=====
        return img, mask