"""
针对SAR和RGB设计不同的方向滤波器
集成到数据加载pipeline中
"""

import cv2
import numpy as np
import torch

class ModalityAdaptiveDirectionalFilter:
    """
    模态自适应方向滤波器

    根据SAR和RGB的不同特征，应用不同的滤波参数
    """

    def __init__(self):
        # SAR参数：宽条纹
        self.sar_params = {
            'angle': 6,  # 主方向
            'bandwidth': 45,  # 方向带宽
            'freq_low': 1024 / 168,  # 对应stripe_spacing上限
            'freq_high': 1024 / 79,  # 对应stripe_width下限
        }

        # RGB参数：细条纹
        self.rgb_params = {
            'angle': 6,
            'bandwidth': 45,
            'freq_low': 1024 / 203,
            'freq_high': 1024 / 7,
        }

    def directional_filter(self, image, angle=0, bandwidth=45):
        """
        方向性滤波：只保留特定方向的条纹

        Args:
            image: 灰度图像 [H, W]
            angle: 条纹方向（度）
            bandwidth: 方向带宽（度）
        """
        if len(image.shape) == 3:
            # 如果是多通道，分别处理每个通道
            channels = []
            for c in range(image.shape[2]):
                filtered_c = self._filter_single_channel(image[:,:,c], angle, bandwidth)
                channels.append(filtered_c)
            return np.stack(channels, axis=2)
        else:
            return self._filter_single_channel(image, angle, bandwidth)

    def _filter_single_channel(self, image, angle, bandwidth):
        """单通道方向滤波"""
        # FFT
        f = np.fft.fft2(image)
        fshift = np.fft.fftshift(f)

        rows, cols = image.shape
        crow, ccol = rows // 2, cols // 2

        # 创建角度坐标
        u = np.arange(rows) - crow
        v = np.arange(cols) - ccol
        U, V = np.meshgrid(v, u)

        # 计算每个频率点的角度
        theta = np.arctan2(V, U) * 180 / np.pi

        # 方向性滤波器
        mask = np.zeros((rows, cols), dtype=np.float32)

        # 保留 angle ± bandwidth 的频率（考虑对称性）
        condition = (np.abs(theta - angle) <= bandwidth) | \
                   (np.abs(theta - angle - 180) <= bandwidth) | \
                   (np.abs(theta - angle + 180) <= bandwidth)
        mask[condition] = 1.0

        # 应用滤波器
        fshift_filtered = fshift * mask

        # 逆FFT
        f_ishift = np.fft.ifftshift(fshift_filtered)
        img_back = np.fft.ifft2(f_ishift)
        img_back = np.abs(img_back)

        # 归一化到[0, 255]
        img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        return img_back

    def __call__(self, image, modality='sar'):
        """
        应用滤波器

        Args:
            image: numpy array [H, W] or [H, W, C]
            modality: 'sar' or 'rgb'

        Returns:
            filtered image
        """
        if modality.lower() == 'sar':
            params = self.sar_params
        elif modality.lower() == 'rgb':
            params = self.rgb_params
        else:
            raise ValueError(f"Unknown modality: {modality}")

        # 应用方向滤波
        filtered = self.directional_filter(
            image,
            angle=params['angle'],
            bandwidth=params['bandwidth']
        )

        return filtered


# ================ 测试代码 ================

def test_on_samples():
    """在SAR和RGB样本上测试滤波效果"""

    print("="*80)
    print("测试模态自适应方向滤波器")
    print("="*80)

    import json
    import os
    from pathlib import Path
    import matplotlib.pyplot as plt

    # 读取困难样本
    difficult_json = 'checkpoints/data_analysis/difficult_samples.json'

    with open(difficult_json, 'r') as f:
        difficult_samples = json.load(f)

    # 创建滤波器
    filter = ModalityAdaptiveDirectionalFilter()

    # 输出目录
    output_dir = 'checkpoints/data_analysis/modality_adaptive_filter'
    os.makedirs(output_dir, exist_ok=True)

    # 找一个SAR样本和一个RGB样本
    sar_sample = None
    rgb_sample = None

    for sample in difficult_samples:
        if 'images_sar' in sample['path'] and sar_sample is None:
            sar_sample = sample
        elif 'images_rgb' in sample['path'] and rgb_sample is None:
            rgb_sample = sample

        if sar_sample and rgb_sample:
            break

    def calculate_snr(image, mask):
        """计算信噪比"""
        mask_binary = (mask > 127).astype(np.uint8)

        fg_pixels = image[mask_binary > 0]
        bg_pixels = image[mask_binary == 0]

        if len(fg_pixels) == 0 or len(bg_pixels) == 0:
            return 0.0

        fg_mean = np.mean(fg_pixels)
        bg_mean = np.mean(bg_pixels)
        bg_std = np.std(bg_pixels)

        snr = abs(fg_mean - bg_mean) / (bg_std + 1e-6)
        return snr

    # 测试SAR样本
    if sar_sample:
        print("\n测试SAR样本...")
        img_path = sar_sample['path']
        mask_path = img_path.replace('images_sar', 'masks_sar')
        mask_path = str(Path(mask_path).with_suffix('.png'))

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if img is not None and mask is not None:
            # 应用滤波
            filtered_sar = filter(img, modality='sar')
            filtered_wrong = filter(img, modality='rgb')  # 错误的参数

            # 计算SNR
            snr_original = calculate_snr(img, mask)
            snr_filtered_correct = calculate_snr(filtered_sar, mask)
            snr_filtered_wrong = calculate_snr(filtered_wrong, mask)

            print(f"  原始SNR:              {snr_original:.3f}")
            print(f"  SAR参数滤波SNR:       {snr_filtered_correct:.3f} (↑{snr_filtered_correct/snr_original:.2f}x)")
            print(f"  RGB参数滤波SNR(错误): {snr_filtered_wrong:.3f} (↑{snr_filtered_wrong/snr_original:.2f}x)")

            # 可视化
            fig, axes = plt.subplots(1, 4, figsize=(20, 5))

            axes[0].imshow(img, cmap='gray')
            axes[0].set_title(f'Original SAR\nSNR: {snr_original:.3f}', fontsize=12)
            axes[0].axis('off')

            axes[1].imshow(filtered_sar, cmap='gray')
            axes[1].set_title(f'SAR Params (Correct)\nSNR: {snr_filtered_correct:.3f} (↑{snr_filtered_correct/snr_original:.2f}x)',
                             fontsize=12, color='green')
            axes[1].axis('off')

            axes[2].imshow(filtered_wrong, cmap='gray')
            axes[2].set_title(f'RGB Params (Wrong)\nSNR: {snr_filtered_wrong:.3f} (↑{snr_filtered_wrong/snr_original:.2f}x)',
                             fontsize=12, color='red')
            axes[2].axis('off')

            axes[3].imshow(mask, cmap='gray')
            axes[3].set_title('Ground Truth', fontsize=12)
            axes[3].axis('off')

            plt.suptitle(f'SAR Sample: {Path(img_path).name}', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(f'{output_dir}/sar_comparison.png', dpi=150, bbox_inches='tight')
            plt.close()

            print(f"  ✅ 保存: {output_dir}/sar_comparison.png")

    # 测试RGB样本
    if rgb_sample:
        print("\n测试RGB样本...")
        img_path = rgb_sample['path']
        mask_path = img_path.replace('images_rgb', 'masks_rgb')
        mask_path = str(Path(mask_path).with_suffix('.png'))

        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if img is not None and mask is not None:
            # 转灰度
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # 应用滤波
            filtered_rgb = filter(img_gray, modality='rgb')
            filtered_wrong = filter(img_gray, modality='sar')  # 错误的参数

            # 计算SNR
            snr_original = calculate_snr(img_gray, mask)
            snr_filtered_correct = calculate_snr(filtered_rgb, mask)
            snr_filtered_wrong = calculate_snr(filtered_wrong, mask)

            print(f"  原始SNR:              {snr_original:.3f}")
            print(f"  RGB参数滤波SNR:       {snr_filtered_correct:.3f} (↑{snr_filtered_correct/snr_original:.2f}x)")
            print(f"  SAR参数滤波SNR(错误): {snr_filtered_wrong:.3f} (↑{snr_filtered_wrong/snr_original:.2f}x)")

            # 可视化
            fig, axes = plt.subplots(1, 4, figsize=(20, 5))

            axes[0].imshow(img_gray, cmap='gray')
            axes[0].set_title(f'Original RGB\nSNR: {snr_original:.3f}', fontsize=12)
            axes[0].axis('off')

            axes[1].imshow(filtered_rgb, cmap='gray')
            axes[1].set_title(f'RGB Params (Correct)\nSNR: {snr_filtered_correct:.3f} (↑{snr_filtered_correct/snr_original:.2f}x)',
                             fontsize=12, color='green')
            axes[1].axis('off')

            axes[2].imshow(filtered_wrong, cmap='gray')
            axes[2].set_title(f'SAR Params (Wrong)\nSNR: {snr_filtered_wrong:.3f} (↑{snr_filtered_wrong/snr_original:.2f}x)',
                             fontsize=12, color='red')
            axes[2].axis('off')

            axes[3].imshow(mask, cmap='gray')
            axes[3].set_title('Ground Truth', fontsize=12)
            axes[3].axis('off')

            plt.suptitle(f'RGB Sample: {Path(img_path).name}', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(f'{output_dir}/rgb_comparison.png', dpi=150, bbox_inches='tight')
            plt.close()

            print(f"  ✅ 保存: {output_dir}/rgb_comparison.png")

    print("\n" + "="*80)
    print("✅ 测试完成!")
    print("="*80)

if __name__ == "__main__":
    test_on_samples()
