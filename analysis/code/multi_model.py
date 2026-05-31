import os
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from PIL import Image
from scipy.ndimage import zoom
from scipy.stats import ks_2samp, wasserstein_distance
from skimage.feature import graycomatrix, graycoprops
from skimage import filters
from scipy.spatial.distance import cosine, jensenshannon
from mpl_toolkits.axes_grid1 import make_axes_locatable
import warnings

# 忽略不必要的警告
warnings.filterwarnings('ignore')

# ============================================================================
# 1. 全局学术风格配置 (Paper-Ready Style)
# ============================================================================
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'axes.linewidth': 1.5,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.major.width': 1.2,
    'ytick.major.width': 1.2,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'legend.fontsize': 10,
    'savefig.bbox': 'tight',
    'savefig.transparent': False
})

# 专业配色：深蓝色(Optical), 暖橙色(SAR)
C_RGB = '#1E40AF' 
C_SAR = '#EA580C'

# ============================================================================
# 2. 文件路径定义
# ============================================================================
rgb_path = '/home/xiaobowen/project/internal_wave_detection_project/analysis/rgb/MODIS_TrueColor_2022-05-28_2Aqua.tiff'
rgb_mask_path = '/home/xiaobowen/project/internal_wave_detection_project/analysis/rgb_maksk/2022-05-28_2Aqua_mask.png'
sar_path = '/home/xiaobowen/project/internal_wave_detection_project/analysis/sars/S1A_IW_GRDH_1SDV_20220528T102524_20220528T102553_043410_052F02_7533.tif'
sar_mask_path = '/home/xiaobowen/project/internal_wave_detection_project/analysis/sars_masks/S1A_IW_GRDH_1SDV_20220528T102524_20220528T102553_043410_052F02_7533.png'
output_dir = 'multimodal_figures'
os.makedirs(output_dir, exist_ok=True)

# ============================================================================
# 3. 核心处理函数
# ============================================================================

def load_image(path):
    """加载图像并处理不同格式"""
    try:
        if path.endswith(('.tiff', '.tif')):
            with rasterio.open(path) as src:
                data = src.read()
                if data.shape[0] == 3:
                    return np.transpose(data, (1, 2, 0))
                return data[0]
        else:
            return np.array(Image.open(path))
    except Exception as e:
        print(f"Loading failed {path}: {e}")
        return None

def normalize(img):
    """归一化到 0-1"""
    img = img.astype(np.float32)
    return (img - img.min()) / (img.max() - img.min() + 1e-8)

def rgb_to_gray(rgb):
    """RGB 转灰度"""
    if len(rgb.shape) == 3:
        return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])
    return rgb

def resize_mask(mask, target_shape):
    """缩放掩码以匹配图像维度"""
    zoom_factors = (target_shape[0] / mask.shape[0], target_shape[1] / mask.shape[1])
    return zoom(mask, zoom_factors, order=0)

def compute_metrics(data1, data2, bins=60):
    """计算分布相似性指标"""
    d1, d2 = data1.flatten(), data2.flatten()
    ks_stat, ks_pval = ks_2samp(d1, d2)
    
    # 构建直方图概率分布用于 JS 散度
    h1, b = np.histogram(d1, bins=bins, range=(0,1), density=True)
    h2, _ = np.histogram(d2, bins=b, density=True)
    p1 = h1 / (h1.sum() + 1e-10) + 1e-10
    p2 = h2 / (h2.sum() + 1e-10) + 1e-10
    
    return {
        'ks_statistic': ks_stat,
        'ks_pvalue': ks_pval,
        'wasserstein': wasserstein_distance(d1, d2),
        'js_divergence': jensenshannon(p1, p2),
        'bhattacharyya': np.sum(np.sqrt(p1 * p2))
    }

def extract_glcm(gray_img, mask):
    """提取 GLCM 纹理特征"""
    gray_q = (gray_img * 255).astype(np.uint8)
    masked = gray_q.copy()
    masked[mask == 0] = 0
    try:
        glcm = graycomatrix(masked, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], 256, symmetric=True, normed=True)
        return {k: graycoprops(glcm, k).mean() for k in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']}
    except:
        return {k: 0 for k in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']}

# ============================================================================
# 4. 绘图增强函数 (Nature-Style)
# ============================================================================

def plot_refined_hist(ax, data1, data2, xlabel):
    """绘制高质感阶梯填充直方图"""
    # Optical (RGB)
    ax.hist(data1.flatten(), bins=60, density=True, histtype='stepfilled', alpha=0.15, color=C_RGB, label='Optical (RGB)')
    ax.hist(data1.flatten(), bins=60, density=True, histtype='step', lw=1.8, color=C_RGB)
    # Microwave (SAR)
    ax.hist(data2.flatten(), bins=60, density=True, histtype='stepfilled', alpha=0.15, color=C_SAR, label='Microwave (SAR)')
    ax.hist(data2.flatten(), bins=60, density=True, histtype='step', lw=1.8, color=C_SAR)
    
    # 不显示坐标轴标签
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(frameon=False)

def plot_refined_map(data, path, title):
    """使用 viridis 色盘绘制高对比度空间图"""
    fig, ax = plt.subplots(figsize=(12, 8))
    # 抑制 1% 的离群值以增强视觉对比
    im = ax.imshow(data, cmap='viridis', aspect='auto', vmin=0, vmax=np.percentile(data, 99))
    ax.axis('off')
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.1)
    plt.colorbar(im, cax=cax).ax.tick_params(labelsize=10)
    
    plt.savefig(path, dpi=300, bbox_inches='tight', transparent=True)
    plt.close()

# ============================================================================
# 5. 主执行流程 (Main Pipeline)
# ============================================================================

print("Step 1: Loading and Preprocessing...")
rgb_img = load_image(rgb_path)
rgb_mask = load_image(rgb_mask_path)
sar_img = load_image(sar_path)
sar_mask = load_image(sar_mask_path)

if any(x is None for x in [rgb_img, rgb_mask, sar_img, sar_mask]):
    print("Error: Missing data files. Check paths.")
    exit()

# 归一化与对齐
rgb_gray = normalize(rgb_to_gray(rgb_img))
sar_gray = normalize(sar_img if sar_img.ndim == 2 else rgb_to_gray(sar_img))

rgb_m = (resize_mask(rgb_mask, rgb_gray.shape) > 0).astype(np.uint8)
sar_m = (resize_mask(sar_mask, sar_gray.shape) > 0).astype(np.uint8)

print("Step 2: Computing Gradients and Textures...")
# 使用 Sobel 算子
rgb_grad = filters.sobel(rgb_gray)
sar_grad = filters.sobel(sar_gray)

# 提取掩码内的有效数据
rgb_v = rgb_gray[rgb_m > 0]
sar_v = sar_gray[sar_m > 0]
rgb_gv = rgb_grad[rgb_m > 0]
sar_gv = sar_grad[sar_m > 0]

print("Step 3: Generating Individual Figures...")

# (c) Intensity Distribution
fig_c, ax_c = plt.subplots(figsize=(8, 6))
plot_refined_hist(ax_c, rgb_v, sar_v, 'Normalized Intensity')
plt.tight_layout()
fig_c.savefig(os.path.join(output_dir, 'c_intensity_distribution.pdf'), dpi=300, bbox_inches='tight')
print("  ✓ c_intensity_distribution.png saved")
plt.close(fig_c)

# (d) Gradient Distribution
fig_d, ax_d = plt.subplots(figsize=(8, 6))
plot_refined_hist(ax_d, rgb_gv, sar_gv, 'Gradient Magnitude')
plt.tight_layout()
fig_d.savefig(os.path.join(output_dir, 'd_gradient_distribution.pdf'), dpi=300, bbox_inches='tight')
print("  ✓ d_gradient_distribution.png saved")
plt.close(fig_d)

# (e) & (f) 空间梯度图
plot_refined_map(rgb_grad, os.path.join(output_dir, 'e_rgb_gradient_map.pdf'), 'RGB Gradient')
print("  ✓ e_rgb_gradient_map.png saved")
plot_refined_map(sar_grad, os.path.join(output_dir, 'f_sar_gradient_map.pdf'), 'SAR Gradient')
print("  ✓ f_sar_gradient_map.png saved")

print(f"\n✓ All figures saved to: {output_dir}/")

# ============================================================================
# 6. 定量分析报告
# ============================================================================

print("\nStep 4: Final Quantitative Report...")
m_int = compute_metrics(rgb_v, sar_v)
m_grad = compute_metrics(rgb_gv, sar_gv)
rgb_tex = extract_glcm(rgb_gray, rgb_m)
sar_tex = extract_glcm(sar_gray, sar_m)

# 计算 GLCM 相似度
glcm_sim = 1 - cosine(list(rgb_tex.values()), list(sar_tex.values()))

print("\n" + "="*80)
print("CROSS-MODAL DISTRIBUTION SIMILARITY ANALYSIS")
print("="*80)

print("\n【1. INTENSITY DISTRIBUTION COMPARISON】")
print("-" * 80)
print(f"  Kolmogorov-Smirnov Test:")
print(f"    • K-S Statistic:           {m_int['ks_statistic']:.4f}")
print(f"    • P-value:                 {m_int['ks_pvalue']:.4e}")
print(f"      → Interpretation: {'Distributions are SIGNIFICANTLY DIFFERENT' if m_int['ks_pvalue'] < 0.05 else 'No significant difference'}")

print(f"\n  Distance Metrics (Lower = More Similar):")
print(f"    • Wasserstein Distance:    {m_int['wasserstein']:.4f}")
print(f"    • Jensen-Shannon Div.:     {m_int['js_divergence']:.4f}  (Range: 0-1)")

print(f"\n  Similarity Metrics (Higher = More Similar):")
print(f"    • Bhattacharyya Coeff.:    {m_int['bhattacharyya']:.4f}  (Range: 0-1)")

print("\n【2. GRADIENT DISTRIBUTION COMPARISON】")
print("-" * 80)
print(f"  Kolmogorov-Smirnov Test:")
print(f"    • K-S Statistic:           {m_grad['ks_statistic']:.4f}")
print(f"    • P-value:                 {m_grad['ks_pvalue']:.4e}")
print(f"      → Interpretation: {'Distributions are SIGNIFICANTLY DIFFERENT' if m_grad['ks_pvalue'] < 0.05 else 'No significant difference'}")

print(f"\n  Distance Metrics (Lower = More Similar):")
print(f"    • Wasserstein Distance:    {m_grad['wasserstein']:.4f}")
print(f"    • Jensen-Shannon Div.:     {m_grad['js_divergence']:.4f}  (Range: 0-1)")

print(f"\n  Similarity Metrics (Higher = More Similar):")
print(f"    • Bhattacharyya Coeff.:    {m_grad['bhattacharyya']:.4f}  (Range: 0-1)")

print("\n【3. GLCM TEXTURE FEATURES】")
print("-" * 80)
print("  RGB GLCM Features:")
for key, val in rgb_tex.items():
    print(f"    • {key:15s}: {val:.4f}")

print("\n  SAR GLCM Features:")
for key, val in sar_tex.items():
    print(f"    • {key:15s}: {val:.4f}")

print(f"\n  Cosine Similarity: {glcm_sim:.4f}")

print("\n" + "="*80)
print("【SUMMARY】")
print("="*80)
print(f"\nRecommended metrics for cross-modal comparison:")
print(f"  1. Jensen-Shannon Divergence (0=identical, 1=completely different)")
print(f"     - Intensity: {m_int['js_divergence']:.4f}")
print(f"     - Gradient:  {m_grad['js_divergence']:.4f}")
print(f"  2. Bhattacharyya Coefficient (0=no overlap, 1=identical)")
print(f"     - Intensity: {m_int['bhattacharyya']:.4f}")
print(f"     - Gradient:  {m_grad['bhattacharyya']:.4f}")
print("="*80)

print(f"\n✓ Analysis complete!")