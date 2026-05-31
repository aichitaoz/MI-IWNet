import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import skew
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.font_manager as fm

# ===================== 1. 核心配置 =====================
MIN_AREA = 50
NPY_FILE = "./data/area_data.npy"
FOLDERS = ["./data/mask_rgb", "./data/sar1", "./data/sar2"]

# 名称映射：文件夹名 -> 论文显示名
NAME_MAP = {
    "mask_rgb": "MODIS-N",
    "sar1": "S1-M",
    "sar2": "S1-G"  # 这里暂定sar2为S1-G，您可以根据需要修改
}

# 字体与配色
FONT_FILENAME = 'data/TIMES.TTF' 
COLORS = ['#E64B35', '#4DBBD5', '#00A087'] # Nature 风格红、蓝、绿

# Table 2 统计区间
BINS_TABLE = [50, 200, 500, 1000, 2000, 5000]
BIN_LABELS = ['50-200', '200-500', '500-1k', '1k-2k', '2k-5k']

# ===================== 2. 环境初始化 =====================
def init_style():
    """初始化字体和全局绘图样式"""
    base_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
    font_path = os.path.join(base_dir, FONT_FILENAME)
    
    custom_font = None
    if os.path.exists(font_path):
        try:
            fm.fontManager.addfont(font_path)
            prop = fm.FontProperties(fname=font_path)
            mpl.rcParams['font.family'] = prop.get_name()
            custom_font = prop
            print(f"✅ 字体加载成功: {prop.get_name()}")
        except:
            mpl.rcParams['font.family'] = 'serif'
    else:
        print(f"⚠️ 未找到字体 {font_path}，使用默认衬线字体")
        mpl.rcParams['font.family'] = 'serif'

    mpl.rcParams.update({
        'axes.unicode_minus': False,
        'axes.linewidth': 1.2,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'pdf.fonttype': 42
    })
    return custom_font

# ===================== 3. 数据处理模块 =====================
def get_areas_from_mask(mask_path, min_area):
    """从单张图提取连通域面积"""
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None: return []
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    num, _, stats, _ = cv2.connectedComponentsWithStats(binary)
    return [stats[i, cv2.CC_STAT_AREA] for i in range(1, num) if stats[i, cv2.CC_STAT_AREA] >= min_area]

def load_or_process_data():
    """获取全量面积数据"""
    if os.path.exists(NPY_FILE):
        print(f"📦 发现缓存，直接加载: {NPY_FILE}")
        return np.load(NPY_FILE, allow_pickle=True).item()
    
    print("🚀 缓存不存在，开始扫描图片提取数据...")
    all_areas = {}
    for folder in FOLDERS:
        p = Path(folder)
        if not p.exists(): continue
        areas = []
        files = list(p.glob("*.png")) + list(p.glob("*.jpg"))
        for f in files:
            areas.extend(get_areas_from_mask(str(f), MIN_AREA))
        all_areas[p.name] = areas
        print(f"   - {p.name}: 提取到 {len(areas)} 个目标")
    
    np.save(NPY_FILE, all_areas)
    return all_areas

# ===================== 4. 统计与绘图核心 =====================
def run_integrated_analysis(all_areas, custom_font):
    # 样式准备
    try:
        plt.style.use('seaborn-v0_8-paper')
    except:
        plt.style.use('seaborn-paper')
    
    fp = custom_font if custom_font else fm.FontProperties(family='serif')
    fig, ax1 = plt.subplots(figsize=(7, 4.8), dpi=300)
    ax2 = ax1.twinx()
    
    # 绘图 Log 分区
    bins_plot = np.logspace(np.log10(MIN_AREA), np.log10(100000), 100)
    bin_centers = (bins_plot[:-1] + bins_plot[1:]) / 2
    
    table_rows = []
    legend_lines = []

    # 核心循环：按文件夹处理数据
    for i, folder_name in enumerate(FOLDERS):
        key = Path(folder_name).name
        if key not in all_areas: continue
        
        areas = np.array(all_areas[key], dtype=np.float64)
        if len(areas) == 0: continue
        
        # A. 名称转换
        display_name = NAME_MAP.get(key, key)
        color = COLORS[i % len(COLORS)]
        
        # B. 计算 Table 2 统计数据
        total = len(areas)
        dist = [round((np.sum((areas >= BINS_TABLE[j]) & (areas < BINS_TABLE[j+1])) / total) * 100, 1) for j in range(len(BINS_TABLE)-1)]
        row = [display_name, total] + dist + [int(np.median(areas)), round(np.mean(areas), 1), round(skew(areas), 2)]
        table_rows.append(row)

        # C. 绘制 PDF (实线阴影)
        cnt, _ = np.histogram(areas, bins=bins_plot)
        pdf = cnt / cnt.sum()
        smooth_pdf = gaussian_filter1d(pdf, sigma=1.5)
        l, = ax1.plot(bin_centers, smooth_pdf, color=color, lw=2, label=display_name, zorder=10)
        ax1.fill_between(bin_centers, smooth_pdf, alpha=0.15, color=color, zorder=1)
        legend_lines.append(l)

        # D. 绘制 CDF (点虚线)
        sorted_areas = np.sort(areas)
        cdf_y = np.arange(len(sorted_areas)) / float(len(sorted_areas))
        ax2.plot(sorted_areas, cdf_y * 100, color=color, linestyle=(0, (3, 1, 1, 1)), lw=1.2, alpha=0.6)

    # --- 完善表格展示与保存 ---
    cols = ['Dataset', 'Count'] + BIN_LABELS + ['Med.', 'Mean', 'Skew.']
    df = pd.DataFrame(table_rows, columns=cols)
    print("\n" + "="*80 + "\nTable 2: Statistical Characterization\n" + "-"*80)
    print(df.to_string(index=False))
    print("="*80)
    df.to_csv("table2_results.csv", index=False)

    # --- 完善图形细节 ---
    ax1.set_xscale('log')
    ax1.set_xlabel('Area (pixels)', fontproperties=fp, fontsize=12, fontweight='bold')
    ax1.set_ylabel('Probability Density (PDF)', fontproperties=fp, fontsize=12, fontweight='bold')
    ax2.set_ylabel('Cumulative Percentage (%)', fontproperties=fp, fontsize=12, fontweight='bold', rotation=270, labelpad=18)
    
    # 坐标轴数字字体
    for label in ax1.get_xticklabels() + ax1.get_yticklabels() + ax2.get_yticklabels():
        label.set_fontproperties(fp)
        label.set_fontsize(10.5)

    ax1.legend(handles=legend_lines, loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=3, frameon=False, prop=fp)
    
    plt.tight_layout()
    plt.savefig("area_distribution.pdf", bbox_inches='tight')
    plt.savefig("area_distribution.png", dpi=300, bbox_inches='tight')
    print(f"\n✅ 分析完成！已生成：\n1. table2_results.csv (统计表)\n2. area_distribution.pdf (矢量图)\n3. area_distribution.png (高清预览图)")

# ===================== 5. 执行 =====================
if __name__ == "__main__":
    c_font = init_style()
    data = load_or_process_data()
    if data:
        run_integrated_analysis(data, c_font)
    else:
        print("❌ 错误：未获取到任何有效数据。")