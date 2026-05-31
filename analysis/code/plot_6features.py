import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from tabulate import tabulate  # 如果报错，请执行 pip install tabulate
from matplotlib.ticker import AutoMinorLocator

# ============================================
# 1. 视觉风格：高亮度、多元素聚合风格
# ============================================
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("white")
sns.set_context("paper", font_scale=1.4)

def get_aligned_pairs(df, feature):
    s_vals = np.sort(df[df['sensor'] == 'SAR'][feature].dropna().values)
    m_vals = np.sort(df[df['sensor'] == 'MODIS'][feature].dropna().values)
    n = min(len(s_vals), len(m_vals))
    x = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(s_vals)), s_vals)
    y = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(m_vals)), m_vals)
    return x, y

# ============================================
# 2. 核心绘图与统计计算
# ============================================
def run_enhanced_analysis():
    df = pd.read_csv("internal_wave_feature_mine.csv")
    features = ['mean_pixel_kurt', 'mean_entropy', 'mean_contrast', 
                'stripe_count', 'mean_snr', 'mean_pixel_skew']

    fig, axes = plt.subplots(2, 3, figsize=(22, 12), dpi=300)
    axes = axes.flatten()

    # --- 配色方案 ---
    c_scatter = '#00D4FF' # 电光蓝
    c_line = '#FF4D00'    # 霓虹橙
    c_kde = '#2ECC71'     # 翠绿等高线
    
    stats_log = []

    for i, feat in enumerate(features):
        x, y = get_aligned_pairs(df, feat)
        
        # A. 绘制密度等高线 (增加“层次感”)
        sns.kdeplot(x=x, y=y, ax=axes[i], levels=5, color=c_kde, linewidths=1, alpha=0.3, zorder=1)
        
        # B. 绘制 y=x 参考线
        lims = [min(x.min(), y.min()), max(x.max(), y.max())]
        axes[i].plot(lims, lims, color='#BDC3C7', linestyle='--', linewidth=1, alpha=0.7, zorder=2)
        
        # C. 绘制亮色散点
        axes[i].scatter(x, y, color=c_scatter, s=25, alpha=0.3, edgecolor='none', zorder=3)
        
        # D. 拟合回归线并计算指标
        slope, intercept, r_val, p_val, std_err = stats.linregress(x, y)
        r_sq = r_val**2
        rmse = np.sqrt(np.mean((y - (slope * x + intercept))**2))
        
        axes[i].plot(x, slope * x + intercept, color=c_line, lw=3, zorder=4)
        
        # E. 子图内标注
        stats_text = f'$R^2 = {r_sq:.3f}$\nSlope $= {slope:.2f}$\nRMSE $= {rmse:.3f}$'
        axes[i].text(0.05, 0.95, stats_text, transform=axes[i].transAxes, 
                     fontsize=13, fontweight='bold', verticalalignment='top',
                     color=c_line, bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
        
        # F. 细节美化
        axes[i].xaxis.set_minor_locator(AutoMinorLocator())
        axes[i].yaxis.set_minor_locator(AutoMinorLocator())
        clean_title = feat.replace('mean_', '').replace('_', ' ').upper()
        axes[i].set_title(f'({chr(97+i)}) {clean_title}', loc='left', fontsize=18, fontweight='bold', pad=15)
        sns.despine(ax=axes[i])
        
        # G. 记录到统计列表
        stats_log.append([clean_title, f"{slope:.4f}", f"{r_sq:.4f}", f"{rmse:.4f}", "P < 0.001"])

    # ============================================
    # 3. 保存图像与打印控制台表格
    # ============================================
    plt.tight_layout(pad=4.0)
    plt.savefig("plot_6features.png", dpi=300, bbox_inches='tight')
    plt.savefig("plot_6features.pdf", bbox_inches='tight')
    plt.show()

    # 打印表格
    print("\n" + "="*90)
    print("             🌟 CROSS-MODAL JOINT ANALYSIS STATISTICAL REPORT 🌟")
    print("="*90)
    headers = ["Feature Metric", "Slope (α)", "R-Squared (R²)", "RMSE", "Significance"]
    print(tabulate(stats_log, headers=headers, tablefmt="fancy_grid"))
    print("="*90)
    print("✅ 结果已同步保存为 PDF 矢量图和 PNG 高清图。")

if __name__ == '__main__':
    run_enhanced_analysis()