"""
安达曼海内波统计分析 — 单张综合图 v4
改进：画布32×20，三个注释移至图外底部文字区，图例保留图内右下角
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
from scipy.stats import gaussian_kde
from scipy.ndimage import gaussian_filter
import warnings
warnings.filterwarnings('ignore')
from matplotlib import font_manager

# ===== 路径配置 =====
CSV_PATH   = "/root/data/iw_clusters_geoinfo_sar.csv"
OUTPUT_DIR = "/root/output/"

# ===== 地理边界 =====
AND = dict(lon_min=90.0, lon_max=100.0, lat_min=5.0, lat_max=10.0)

FONT_PATH = "/root/data/TIMES.TTF"
RIDGE_LON = 92.8

# ===================================================================
# 工具函数
# ===================================================================

def filter_region(df, bounds):
    m = ((df['Center_Lon'] >= bounds['lon_min']) & (df['Center_Lon'] <= bounds['lon_max']) &
         (df['Center_Lat'] >= bounds['lat_min']) & (df['Center_Lat'] <= bounds['lat_max']))
    return df[m].copy()


def obb_propagation_direction(row):
    pts = np.array([[row[f'OBB_P{i}_Lon'], row[f'OBB_P{i}_Lat']] for i in range(1, 5)])
    edges   = [pts[i] - pts[(i+1) % 4] for i in range(4)]
    lengths = [np.linalg.norm(e) for e in edges]
    longest = edges[int(np.argmax(lengths))]
    stripe  = np.degrees(np.arctan2(longest[1], longest[0]))

    prop_a = stripe + 90.0
    prop_b = stripe - 90.0
    dx_a = np.cos(np.deg2rad(prop_a))
    dx_b = np.cos(np.deg2rad(prop_b))

    if row['Center_Lon'] >= RIDGE_LON:
        prop = prop_a if dx_a > dx_b else prop_b
    else:
        prop = prop_a if dx_a < dx_b else prop_b

    return prop % 360.0


def obb_long_axis_endpoints(row):
    pts = np.array([[row[f'OBB_P{i}_Lon'], row[f'OBB_P{i}_Lat']] for i in range(1, 5)])
    edges   = [pts[i] - pts[(i+1) % 4] for i in range(4)]
    lengths = [np.linalg.norm(e) for e in edges]
    idx     = int(np.argmax(lengths))
    return pts[idx], pts[(idx + 1) % 4]


# ===================================================================
# 主绘图函数
# ===================================================================

def draw_unified_figure(df_and):

    # ---------------------------------------------------------------
    # 画布：主图区 + 底部注释区
    # 用 GridSpec 分割：上方主图占 85%，下方注释区占 15%
    # ---------------------------------------------------------------
    fig = plt.figure(figsize=(40, 35))
    fig.patch.set_alpha(0.0)

    from matplotlib.gridspec import GridSpec
    gs = GridSpec(2, 1, figure=fig,
                  height_ratios=[8.5, 1.5],
                  hspace=0.08)

    ax  = fig.add_subplot(gs[0])   # 主图
    ax_ann = fig.add_subplot(gs[1])  # 注释区
    ax.patch.set_alpha(0.0)
    ax_ann.patch.set_alpha(0.0)
    ax_ann.axis('off')             # 注释区不需要坐标轴

    lons = df_and['Center_Lon'].values
    lats = df_and['Center_Lat'].values
    dirs = df_and['direction'].values

    # ---------------------------------------------------------------
    # 层1: 核密度热力图
    # ---------------------------------------------------------------
    xy  = np.vstack([lons, lats])
    kde = gaussian_kde(xy, bw_method=0.22)
    lon_g, lat_g = np.meshgrid(
        np.linspace(AND['lon_min'], AND['lon_max'], 400),
        np.linspace(AND['lat_min'], AND['lat_max'], 250)
    )
    z = kde(np.vstack([lon_g.ravel(), lat_g.ravel()])).reshape(lon_g.shape)
    z_smooth = gaussian_filter(z, sigma=2)
    z_masked = np.ma.masked_where(z_smooth < z_smooth.max() * 0.05, z_smooth)

    ax.contourf(lon_g, lat_g, z_masked, levels=20,
                cmap='YlOrRd', alpha=0.55, zorder=1)

    # ---------------------------------------------------------------
    # 层2: 条纹线段
    # ---------------------------------------------------------------
    segments_east, segments_west = [], []
    for _, row in df_and.iterrows():
        p1, p2 = obb_long_axis_endpoints(row)
        seg = [[p1[0], p1[1]], [p2[0], p2[1]]]
        if row['Center_Lon'] >= RIDGE_LON:
            segments_east.append(seg)
        else:
            segments_west.append(seg)

    if segments_west:
        ax.add_collection(LineCollection(segments_west, linewidths=3.0,
                                         colors='#5DADE2', alpha=0.60, zorder=2))
    if segments_east:
        ax.add_collection(LineCollection(segments_east, linewidths=3.5,
                                         colors='#1A237E', alpha=0.50, zorder=3))

    # ---------------------------------------------------------------
    # 层3: 传播方向箭头
    # ---------------------------------------------------------------
    arrow_len        = 0.16
    COLOR_WEST       = '#1A5276'
    COLOR_EAST_NORTH = '#E67E22'
    COLOR_EAST_EAST  = '#C0392B'
    COLOR_EAST_SOUTH = '#6C3483'

    for _, row in df_and.iterrows():
        cx, cy = row['Center_Lon'], row['Center_Lat']
        d      = row['direction']
        dx     = arrow_len * np.cos(np.deg2rad(d))
        dy     = arrow_len * np.sin(np.deg2rad(d))

        if row['Center_Lon'] < RIDGE_LON:
            color = COLOR_WEST
        else:
            threshold = arrow_len * np.sin(np.deg2rad(20))
            if dy > threshold:
                color = COLOR_EAST_NORTH
            elif dy < -threshold:
                color = COLOR_EAST_SOUTH
            else:
                color = COLOR_EAST_EAST

        ax.annotate('',
                    xy=(cx + dx, cy + dy),
                    xytext=(cx - dx, cy - dy),
                    arrowprops=dict(arrowstyle='->', color=color,
                                    lw=3.0, mutation_scale=25),
                    zorder=4)

    # ---------------------------------------------------------------
    # 海岭轴线 + 图内标签（仅保留轴线文字，无注释框）
    # ---------------------------------------------------------------
    ax.axvline(x=RIDGE_LON, color='#2C3E50', lw=4.0, ls='--', zorder=6)
    ax.text(RIDGE_LON - 0.12, AND['lat_min'] + 0.2,
            'Andaman–Nicobar Ridge', fontsize=34, color='#2C3E50',
            rotation=90, va='bottom', ha='right', zorder=7, fontweight='bold')

    # ---------------------------------------------------------------
    # 图内：①②③ 编号标记（小圆点 + 数字，不含长文字）
    # ---------------------------------------------------------------
    marker_props = dict(fontsize=34, fontweight='bold', zorder=8,
                        bbox=dict(boxstyle='circle,pad=0.3', facecolor='white',
                                  edgecolor='#AAAAAA', alpha=0.92, lw=2.0))

    # ① 海岭处
    ax.text(RIDGE_LON + 0.05, 9.3, '①', color='#1A5276', **marker_props)
    # ② 东盆地中部
    ax.text(96.2, 5.5, '②', color='#6E2C00', **marker_props)
    # ③ 右上角
    ax.text(99.5, 9.6, '③', color='#4A235A', **marker_props)

    # ---------------------------------------------------------------
    # 图形修饰
    # ---------------------------------------------------------------
    ax.set_xlim(AND['lon_min'], AND['lon_max'])
    ax.set_ylim(AND['lat_min'], AND['lat_max'])
    ax.set_xlabel('Longitude (°E)', fontsize=34, labelpad=10)
    ax.set_ylabel('Latitude (°N)', fontsize=34, labelpad=10)

    ax.set_xticks(np.arange(92, 101, 2))   # 去掉 90，从 92 开始
    ax.set_yticks(np.arange(6, 11, 1))     # 去掉 5，从 6 开始
    ax.tick_params(labelsize=42, width=2, length=8)  # 字体从 34 → 42
    ax.grid(True, alpha=0.3, linestyle='--', color='gray', linewidth=1.5)

    for spine in ax.spines.values():
        spine.set_linewidth(2.0)

    # ---------------------------------------------------------------
    # 图例（图内右下角）
    # ---------------------------------------------------------------
    h_east_stripe  = mpatches.Patch(color='#1A237E', alpha=0.6, label='Stripes (east basin)')
    h_west_stripe  = mpatches.Patch(color='#5DADE2', alpha=0.6, label='Stripes (west basin)')
    h_arrow_north  = plt.Line2D([0],[0], color=COLOR_EAST_NORTH, lw=4.0,
                                 marker='>', markersize=15, label='Prop. dir. (east, northward)')
    h_arrow_east   = plt.Line2D([0],[0], color=COLOR_EAST_EAST,  lw=4.0,
                                 marker='>', markersize=15, label='Prop. dir. (east, eastward)')
    h_arrow_south  = plt.Line2D([0],[0], color=COLOR_EAST_SOUTH, lw=4.0,
                                 marker='>', markersize=15, label='Prop. dir. (east, southward)')
    h_arrow_west   = plt.Line2D([0],[0], color=COLOR_WEST, lw=4.0,
                                 marker='<', markersize=15, label='Prop. dir. (west)')
    h_ridge        = plt.Line2D([0],[0], color='#2C3E50', lw=4.0, ls='--', label='Ridge axis')

    ax.legend(handles=[h_east_stripe, h_west_stripe,
                        h_arrow_north, h_arrow_east, h_arrow_south,
                        h_arrow_west, h_ridge],
              fontsize=28, loc='lower right', framealpha=0.88,
              edgecolor='gray', borderpad=0.8, labelspacing=0.5)

    # ---------------------------------------------------------------
    # 底部注释区：①②③ 说明文字，横排三列
    # ---------------------------------------------------------------
    east_pct = (lons >= RIDGE_LON).mean() * 100
    angles_rad = np.deg2rad(dirs)
    R_bar = np.abs(np.mean(np.exp(1j * angles_rad)))
    R_bar = np.clip(R_bar, 1e-6, 1 - 1e-6)
    sigma = np.degrees(np.sqrt(-2.0 * np.log(R_bar)))

    ann_kw = dict(transform=ax_ann.transAxes, fontsize=30,
                  va='center', ha='center',
                  bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                            edgecolor='#AAAAAA', alpha=0.90, lw=2.0))

    ax_ann.text(0.17, 0.55,
                f'① Ridge barrier\n{east_pct:.0f}% of packets east of ridge',
                color='#1A5276', **ann_kw)

    ax_ann.text(0.50, 0.55,
                '② Mesh-like interference\n(multiple generation segments, pathway interaction)',
                color='#6E2C00', **ann_kw)

    ax_ann.text(0.83, 0.55,
                f'③ Multi-directional anisotropy\nσ = {sigma:.1f}° (direction spread)',
                color='#4A235A', **ann_kw)

    return fig, east_pct, sigma


# ===================================================================
# 主函数
# ===================================================================

def main():
    font_manager.fontManager.addfont(FONT_PATH)
    prop = font_manager.FontProperties(fname=FONT_PATH)
    plt.rcParams.update({
        'font.family': prop.get_name(),
        'font.size': 34,
        'figure.dpi': 150,
    })

    df     = pd.read_csv(CSV_PATH)
    df_and = filter_region(df, AND)
    df_and = df_and.copy()
    df_and['direction'] = df_and.apply(obb_propagation_direction, axis=1)

    print(f"安达曼海波包数：{len(df_and)}")

    fig, east_pct, sigma = draw_unified_figure(df_and)

    out_pdf = os.path.join(OUTPUT_DIR, "andaman_unified.pdf")
    out_png = os.path.join(OUTPUT_DIR, "andaman_unified.png")

    fig.savefig(out_pdf, dpi=300, bbox_inches='tight', transparent=True)
    fig.savefig(out_png, dpi=300, bbox_inches='tight', transparent=True)

    print(f"\n已保存：\n  {out_pdf}\n  {out_png}")
    print(f"\n========== 论文引用数值 ==========")
    print(f"  波包总数：    {len(df_and)}")
    print(f"  海岭以东占比：{east_pct:.1f}%")
    print(f"  方向离散度 σ：{sigma:.1f}°")
    print("===================================\n")


if __name__ == "__main__":
    main()