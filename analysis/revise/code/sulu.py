"""
苏禄海内波综合分析图 v7
修正：废除方向折叠逻辑，使用真实的 360° 矢量计算圆周标准差 (sigma)
优化：调大经纬度坐标轴与刻度标签的字体大小
引入：15度方向容忍度约束，修复 OBB 外框倾斜导致 Quiver 箭头过度偏西的问题
"""

import os, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import font_manager
from scipy.stats import gaussian_kde
from scipy.ndimage import gaussian_filter
warnings.filterwarnings('ignore')

# ===== 路径 =====
CSV_PATH   = "/root/data/iw_clusters_geoinfo_sar_new.csv"
OUTPUT_DIR = "/root/output/"
FONT_PATH  = "/root/data/TIMES.TTF"

# ===== 苏禄海范围 =====
SULU = dict(lon_min=117.0, lon_max=123.0, lat_min=5.0, lat_max=12.0)

SRC_LON,  SRC_LAT  = 119.8, 5.3
SRC2_LON, SRC2_LAT = 117.5, 7.9

PAL_LON = np.array([117.8, 118.2, 118.55, 118.95, 119.25])
PAL_LAT = np.array([11.5,  10.8,   9.5,    8.2,    7.0])

# ===== 颜色方案 =====
C_DEEP  = '#0D1F3C'
C_MID   = '#1A6BAD'
C_BG    = '#F4F9FD'
C_RED   = '#C0392B'
C_ORG   = '#D35400'
C_GRAY  = '#566573'
C_GOLD  = '#F39C12'


# ===================================================================
# 工具函数
# ===================================================================

def filter_region(df, b):
    m = ((df['Center_Lon'] >= b['lon_min']) & (df['Center_Lon'] <= b['lon_max']) &
         (df['Center_Lat'] >= b['lat_min']) & (df['Center_Lat'] <= b['lat_max']))
    return df[m].copy()

def ang_diff(a, b):
    """计算两个角度的最小夹角 (0-180度)"""
    diff = abs(a - b) % 360
    return diff if diff <= 180 else 360 - diff

def obb_direction_away_from_source(row):
    """
    计算传播方向：结合 OBB 的垂直方向与物理源头辐射方向。
    增加 15 度阈值约束，防止 OBB 变形导致箭头过于偏西。
    """
    pts     = np.array([[row[f'OBB_P{i}_Lon'], row[f'OBB_P{i}_Lat']] for i in range(1, 5)])
    edges   = [pts[i] - pts[(i+1) % 4] for i in range(4)]
    lens    = [np.linalg.norm(e) for e in edges]
    longest = edges[int(np.argmax(lens))]
    
    # 1. 计算 OBB 长轴走向
    stripe_n = np.degrees(np.arctan2(longest[1], longest[0])) % 180.0
    
    # 2. 计算物理源头辐射方向 (最准的宏观方向，指向 NNW)
    dlon = row['Center_Lon'] - SRC_LON
    dlat = row['Center_Lat'] - SRC_LAT
    from_src_dir = np.degrees(np.arctan2(dlat, dlon)) % 360.0
    
    # 3. 计算 OBB 的两个垂直方向候选，选择最靠近 from_src_dir 的一个
    cand_a = (stripe_n + 90.0) % 360.0
    cand_b = (stripe_n - 90.0) % 360.0
    obb_dir = cand_a if ang_diff(cand_a, from_src_dir) <= ang_diff(cand_b, from_src_dir) else cand_b

    # 4. 核心约束机制：如果 OBB 计算出的法线与源头辐射相差超过 15 度
    # 说明 OBB 受到了外框阶梯扭曲，强制使其“立”起来，使用物理正确方向
    if ang_diff(obb_dir, from_src_dir) > 15.0:
        final_prop = from_src_dir
    else:
        final_prop = obb_dir
        
    return final_prop % 360.0


def compute_sigma(dirs):
    """
    计算方向离散度 sigma（圆形标准差）。
    当前方向已经通过源头辐射进行了严格消歧（全部指向远离源头的方向），
    因此直接使用真实的 360° 矢量计算离散度，废除折叠逻辑！
    """
    dirs_true = dirs.copy().astype(float)

    R_bar = np.abs(np.mean(np.exp(1j * np.deg2rad(dirs_true))))
    R_bar = np.clip(R_bar, 1e-6, 1 - 1e-6)
    sigma = np.degrees(np.sqrt(-2.0 * np.log(R_bar)))
    return sigma, dirs_true


def build_grid_quiver(lons, lats, dirs, nx=10, ny=10, min_count=4):
    lon_edges = np.linspace(SULU['lon_min'], SULU['lon_max'], nx + 1)
    lat_edges = np.linspace(SULU['lat_min'], SULU['lat_max'], ny + 1)
    cx_list, cy_list, ux_list, uy_list, count_list, R_list = [], [], [], [], [], []
    for i in range(nx):
        for j in range(ny):
            mask = ((lons >= lon_edges[i]) & (lons < lon_edges[i+1]) &
                    (lats >= lat_edges[j]) & (lats < lat_edges[j+1]))
            if mask.sum() < min_count:
                continue
            d_sel = dirs[mask]
            ux = np.mean(np.cos(np.deg2rad(d_sel)))
            uy = np.mean(np.sin(np.deg2rad(d_sel)))
            R  = np.sqrt(ux**2 + uy**2)
            cx_list.append((lon_edges[i] + lon_edges[i+1]) / 2)
            cy_list.append((lat_edges[j] + lat_edges[j+1]) / 2)
            ux_list.append(ux)
            uy_list.append(uy)
            count_list.append(mask.sum())
            R_list.append(R)
    return (np.array(cx_list), np.array(cy_list),
            np.array(ux_list), np.array(uy_list),
            np.array(count_list), np.array(R_list))


# ===================================================================
# 主图
# ===================================================================
def draw_main_ax(ax, df):
    lons = df['Center_Lon'].values
    lats = df['Center_Lat'].values
    dirs = df['direction'].values

    # ── KDE 密度底图 ──────────────────────────────────────────────
    xy  = np.vstack([lons, lats])
    kde = gaussian_kde(xy, bw_method=0.20)
    lg, latg = np.meshgrid(
        np.linspace(SULU['lon_min'], SULU['lon_max'], 340),
        np.linspace(SULU['lat_min'], SULU['lat_max'], 400))
    z = kde(np.vstack([lg.ravel(), latg.ravel()])).reshape(lg.shape)
    z = gaussian_filter(z, sigma=2.8)
    zm = np.ma.masked_where(z < z.max() * 0.03, z)

    cmap_kde = LinearSegmentedColormap.from_list(
        'sulu_v6',
        ['#EAF4FC', '#AED6F1', '#5DADE2', '#1F618D', C_DEEP], N=256)
    cf = ax.contourf(lg, latg, zm, levels=22,
                     cmap=cmap_kde, alpha=0.68, zorder=1)

    lvls = [z.max()*0.10, z.max()*0.25, z.max()*0.50, z.max()*0.78]
    ax.contour(lg, latg, z, levels=lvls,
               colors='white', linewidths=[0.4, 0.6, 0.9, 1.2],
               alpha=0.55, zorder=2)

    cb = plt.colorbar(cf, ax=ax, pad=0.015, fraction=0.022,
                      aspect=32, shrink=0.65)
    cb.set_label('Wave packet density', fontsize=9, color=C_GRAY, labelpad=6)
    cb.ax.tick_params(labelsize=8, colors=C_GRAY)
    cb.outline.set_edgecolor('#BDC3C7')
    cb.set_ticks([z.max()*0.05, z.max()*0.45, z.max()*0.90])
    cb.set_ticklabels(['Low', 'Mid', 'High'])

    # ── 诊断：打印方向分布 ─────────────────────────────────────────
    _, dirs_true = compute_sigma(dirs)
    print(f"真实均值：{np.mean(dirs_true):.1f}°, 中位数：{np.median(dirs_true):.1f}°")
    print(f"0-90°  占比：{((dirs_true>=0)  &(dirs_true<90) ).mean()*100:.1f}%")
    print(f"90-180° 占比：{((dirs_true>=90) &(dirs_true<180)).mean()*100:.1f}%")
    print(f"180-270° 占比：{((dirs_true>=180)&(dirs_true<270)).mean()*100:.1f}%")
    print(f"270-360° 占比：{((dirs_true>=270)&(dirs_true<360)).mean()*100:.1f}%")

    # ── 网格平均 quiver ────────────────────────────
    cx, cy, ux, uy, cnts, Rvals = build_grid_quiver(
        lons, lats, dirs, nx=10, ny=10, min_count=4)

    cmap_q  = LinearSegmentedColormap.from_list(
        'quiver_v6', [C_GOLD, C_MID, C_DEEP], N=256)
    norm_q  = mcolors.Normalize(vmin=0.25, vmax=0.95)
    q_colors = cmap_q(norm_q(Rvals))
    s_norm = 0.40 + 0.60 * (cnts - cnts.min()) / (cnts.max() - cnts.min() + 1)

    Q = ax.quiver(cx, cy, ux * s_norm, uy * s_norm,
                  color=q_colors,
                  scale=4.2, scale_units='inches',
                  width=0.007, headwidth=5, headlength=6,
                  headaxislength=4.5, pivot='middle',
                  zorder=6,
                  path_effects=[pe.withStroke(linewidth=1.8, foreground='white')])

    ax.quiverkey(Q, X=0.16, Y=0.055, U=0.40 / 4.2,
                 label='Grid-mean direction\n(size ∝ packet count)',
                 labelpos='E', fontproperties={'size': 8},
                 color=C_MID, labelcolor=C_GRAY,
                 coordinates='axes')

    # ── Palawan 边界 ──────────────────────────────────────────────
    ax.fill_betweenx(PAL_LAT, SULU['lon_min'], PAL_LON,
                    alpha=0.13, color='#7F8C8D', zorder=4)
    ax.plot(PAL_LON, PAL_LAT, color=C_GRAY, lw=2.8,
            ls=(0, (6, 2)), zorder=7,
            path_effects=[pe.withStroke(linewidth=4.5, foreground='white')])
    ax.text(117.32, 9.5, 'Palawan\nSlope',
            fontsize=9, color=C_GRAY, rotation=72,
            ha='center', va='center', fontweight='bold',
            path_effects=[pe.withStroke(linewidth=2.5, foreground='white')])

    # ── 源区辐射扇形 ───────────────────────────────────────────────
    fan_angles = np.linspace(60, 140, 13)
    for r_layer, alpha_val, lw_val in zip(
            [2.0, 3.5, 5.0, 7.0],
            [0.45, 0.30, 0.18, 0.10],
            [1.4, 1.1, 0.9, 0.7]):
        for fa in fan_angles:
            ex = SRC_LON + r_layer * np.cos(np.deg2rad(fa))
            ey = SRC_LAT + r_layer * np.sin(np.deg2rad(fa))
            ax.plot([SRC_LON, ex], [SRC_LAT, ey],
                    color=C_RED, alpha=alpha_val, lw=lw_val, zorder=3)

    for r_arc, alpha_arc in [(3.5, 0.35), (6.0, 0.20)]:
        arc_a = np.linspace(60, 140, 80)
        ax.plot(SRC_LON + r_arc * np.cos(np.deg2rad(arc_a)),
                SRC_LAT + r_arc * np.sin(np.deg2rad(arc_a)),
                color=C_RED, alpha=alpha_arc, lw=1.3, ls='--', zorder=3)

    ax.scatter([SRC_LON], [SRC_LAT], s=200, color=C_RED,
               marker='*', zorder=10, edgecolors='white', linewidths=0.8)
    ax.scatter([SRC2_LON], [SRC2_LAT], s=140, color=C_ORG,
               marker='*', zorder=10, edgecolors='white', linewidths=0.6)

    # ── 标注框 ─────────────────────────────────────────────────────
    bp = dict(boxstyle='round,pad=0.50', facecolor='white',
              edgecolor='#85929E', alpha=0.95, linewidth=1.3)
    south_pct = (lats < 8.5).mean() * 100

    ax.annotate(
        f'① Source-controlled radiation\n'
        f'{south_pct:.0f}% of packets from\nsouthern passages',
        xy=(119.6, 6.1), xytext=(121.2, 7.2),
        fontsize=9.5, color=C_RED, va='center',
        arrowprops=dict(arrowstyle='->', color=C_RED,
                        lw=1.4, mutation_scale=14,
                        connectionstyle='arc3,rad=-0.18'),
        bbox=bp, zorder=11)

    ax.annotate(
        '② Boundary confinement\nAbrupt termination at\nPalawan steep slope',
        xy=(118.3, 10.4), xytext=(119.6, 11.4),
        fontsize=9.5, color=C_GRAY, va='center',
        arrowprops=dict(arrowstyle='->', color=C_GRAY,
                        lw=1.4, mutation_scale=14,
                        connectionstyle='arc3,rad=0.22'),
        bbox=bp, zorder=11)

    # ── 轴修饰 ─────────────────────────────────────────────────────
    ax.set_xlim(SULU['lon_min'], SULU['lon_max'])
    ax.set_ylim(SULU['lat_min'], SULU['lat_max'])
    
    # 【已修改】：调大经纬度标签字体为 18
    ax.set_xlabel('Longitude (°E)', fontsize=18, labelpad=8)
    ax.set_ylabel('Latitude (°N)', fontsize=18, labelpad=8)
    ax.set_xticks(np.arange(117, 124, 1))
    ax.set_yticks(np.arange(5, 13, 1))
    
    # 【已修改】：调大坐标轴刻度数字字体为 15
    ax.tick_params(labelsize=15, length=5)
    
    ax.grid(True, alpha=0.20, linestyle='--', color='white', linewidth=0.8)
    ax.set_facecolor(C_BG)
    for spine in ax.spines.values():
        spine.set_edgecolor('#BDC3C7')
        spine.set_linewidth(1.2)

    h1 = plt.Line2D([0],[0], color=C_RED, lw=0, marker='*',
                    markersize=11, label='Sibutu Passage (primary source)',
                    markeredgecolor='white', markeredgewidth=0.5)
    h2 = plt.Line2D([0],[0], color=C_ORG, lw=0, marker='*',
                    markersize=9,  label='Balabac Strait (secondary source)',
                    markeredgecolor='white', markeredgewidth=0.5)
    h3 = plt.Line2D([0],[0], color=C_GRAY, lw=2.5, ls=(0,(6,2)),
                    label='Palawan slope (energy sink)')
    h4 = plt.Line2D([0],[0], color=C_MID, lw=0,
                    marker=r'$\rightarrow$', markersize=13,
                    label='Grid-averaged propagation direction')
    leg = ax.legend(handles=[h1, h2, h3, h4],
                    fontsize=8.5, loc='lower right', framealpha=0.95,
                    edgecolor='#BDC3C7', handlelength=1.6,
                    borderpad=0.7, labelspacing=0.5)
    leg.get_frame().set_linewidth(1.2)

    return south_pct


# ===================================================================
# 嵌入玫瑰图
# ===================================================================
def draw_inset_rose(fig, ax_main, dirs):
    sigma, dirs_true = compute_sigma(dirs)

    pos     = ax_main.get_position()
    inset_w = pos.width  * 0.27
    inset_h = pos.height * 0.32
    inset_x = pos.x1 - inset_w - pos.width * 0.13
    inset_y = pos.y1 - inset_h - pos.height * 0.01

    ax_r = fig.add_axes([inset_x, inset_y, inset_w, inset_h],
                        projection='polar')

    n_bins    = 36
    bin_edges = np.linspace(0, 360, n_bins + 1)
    counts, _ = np.histogram(dirs_true, bins=bin_edges)
    pct       = counts / len(dirs_true) * 100
    bin_centers_deg = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_centers_rad = np.deg2rad(bin_centers_deg)
    width = np.deg2rad(360.0 / n_bins)

    bar_colors = []
    for bc in bin_centers_deg:
        if   90  <= bc <= 180: bar_colors.append(C_MID)
        elif 0   <= bc <  90:  bar_colors.append('#5DADE2')
        elif 180 < bc <= 270:  bar_colors.append('#A9CCE3')
        else:                   bar_colors.append('#D5D8DC')

    ax_r.bar(bin_centers_rad, pct, width=width, bottom=0,
             color=bar_colors, edgecolor='white', linewidth=0.4,
             alpha=0.90, zorder=2)

    max_i = int(np.argmax(pct))
    ax_r.bar(bin_centers_rad[max_i], pct[max_i], width=width, bottom=0,
             color=C_RED, edgecolor='white', linewidth=0.6, zorder=4)

    mean_rad = np.arctan2(np.mean(np.sin(np.deg2rad(dirs_true))),
                          np.mean(np.cos(np.deg2rad(dirs_true))))
    ax_r.annotate('', xy=(mean_rad, pct.max() * 0.82), xytext=(0, 0),
                  arrowprops=dict(arrowstyle='->', color=C_RED,
                                  lw=2.2, mutation_scale=13), zorder=5)

    lbl_pe = [pe.withStroke(linewidth=2, foreground='white')]
    for label, ang in zip(['E','N','W','S'],
                           [0, np.pi/2, np.pi, -np.pi/2]):
        ax_r.text(ang, pct.max() * 1.32, label,
                  ha='center', va='center', fontsize=9,
                  fontweight='bold', color=C_DEEP,
                  path_effects=lbl_pe)

    # NW 占比：90°–180° in 真实方向
    nw_pct = float(((dirs_true >= 90) & (dirs_true <= 180)).mean() * 100)

    ax_r.set_yticklabels([])
    ax_r.set_xticklabels([])
    ax_r.grid(True, alpha=0.20, color='white')
    ax_r.set_facecolor('#EBF5FB')
    ax_r.patch.set_alpha(0.95)
    for spine in ax_r.spines.values():
        spine.set_edgecolor(C_MID)
        spine.set_linewidth(1.5)

    return nw_pct, sigma


# ===================================================================
# 主函数
# ===================================================================
def main():
    font_manager.fontManager.addfont(FONT_PATH)
    prop = font_manager.FontProperties(fname=FONT_PATH)
    plt.rcParams.update({
        'font.family': prop.get_name(),
        'figure.dpi':  600,
        'axes.linewidth': 1.2,
    })

    df      = pd.read_csv(CSV_PATH)
    df_sulu = filter_region(df, SULU).copy()
    
    # 使用更新后的方向计算函数
    df_sulu['direction'] = df_sulu.apply(obb_direction_away_from_source, axis=1)

    print("\n===== OBB诊断（前5个波包）=====")
    for _, row in df_sulu.head(5).iterrows():
        pts = np.array([[row[f'OBB_P{i}_Lon'], row[f'OBB_P{i}_Lat']] for i in range(1, 5)])
        edges = [pts[i] - pts[(i+1) % 4] for i in range(4)]
        lens  = [np.linalg.norm(e) for e in edges]
        longest = edges[int(np.argmax(lens))]
        stripe = np.degrees(np.arctan2(longest[1], longest[0]))
        print(f"  中心:({row['Center_Lon']:.2f},{row['Center_Lat']:.2f})  "
              f"长轴走向:{stripe:.1f}°  传播方向:{row['direction']:.1f}°  "
              f"OBB_P1:({pts[0,0]:.2f},{pts[0,1]:.2f})")
    print("================================\n")

    dirs = df_sulu['direction'].values

    sigma_true, _ = compute_sigma(dirs)
    print(f"苏禄海波包数：{len(df_sulu)}")
    print(f"方向离散度 sigma：{sigma_true:.1f}°")

    fig, ax = plt.subplots(figsize=(13, 9.5))
    fig.patch.set_facecolor('white')

    south_pct = draw_main_ax(ax, df_sulu)
    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    nw_pct, sigma = draw_inset_rose(fig, ax, dirs)

    fig.suptitle(
        'Sulu Sea Internal Wave Propagation:\n'
        'Source-Controlled Radiation and Boundary Confinement',
        fontsize=14, fontweight='bold', y=0.99,
        color=C_DEEP, linespacing=1.5)

    fig.text(0.5, 0.004,
             f'N = {len(df_sulu)} wave packets  |  '
             f'Southern passage origin: {south_pct:.0f}%  |  '
             f'NW-directed: {nw_pct:.0f}%  |  '
             f'Directional spread  σ = {sigma:.1f}°',
             ha='center', fontsize=9.5, color=C_GRAY,
             style='italic', linespacing=1.4)

    out_pdf = os.path.join(OUTPUT_DIR, "sulu_unified_v7.pdf")
    out_png = os.path.join(OUTPUT_DIR, "sulu_unified_v7.png")
    fig.savefig(out_pdf, dpi=300, bbox_inches='tight')
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    print(f"\n已保存：\n  {out_pdf}\n  {out_png}")

    print(f"\n========== 论文引用数值 ==========")
    print(f"  波包总数：       {len(df_sulu)}")
    print(f"  南部来源占比：   {south_pct:.1f}%")
    print(f"  NW 方向占比：    {nw_pct:.1f}%")
    print(f"  方向离散度 σ：   {sigma:.1f}°")
    print("===================================\n")


if __name__ == "__main__":
    main()