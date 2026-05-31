import pandas as pd
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl  
import matplotlib.ticker as mticker
import seaborn as sns
from pyproj import Geod
from scipy.stats import pearsonr
import scipy.signal as signal  
from matplotlib import font_manager
import warnings

# ================= 配置 =================
CSV_PATH = "/root/data/iw_clusters_geoinfo_rgb.csv"
BATHY_NC_PATH = "/root/data/GEBCO_15_Dec_2025_6d47b43e62e6/gebco_2025_n23.689_s-1.2161_w80.4424_e128.6421.nc"
FONT_PATH = "/root/data/TIMES.TTF"

# ================= 1. 计算OBB面积 =================
def calculate_obb_area(row):
    geod = Geod(ellps="WGS84")
    _, _, d12 = geod.inv(row['OBB_P1_Lon'], row['OBB_P1_Lat'],
                         row['OBB_P2_Lon'], row['OBB_P2_Lat'])
    _, _, d23 = geod.inv(row['OBB_P2_Lon'], row['OBB_P2_Lat'],
                         row['OBB_P3_Lon'], row['OBB_P3_Lat'])
    return (d12 * d23) / 1e6

# ================= 2. 主流程 =================
def main():
    warnings.filterwarnings("ignore", category=FutureWarning)

    font_manager.fontManager.addfont(FONT_PATH)
    prop = font_manager.FontProperties(fname=FONT_PATH)
    plt.rcParams.update({
        'font.family': prop.get_name(),
        'figure.dpi':  600,
    })

    # --- 数据加载与预处理 ---
    print("Loading data...")
    df = pd.read_csv(CSV_PATH)
    df['Area_km2'] = df.apply(calculate_obb_area, axis=1)

    ds = xr.open_dataset(BATHY_NC_PATH)
    lons = xr.DataArray(df['Center_Lon'].values, dims="points")
    lats = xr.DataArray(df['Center_Lat'].values, dims="points")
    df['Depth_m'] = -ds.interp(lon=lons, lat=lats, method="linear")['elevation'].values

    # --- 核心过滤逻辑 ---
    df_scs = df[
        (df['Depth_m'] >= 50) & 
        (df['Depth_m'] <= 4500) &
        (df['Area_km2'] < 1500) &
        (df['Area_km2'] > 0) &
        (df['Center_Lon'] > 112) &
        (df['Center_Lon'] < 122)
    ].copy()

    # ================= 数据清洗：剔除1%异常极值 =================
    q_low = df_scs['Area_km2'].quantile(0.01)  
    q_high = df_scs['Area_km2'].quantile(0.99) 
    df_scs_clean = df_scs[(df_scs['Area_km2'] >= q_low) & (df_scs['Area_km2'] <= q_high)].copy()
    
    df_scs_clean['Log_Area'] = np.log10(df_scs_clean['Area_km2'])

    # ================= 统计打印 =================
    print("\n" + "="*65)
    print("【统计 1：基于清洗散点数据的分区 Pearson 相关系数】")
    zones = [
        ("Inner Shelf (<200m)", 50, 200),
        ("Shelf Break (200-500m)", 200, 500),
        ("Upper Slope (500-1000m)", 500, 1000),
        ("Lower Slope (1000-2000m)", 1000, 2000),
        ("Deep Basin (>2000m)", 2000, 4500)
    ]
    for name, d_min, d_max in zones:
        df_sub = df_scs_clean[(df_scs_clean['Depth_m'] >= d_min) & (df_scs_clean['Depth_m'] < d_max)]
        if len(df_sub) >= 10: 
            r_p, p_p = pearsonr(df_sub['Depth_m'], df_sub['Log_Area'])
            print(f"[{name:<25}] 样本量: {len(df_sub):<4} | r: {r_p:>7.4f} | p: {p_p:.4e}")
    
    bin_edges = np.arange(0, 4600, 50)
    df_scs_clean['Depth_Bin'] = pd.cut(df_scs_clean['Depth_m'], bins=bin_edges)
    bin_stats = df_scs_clean.groupby('Depth_Bin', observed=False).agg(
        Median_Log_Area=('Log_Area', 'median'), Count=('Log_Area', 'count')
    ).reset_index()
    bin_stats['Bin_Center'] = bin_stats['Depth_Bin'].apply(lambda x: x.mid if pd.notnull(x) else np.nan).astype(float)
    bin_stats = bin_stats[bin_stats['Count'] >= 3]
    r_global, p_global = pearsonr(bin_stats['Bin_Center'], bin_stats['Median_Log_Area'])
    print("-" * 65)
    print(f"[Global Binned (Macro Trend)] 桶数量: {len(bin_stats):<4} | r: {r_global:>7.4f} | p: {p_global:.4e}")
    print("="*65 + "\n")

    # ================= 可视化 =================
    sns.set_theme(style="ticks", context="paper") 
    mpl.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman'], 
        'mathtext.fontset': 'stix',        
        'axes.unicode_minus': False,
        'font.size': 34,               
        'xtick.labelsize': 34,         
        'ytick.labelsize': 34          
    })

    fig, ax = plt.subplots(figsize=(15, 12))

    # 1. 底图 Hexbin
    hb = ax.hexbin(
        df_scs_clean['Depth_m'], df_scs_clean['Area_km2'],
        gridsize=40, cmap='GnBu', mincnt=1, linewidths=0.1
    )
    cb = fig.colorbar(hb, ax=ax, shrink=0.8, pad=0.03)
    cb.set_label('Packet count', fontsize=34) 
    cb.ax.tick_params(labelsize=34)           

    # 2. 分桶计算
    depth_bins = np.linspace(50, 4500, 40) 
    bin_c, q25, q50, q75 = [], [], [], []
    for lo, hi in zip(depth_bins[:-1], depth_bins[1:]):
        sub = df_scs_clean[(df_scs_clean['Depth_m'] >= lo) & (df_scs_clean['Depth_m'] < hi)]['Area_km2']
        if len(sub) >= 5: 
            bin_c.append((lo + hi) / 2)
            q25.append(np.percentile(sub, 25))
            q50.append(np.percentile(sub, 50))
            q75.append(np.percentile(sub, 75))

    # 3. 使用滑动平均进行平滑
    q50_s = pd.Series(q50)
    q25_s = pd.Series(q25)
    q75_s = pd.Series(q75)
    q50_smooth = q50_s.rolling(window=3, center=True, min_periods=1).mean().values
    q25_smooth = q25_s.rolling(window=3, center=True, min_periods=1).mean().values
    q75_smooth = q75_s.rolling(window=3, center=True, min_periods=1).mean().values

    # 4. 绘制平滑趋势
    ax.plot(bin_c, q50_smooth, color='#C0392B', lw=4, label='Smoothed Median Trend', zorder=5) 
    ax.fill_between(bin_c, np.maximum(0, q25_smooth), q75_smooth, color='#C0392B', alpha=0.15, label='25th-75th pct', zorder=4)
    
    # 5. 截断 Y 轴
    ax.set_ylim(-20, 600) 

    # 6. 画辅助线和文字
    for d, lbl in zip([200, 500, 1000, 2000], ['Inner Shelf', 'Shelf Break', 'Upper Slope', 'Lower Slope']):
        ax.axvline(d, color='gray', lw=2, ls='--', alpha=0.4)
        ax.text(d + 50, ax.get_ylim()[1] * 0.95, lbl, rotation=90, 
                fontsize=34, color='gray', va='top', ha='left')

    # 7. 坐标轴与图例 (图例挪到左上角)
    ax.invert_xaxis()
    ax.set_xlabel('Water Depth (m)  $\leftarrow$ Shoaling Direction', fontsize=34, fontweight='bold')
    ax.set_ylabel('OBB Packet Area (km$^2$)', fontsize=34, fontweight='bold')
    
    ax.legend(loc='upper left', fontsize=28) 
    ax.grid(axis='y', linestyle=':', alpha=0.5)

    # 保存
    out = 'scs_depth_area_final.png'
    plt.savefig(out, dpi=600, bbox_inches='tight')
    print(f"Figure saved: {out}")

if __name__ == "__main__":
    main()