import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

# ================= 1. 全局配置 (Configuration) =================
# 全局随机种子 (保证完全可复现)
GLOBAL_SEED = 2024 

# 更新后的数据配置结构：支持全量读取与经纬度筛选
DATA_CONFIG = {
    'South_China_Sea': {
        'path': '/root/data/iw_clusters_geoinfo_rgb.csv',
        'bbox': None  # 不筛选，读取全部
    },
    'Sulu_Sea': {
        'path': '/root/data/iw_clusters_geoinfo_sar.csv',
        'bbox': dict(lon_min=117.0, lon_max=123.0, lat_min=5.0, lat_max=12.0)
    },
    'Andaman_Sea': {
        'path': '/root/data/iw_clusters_geoinfo_sar.csv',
        'bbox': dict(lon_min=90.0, lon_max=100.0, lat_min=5.0, lat_max=10.0)
    }
}

# 绘图配色
REGION_COLOR = {
    'South_China_Sea': '#E64B35', # 红
    'Sulu_Sea':        '#3C5488', # 蓝
    'Andaman_Sea':     '#FFD700'  # 黄
}

# 输出路径
OUTPUT_DIR = './output_bootstrap_analysis'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================= 2. 物理计算模块 (Physics Kernel) =================
def calculate_polygon_area_km2_relative(lon_list, lat_list):
    """
    计算多边形面积 (km^2)。
    【核心修正】使用质心相对坐标投影 (Relative Coordinate Projection)，
    彻底解决直接投影导致的 'Catastrophic Cancellation' 数值精度问题。
    """
    # 1. 强制类型转换，确保数值计算
    lons = np.array(lon_list, dtype=float)
    lats = np.array(lat_list, dtype=float)
    
    # 地球平均半径 R = 6371 km
    R = 6371.0
    
    # 2. 转换为弧度
    lons_rad = np.radians(lons)
    lats_rad = np.radians(lats)
    
    # 3. 计算质心 (Centroid) 作为局部投影原点
    lon0 = np.mean(lons_rad)
    lat0 = np.mean(lats_rad)
    
    # 4. 计算相对坐标 (Delta)
    # 使用等距圆柱投影近似 (Equirectangular approximation)
    d_lon = lons_rad - lon0
    d_lat = lats_rad - lat0
    
    x = R * d_lon * np.cos(lat0) 
    y = R * d_lat
    
    # 5. 鞋带公式 (Shoelace Formula)
    area = 0.0
    n = len(x)
    for i in range(n):
        j = (i + 1) % n
        area += x[i] * y[j]
        area -= x[j] * y[i]
    
    return abs(area) / 2.0

def process_dataframe(csv_path, bbox=None):
    """
    读取并清洗数据。支持基于 bbox 的经纬度筛选。
    """
    if not os.path.exists(csv_path):
        print(f"[Error] File not found: {csv_path}")
        return None
    
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"[Error] Failed to read CSV {csv_path}: {e}")
        return None
    
    # 新增逻辑：如果提供了包围盒 bbox，先通过 Center_Lon 和 Center_Lat 筛选数据
    if bbox is not None:
        if 'Center_Lon' not in df.columns or 'Center_Lat' not in df.columns:
            print(f"[Warning] Missing Center_Lon/Center_Lat in {csv_path} for bbox filtering.")
            return None
        
        # 确保中心坐标为数值型
        df['Center_Lon'] = pd.to_numeric(df['Center_Lon'], errors='coerce')
        df['Center_Lat'] = pd.to_numeric(df['Center_Lat'], errors='coerce')
        
        # 筛选经纬度区间
        df = df[
            (df['Center_Lon'] >= bbox['lon_min']) &
            (df['Center_Lon'] <= bbox['lon_max']) &
            (df['Center_Lat'] >= bbox['lat_min']) &
            (df['Center_Lat'] <= bbox['lat_max'])
        ].copy()
        
    cols_lon = ['OBB_P1_Lon', 'OBB_P2_Lon', 'OBB_P3_Lon', 'OBB_P4_Lon']
    cols_lat = ['OBB_P1_Lat', 'OBB_P2_Lat', 'OBB_P3_Lat', 'OBB_P4_Lat']
    target_cols = cols_lon + cols_lat
    
    # 完整性检查
    if not all(col in df.columns for col in target_cols):
        print(f"[Warning] Missing coordinate columns in {csv_path}")
        return None

    # 类型转换与 NaN 清洗
    for col in target_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    initial_len = len(df)
    df = df.dropna(subset=target_cols)
    
    # 计算面积
    areas = []
    for _, row in df.iterrows():
        lons = row[cols_lon].values
        lats = row[cols_lat].values
        area = calculate_polygon_area_km2_relative(lons, lats)
        areas.append(area)
    
    df['Packet_Area_km2'] = areas
    
    # 【固定规则】过滤无效面积
    df_clean = df[df['Packet_Area_km2'] > 0].copy()
    
    if len(df_clean) < initial_len:
        print(f"  > Cleaned {initial_len - len(df_clean)} rows (NaN or Area<=0)")
        
    return df_clean

# ================= 3. 统计分析模块 (Statistical Kernel) =================
class BootstrapResult:
    def __init__(self, mean_dist, geomean_dist, n_samples):
        self.mean_dist = mean_dist         # 算术平均的 Bootstrap 分布
        self.geomean_dist = geomean_dist   # 几何平均的 Bootstrap 分布
        self.n = n_samples

def run_bootstrap_analysis(data_array, n_boot=1000, seed=42):
    rng = np.random.default_rng(seed) 
    n = len(data_array)
    log_data = np.log(data_array)
    
    boot_means = np.zeros(n_boot)
    boot_geomeans = np.zeros(n_boot)
    
    for i in range(n_boot):
        indices = rng.integers(0, n, size=n)
        sample_data = data_array[indices]
        boot_means[i] = np.mean(sample_data)
        
        sample_log_data = log_data[indices]
        boot_geomeans[i] = np.exp(np.mean(sample_log_data))
        
    return BootstrapResult(boot_means, boot_geomeans, n)

def calculate_ci_metrics(boot_distribution, ci=95):
    estimate = np.mean(boot_distribution)
    alpha = (100 - ci) / 2
    lower = np.percentile(boot_distribution, alpha)
    upper = np.percentile(boot_distribution, 100 - alpha)
    width = upper - lower
    rel_uncertainty = (width / estimate) * 100
    return estimate, lower, upper, rel_uncertainty

# ================= 4. 主程序 (Main Execution) =================
if __name__ == "__main__":
    sns.set_theme(style="whitegrid", font_scale=1.1)
    
    results_map = {} 
    stats_summary = []
    
    print("Starting Bootstrap Analysis...")
    print(f"Global Seed: {GLOBAL_SEED}")
    
    for region_name, config in DATA_CONFIG.items():
        print(f"\nProcessing: {region_name}...")
        
        # 1. 数据处理 (传入路径和可选的 bbox)
        df = process_dataframe(config['path'], config['bbox'])
        if df is None or len(df) < 10:
            print(f"  > Skipping: Not enough data (N={len(df) if df is not None else 0})")
            continue
            
        areas = df['Packet_Area_km2'].values
        
        # 2. Bootstrap 核心计算
        boot_res = run_bootstrap_analysis(areas, n_boot=1000, seed=GLOBAL_SEED)
        results_map[region_name] = boot_res
        
        # 3. 计算指标
        mean_est, mean_l, mean_u, mean_unc = calculate_ci_metrics(boot_res.mean_dist)
        geo_est, geo_l, geo_u, geo_unc = calculate_ci_metrics(boot_res.geomean_dist)
        
        # 4. 记录数据
        stats_summary.append({
            'Region': region_name,
            'N': boot_res.n,
            'Mean_Est': mean_est, 'Mean_CI_L': mean_l, 'Mean_CI_U': mean_u, 'Mean_Unc_Pct': mean_unc,
            'GeoMean_Est': geo_est, 'GeoMean_CI_L': geo_l, 'GeoMean_CI_U': geo_u, 'GeoMean_Unc_Pct': geo_unc
        })

    # ================= 5. 输出结果 (Outputs) =================
    if stats_summary:
        summary_df = pd.DataFrame(stats_summary)
        cols = ['Region', 'N', 
                'Mean_Est', 'Mean_CI_L', 'Mean_CI_U', 'Mean_Unc_Pct',
                'GeoMean_Est', 'GeoMean_CI_L', 'GeoMean_CI_U', 'GeoMean_Unc_Pct']
        summary_df = summary_df[cols]
        
        csv_path = os.path.join(OUTPUT_DIR, 'bootstrap_stats_summary.csv')
        summary_df.to_csv(csv_path, index=False, float_format='%.4f')
        print(f"\n[Success] Table saved to: {csv_path}")
        print("\n=== Summary Table (Preview) ===")
        print(summary_df.to_string(float_format="%.2f"))

    metrics_to_plot = [
        ('Geometric Mean', 'geomean_dist', 'GeoMean'),
        ('Arithmetic Mean', 'mean_dist', 'Mean')
    ]
    
    for title, attr_name, short_name in metrics_to_plot:
        plt.figure(figsize=(10, 6))
        
        for i, (region_name, boot_res) in enumerate(results_map.items()):
            dist_data = getattr(boot_res, attr_name)
            est, low, up, _ = calculate_ci_metrics(dist_data)
            color = REGION_COLOR[region_name]
            
            ax = sns.kdeplot(dist_data, fill=True, alpha=0.2, color=color, 
                           label=f"{region_name} (N={boot_res.n})", linewidth=2)
            
            line_y = ax.get_ylim()[1] * (0.05 + 0.08 * i) 
            
            plt.hlines(y=line_y, xmin=low, xmax=up, color=color, linewidth=3)
            plt.plot(est, line_y, 'o', color=color, markersize=8, markeredgecolor='white')
            
            width_val = up - low
            plt.text(up, line_y, f" CI Width: {width_val:.1f}", 
                     color=color, va='center', fontsize=9, fontweight='bold')

        plt.title(f'Bootstrap Stability Analysis: {title}', fontsize=14, fontweight='bold')
        plt.xlabel(f'{title} Packet Area ($km^2$)', fontsize=12)
        plt.ylabel('Probability Density', fontsize=12)
        plt.legend(loc='upper right')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        
        save_path = os.path.join(OUTPUT_DIR, f'bootstrap_stability_{short_name}.png')
        plt.savefig(save_path, dpi=300)
        print(f"[Success] Plot saved to: {save_path}")

    print("\nDone.")