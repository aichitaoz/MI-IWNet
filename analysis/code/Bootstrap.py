import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

# ================= 1. 全局配置 (Configuration) =================
# 全局随机种子 (保证完全可复现)
GLOBAL_SEED = 2024 

# 数据路径
DATA_PATHS = {
    'South_China_Sea': '/home/xiaobowen/project/internal_wave_detection_project/analysis/iw_clusters_geoinfo_rgb.csv',
    'Sulu_Sea':        '/home/xiaobowen/project/internal_wave_detection_project/analysis/iw_clusters_Sulu.csv',
    'Andaman_Sea':     '/home/xiaobowen/project/internal_wave_detection_project/analysis/iw_clusters_Andaman.csv'
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
    # dx = R * (lambda - lambda0) * cos(phi0)
    # dy = R * (phi - phi0)
    d_lon = lons_rad - lon0
    d_lat = lats_rad - lat0
    
    x = R * d_lon * np.cos(lat0) 
    y = R * d_lat
    
    # 5. 鞋带公式 (Shoelace Formula)
    # Area = 0.5 * |sum(x_i * y_{i+1} - x_{i+1} * y_i)|
    area = 0.0
    n = len(x)
    for i in range(n):
        j = (i + 1) % n
        area += x[i] * y[j]
        area -= x[j] * y[i]
    
    return abs(area) / 2.0

def process_dataframe(csv_path):
    """
    读取并清洗数据。
    规则：
    1. 丢弃 NaN
    2. 丢弃 Area <= 0 (物理上无意义)
    """
    if not os.path.exists(csv_path):
        print(f"[Error] File not found: {csv_path}")
        return None
    
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"[Error] Failed to read CSV {csv_path}: {e}")
        return None
    
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
    """数据容器，确保图表引用的是同一批数据"""
    def __init__(self, mean_dist, geomean_dist, n_samples):
        self.mean_dist = mean_dist         # 算术平均的 Bootstrap 分布
        self.geomean_dist = geomean_dist   # 几何平均的 Bootstrap 分布
        self.n = n_samples

def run_bootstrap_analysis(data_array, n_boot=1000, seed=42):
    """
    执行 Bootstrap 重采样。
    同时计算 Arithmetic Mean 和 Geometric Mean 的分布。
    """
    rng = np.random.default_rng(seed) # 统一随机生成器
    n = len(data_array)
    
    # 预计算 log 数据以加速 GeoMean 计算
    # 注意：输入数据已确保 > 0
    log_data = np.log(data_array)
    
    boot_means = np.zeros(n_boot)
    boot_geomeans = np.zeros(n_boot)
    
    for i in range(n_boot):
        # 1. 生成重采样索引 (Resampling Indices)
        # 使用索引抽样比直接 choice 数据稍微快一点，且逻辑清晰
        indices = rng.integers(0, n, size=n)
        
        # 2. 计算 Arithmetic Mean (针对当前样本)
        sample_data = data_array[indices]
        boot_means[i] = np.mean(sample_data)
        
        # 3. 计算 Geometric Mean (针对同一批样本)
        # GeoMean = exp( mean( log(sample) ) )
        sample_log_data = log_data[indices]
        boot_geomeans[i] = np.exp(np.mean(sample_log_data))
        
    return BootstrapResult(boot_means, boot_geomeans, n)

def calculate_ci_metrics(boot_distribution, ci=95):
    """
    基于 Bootstrap 分布计算统计指标。
    """
    # 1. 点估计 (Point Estimate)
    # 推荐使用 Bootstrap 分布的均值作为最稳健的中心估计
    estimate = np.mean(boot_distribution)
    
    # 2. 置信区间 (Confidence Interval)
    alpha = (100 - ci) / 2
    lower = np.percentile(boot_distribution, alpha)
    upper = np.percentile(boot_distribution, 100 - alpha)
    
    # 3. 相对不确定性 (Relative Uncertainty)
    # 定义：CI 宽度 / 点估计值
    width = upper - lower
    rel_uncertainty = (width / estimate) * 100
    
    return estimate, lower, upper, rel_uncertainty

# ================= 4. 主程序 (Main Execution) =================
if __name__ == "__main__":
    # 设置绘图风格 (学术风)
    sns.set_theme(style="whitegrid", font_scale=1.1)
    
    # 容器：存储用于绘图的结果
    results_map = {} 
    # 容器：存储用于生成表格的统计数据
    stats_summary = []
    
    print("Starting Bootstrap Analysis...")
    print(f"Global Seed: {GLOBAL_SEED}")
    
    for region_name, path in DATA_PATHS.items():
        print(f"\nProcessing: {region_name}...")
        
        # 1. 数据处理
        df = process_dataframe(path)
        if df is None or len(df) < 10:
            print(f"  > Skipping: Not enough data (N={len(df) if df is not None else 0})")
            continue
            
        areas = df['Packet_Area_km2'].values
        
        # 2. Bootstrap 核心计算
        # 使用全局种子初始化，确保每次运行结果完全一致
        boot_res = run_bootstrap_analysis(areas, n_boot=1000, seed=GLOBAL_SEED)
        results_map[region_name] = boot_res
        
        # 3. 计算指标 (Mean)
        mean_est, mean_l, mean_u, mean_unc = calculate_ci_metrics(boot_res.mean_dist)
        
        # 4. 计算指标 (GeoMean)
        geo_est, geo_l, geo_u, geo_unc = calculate_ci_metrics(boot_res.geomean_dist)
        
        # 5. 记录到表格数据
        stats_summary.append({
            'Region': region_name,
            'N': boot_res.n,
            # Arithmetic Mean Stats
            'Mean_Est': mean_est,
            'Mean_CI_L': mean_l,
            'Mean_CI_U': mean_u,
            'Mean_Unc_Pct': mean_unc,
            # Geometric Mean Stats
            'GeoMean_Est': geo_est,
            'GeoMean_CI_L': geo_l,
            'GeoMean_CI_U': geo_u,
            'GeoMean_Unc_Pct': geo_unc
        })

    # ================= 5. 输出结果 (Outputs) =================
    
    # --- A. 生成统计表格 ---
    if stats_summary:
        summary_df = pd.DataFrame(stats_summary)
        # 调整列顺序
        cols = ['Region', 'N', 
                'Mean_Est', 'Mean_CI_L', 'Mean_CI_U', 'Mean_Unc_Pct',
                'GeoMean_Est', 'GeoMean_CI_L', 'GeoMean_CI_U', 'GeoMean_Unc_Pct']
        summary_df = summary_df[cols]
        
        # 保存 CSV
        csv_path = os.path.join(OUTPUT_DIR, 'bootstrap_stats_summary.csv')
        summary_df.to_csv(csv_path, index=False, float_format='%.4f')
        print(f"\n[Success] Table saved to: {csv_path}")
        print("\n=== Summary Table (Preview) ===")
        print(summary_df.to_string(float_format="%.2f"))

    # --- B. 绘图：双图对比 (Figure 1: GeoMean, Figure 2: Mean) ---
    # 我们生成两张图，分别展示稳健统计量(GeoMean)和总量统计量(Mean)
    
    metrics_to_plot = [
        ('Geometric Mean', 'geomean_dist', 'GeoMean'),
        ('Arithmetic Mean', 'mean_dist', 'Mean')
    ]
    
    for title, attr_name, short_name in metrics_to_plot:
        plt.figure(figsize=(10, 6))
        
        # 遍历区域画 KDE
        y_max_global = 0
        for i, (region_name, boot_res) in enumerate(results_map.items()):
            # 获取对应的 bootstrap 样本数组
            dist_data = getattr(boot_res, attr_name)
            
            # 计算对应的 CI (直接用之前算好的，或者这里重算也行，保证一致即可)
            # 这里重调函数只是为了获取坐标，数值完全一致
            est, low, up, _ = calculate_ci_metrics(dist_data)
            
            color = REGION_COLOR[region_name]
            
            # 绘制真实的 Bootstrap 分布 KDE
            ax = sns.kdeplot(dist_data, fill=True, alpha=0.2, color=color, 
                             label=f"{region_name} (N={boot_res.n})", linewidth=2)
            
            # 获取当前 Y 轴范围用于画线
            line_y = ax.get_ylim()[1] * (0.05 + 0.08 * i) # 错开高度
            
            # 画 CI 线段
            plt.hlines(y=line_y, xmin=low, xmax=up, color=color, linewidth=3)
            # 画点估计圆点
            plt.plot(est, line_y, 'o', color=color, markersize=8, markeredgecolor='white')
            
            # 标注文字
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