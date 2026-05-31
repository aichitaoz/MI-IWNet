import pandas as pd
import numpy as np
import xarray as xr
import plotly.graph_objects as go
import math

# ================= 1. 核心配置 =================
CSV_PATH = "./data/iw_clusters_geoinfo_sar_new.csv"
GEBCO_PATH = "/home/xiaobowen/project/internal_wave_detection_project/analysis/GEBCO_15_Dec_2025_6d47b43e62e6/gebco_2025_n23.689_s-1.2161_w80.4424_e128.6421.nc"
OUTPUT_HTML = "Sulu_Sea_3D_Tracing.html"

# 苏禄海边界
MIN_LON, MAX_LON = 117.0, 122.5
MIN_LAT, MAX_LAT = 5.0, 11.5

MIN_STRIPES = 3          # 降噪阈值
RAY_LENGTH_KM = 300      # 射线溯源长度
STRIDE = 4               # 🚀 地形降采样率 (数值越大渲染越快，但地形越粗糙。推荐 4-8)

# ================= 2. 数学引擎 =================
def get_destination(lat, lon, bearing, dist_km):
    """大圆航线坐标推算"""
    R = 6371.0 
    lat1, lon1, brg = map(math.radians, [lat, lon, bearing])
    lat2 = math.asin(math.sin(lat1)*math.cos(dist_km/R) + 
                     math.cos(lat1)*math.sin(dist_km/R)*math.cos(brg))
    lon2 = lon1 + math.atan2(math.sin(brg)*math.sin(dist_km/R)*math.cos(lat1), 
                             math.cos(dist_km/R)-math.sin(lat1)*math.sin(lat2))
    return math.degrees(lat2), math.degrees(lon2)

# ================= 3. 主处理与 3D 渲染 =================
def main():
    print("⏳ [1/4] 加载内波 CSV 数据并过滤...")
    df = pd.read_csv(CSV_PATH)
    df = df[
        (df['Center_Lon'] >= MIN_LON) & (df['Center_Lon'] <= MAX_LON) &
        (df['Center_Lat'] >= MIN_LAT) & (df['Center_Lat'] <= MAX_LAT) &
        (df['StripeCount'] >= MIN_STRIPES)
    ]
    
    print("⏳ [2/4] 读取并裁剪 GEBCO 海底地形...")
    ds = xr.open_dataset(GEBCO_PATH)
    bathy = ds['elevation'].sel(lon=slice(MIN_LON, MAX_LON), lat=slice(MIN_LAT, MAX_LAT))
    
    # 关键：降采样以防浏览器崩溃
    bathy_downsampled = bathy[::STRIDE, ::STRIDE]
    X_lon = bathy_downsampled.lon.values
    Y_lat = bathy_downsampled.lat.values
    Z_depth = bathy_downsampled.values

    print("⏳ [3/4] 构建 3D 图层...")
    fig = go.Figure()

    # ---- 图层 1: 海底地形 3D 曲面 ----
    fig.add_trace(go.Surface(
        x=X_lon, y=Y_lat, z=Z_depth,
        colorscale='Viridis', # 颜色映射：深海偏紫，浅滩偏黄
        cmin=-5000, cmax=100,
        colorbar=dict(title='Elevation / Depth (m)', len=0.7),
        opacity=0.95,
        name='Bathymetry'
    ))

    # ---- 图层 2: 半透明海平面 (Z=0) ----
    fig.add_trace(go.Surface(
        x=[MIN_LON, MAX_LON],
        y=[MIN_LAT, MAX_LAT],
        z=[[0, 0], [0, 0]],
        colorscale=[[0, 'cyan'], [1, 'cyan']],
        showscale=False,
        opacity=0.15,
        hoverinfo='skip',
        name='Sea Surface'
    ))

    # ---- 准备射线数据 ----
    ray_x, ray_y, ray_z = [], [], []
    
    for idx, row in df.iterrows():
        lat, lon = row['Center_Lat'], row['Center_Lon']
        count = row['StripeCount']
        
        # 修复坐标系并反演
        map_prop_dir = (90.0 - row['Propagation_Dir']) % 360.0
        reverse_dir = (map_prop_dir + 180.0) % 360.0
        end_lat, end_lon = get_destination(lat, lon, reverse_dir, RAY_LENGTH_KM)

        # 优化：通过插入 None 将所有线段合并为一条轨迹，极大提升渲染速度
        ray_x.extend([lon, end_lon, None])
        ray_y.extend([lat, end_lat, None])
        ray_z.extend([0, 0, None]) # 射线紧贴海平面 Z=0

    # ---- 图层 3: 反向溯源射线 (画在海平面上) ----
    fig.add_trace(go.Scatter3d(
        x=ray_x, y=ray_y, z=ray_z,
        mode='lines',
        line=dict(color='red', width=2, dash='dash'),
        opacity=0.6,
        name='Reverse Rays',
        hoverinfo='none'
    ))

    # ---- 图层 4: 内波发生点 ----
    fig.add_trace(go.Scatter3d(
        x=df['Center_Lon'], y=df['Center_Lat'], z=np.zeros(len(df)),
        mode='markers',
        marker=dict(
            size=df['StripeCount'] * 1.2, # 圆球大小随条纹数缩放
            color='#FFD700',
            line=dict(color='black', width=1),
            opacity=0.9
        ),
        text=df.apply(lambda r: f"文件: {r['FileName']}<br>条纹数: {r['StripeCount']}", axis=1),
        hoverinfo='text',
        name='Internal Wave Packets'
    ))

    print("⏳ [4/4] 调整相机与光影...")
    # 设置 3D 比例：经纬度比例 1:1，把深度 (Z轴) 压缩一下，否则地形看起来像针一样尖锐
    fig.update_layout(
        title='Sulu Sea Internal Wave 3D Source Tracing',
        scene=dict(
            xaxis_title='Longitude',
            yaxis_title='Latitude',
            zaxis_title='Depth (m)',
            aspectmode='manual',
            aspectratio=dict(x=1, y=(MAX_LAT-MIN_LAT)/(MAX_LON-MIN_LON), z=0.15),
            camera=dict(
                eye=dict(x=1.5, y=-1.5, z=1.0) # 默认视角：斜向下俯视
            )
        ),
        margin=dict(l=0, r=0, b=0, t=50)
    )

    # 导出为独立网页
    fig.write_html(OUTPUT_HTML)
    print(f"\n🎉 搞定！3D 地形与射线已融合，请下载并用浏览器打开: {OUTPUT_HTML}")
    print("👉 提示：你可以用鼠标【左键旋转】、【右键平移】、【滚轮缩放】进行 3D 探索！")

if __name__ == "__main__":
    main()