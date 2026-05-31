import xarray as xr
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.ndimage import gaussian_filter
import os

# ================= 配置 =================
NC_PATH = "/home/xiaobowen/project/internal_wave_detection_project/analysis/GEBCO_15_Dec_2025_6d47b43e62e6/gebco_2025_n23.689_s-1.2161_w80.4424_e128.6421.nc"
OUTPUT_DIR = "./output_plotly_3d"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DATA_PATHS = {
    'South_China_Sea': '/home/xiaobowen/project/internal_wave_detection_project/analysis/iw_clusters_geoinfo_rgb.csv',
    'Sulu_Sea':        '/home/xiaobowen/project/internal_wave_detection_project/analysis/iw_clusters_Sulu.csv',
    'Andaman_Sea':     '/home/xiaobowen/project/internal_wave_detection_project/analysis/iw_clusters_Andaman.csv'
}

REGIONS = {
    'South_China_Sea': [18.32, 23.19, 112.4, 121.32],    
    'Sulu_Sea':        [4.5, 15, 117, 125.5], 
    'Andaman_Sea':     [5, 10, 90, 100]        
}

# 配色
REGION_COLOR = {
    'South_China_Sea': 'rgb(230, 75, 53)',
    'Sulu_Sea':        'rgb(64, 224, 208)',
    'Andaman_Sea':     'rgb(255, 215, 0)'
}

# ================= 优化海底地形配色方案 =================
BATHYMETRY_COLORSCALE = [
    [0.0,   '#1a1a4d'],   # 深蓝 (深海 < -5000m)
    [0.15,  '#0047ab'],   # 蓝 (-5000m)
    [0.30,  '#0066cc'],   # 浅蓝 (-2000m)
    [0.45,  '#4da6ff'],   # 浅蓝 (-500m)
    [0.60,  '#99ccff'],   # 浅蓝 (-100m)
    [0.70,  '#e6d9a8'],   # 沙色 (-50m)
    [0.80,  '#c4b896'],   # 沙棕 (0m)
    [0.90,  '#8b7355'],   # 棕色 (高地)
    [1.0,   '#5c4033']    # 深棕 (陆地)
]

# ================= 统一的相机配置 (斜45度等距视图/立方体画法) =================
isometric_camera = dict(
    eye=dict(x=0, y=-0.95, z=0.95),     
    center=dict(x=0, y=0, z=0),
    up=dict(x=0, y=0, z=1)  # 保持 Z 轴朝上
)

CAMERA_CONFIG = {
    'South_China_Sea': dict(
    eye=dict(x=0, y=-0.95, z=0.95),     
    center=dict(x=0, y=0, z=0),
    up=dict(x=0, y=0, z=1)  
),
    'Sulu_Sea':        isometric_camera,
    'Andaman_Sea':     dict(
    eye=dict(x=0, y=-0.95, z=0.95),     
    center=dict(x=0, y=0, z=0),
    up=dict(x=0, y=0, z=1)  
),
}

# ================= 注入 HTML 的相机信息面板 JS/CSS =================
CAMERA_PANEL_INJECTION = """
<style>
  #camera-panel {
    position: fixed;
    top: 16px;
    right: 16px;
    background: rgba(255, 255, 255, 0.93);
    border: 1px solid #bbb;
    border-radius: 8px;
    padding: 12px 16px;
    font-family: 'Courier New', monospace;
    font-size: 13px;
    color: #222;
    z-index: 9999;
    min-width: 260px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.18);
    line-height: 1.9;
    user-select: text;
  }
  #camera-panel h4 {
    margin: 0 0 6px 0;
    font-size: 14px;
    font-family: 'Times New Roman', serif;
    color: #333;
    border-bottom: 1px solid #ddd;
    padding-bottom: 4px;
  }
  .cp-row { display: flex; justify-content: space-between; gap: 12px; }
  .cp-label { color: #666; }
  .cp-val   { font-weight: bold; color: #1a1aff; min-width: 70px; text-align: right; }
  .cp-zoom  { font-weight: bold; color: #cc0000; min-width: 70px; text-align: right; }
  #cp-copy  {
    margin-top: 8px; width: 100%; padding: 4px 0;
    background: #f0f0f0; border: 1px solid #ccc; border-radius: 4px;
    cursor: pointer; font-family: 'Courier New', monospace; font-size: 12px;
  }
  #cp-copy:hover { background: #ddd; }
</style>

<div id="camera-panel">
  <h4>📷 Camera Info</h4>
  <div class="cp-row"><span class="cp-label">eye.x</span><span class="cp-val"  id="cp-ex">—</span></div>
  <div class="cp-row"><span class="cp-label">eye.y</span><span class="cp-val"  id="cp-ey">—</span></div>
  <div class="cp-row"><span class="cp-label">eye.z</span><span class="cp-val"  id="cp-ez">—</span></div>
  <div class="cp-row"><span class="cp-label">center.x</span><span class="cp-val" id="cp-cx">—</span></div>
  <div class="cp-row"><span class="cp-label">center.y</span><span class="cp-val" id="cp-cy">—</span></div>
  <div class="cp-row"><span class="cp-label">center.z</span><span class="cp-val" id="cp-cz">—</span></div>
  <div class="cp-row"><span class="cp-label">up.x</span><span class="cp-val" id="cp-ux">—</span></div>
  <div class="cp-row"><span class="cp-label">up.y</span><span class="cp-val" id="cp-uy">—</span></div>
  <div class="cp-row"><span class="cp-label">up.z</span><span class="cp-val" id="cp-uz">—</span></div>
  <div class="cp-row" style="margin-top:6px; border-top:1px solid #ddd; padding-top:6px;">
    <span class="cp-label">🔍 zoom (‖eye‖)</span>
    <span class="cp-zoom" id="cp-zoom">—</span>
  </div>
  <button id="cp-copy" onclick="copyCamera()">📋 Copy as Python dict</button>
</div>

<script>
(function() {
  function attachListener() {
    var gd = document.querySelector('.plotly-graph-div');
    if (!gd) { setTimeout(attachListener, 300); return; }
    function fmt(v) { return (v !== undefined && v !== null) ? v.toFixed(4) : '—'; }
    function updatePanel(camera) {
      var eye    = camera.eye    || {};
      var center = camera.center || {};
      var up     = camera.up     || {};
      document.getElementById('cp-ex').textContent = fmt(eye.x);
      document.getElementById('cp-ey').textContent = fmt(eye.y);
      document.getElementById('cp-ez').textContent = fmt(eye.z);
      document.getElementById('cp-cx').textContent = fmt(center.x);
      document.getElementById('cp-cy').textContent = fmt(center.y);
      document.getElementById('cp-cz').textContent = fmt(center.z);
      document.getElementById('cp-ux').textContent = fmt(up.x);
      document.getElementById('cp-uy').textContent = fmt(up.y);
      document.getElementById('cp-uz').textContent = fmt(up.z);
      var norm = Math.sqrt((eye.x||0)*(eye.x||0) + (eye.y||0)*(eye.y||0) + (eye.z||0)*(eye.z||0));
      document.getElementById('cp-zoom').textContent = norm.toFixed(4);
    }
    var initCamera = gd._fullLayout && gd._fullLayout.scene && gd._fullLayout.scene.camera;
    if (initCamera) updatePanel(initCamera);
    gd.on('plotly_relayout', function(eventdata) {
      var cam = gd._fullLayout.scene.camera;
      if (cam) updatePanel(cam);
    });
  }
  window.copyCamera = function() {
    var txt = "camera = dict(\\neye=dict(x=" + document.getElementById('cp-ex').textContent + ", y=" + document.getElementById('cp-ey').textContent + ", z=" + document.getElementById('cp-ez').textContent + "),\\ncenter=dict(x=" + document.getElementById('cp-cx').textContent + ", y=" + document.getElementById('cp-cy').textContent + ", z=" + document.getElementById('cp-cz').textContent + "),\\nup=dict(x=" + document.getElementById('cp-ux').textContent + ", y=" + document.getElementById('cp-uy').textContent + ", z=" + document.getElementById('cp-uz').textContent + ")\\n)\\n# zoom = " + document.getElementById('cp-zoom').textContent;
    navigator.clipboard.writeText(txt).then(function() {
      var btn = document.getElementById('cp-copy'); btn.textContent = '✅ Copied!';
      setTimeout(function(){ btn.textContent = '📋 Copy as Python dict'; }, 1500);
    });
  };
  attachListener();
})();
</script>
"""

def plot_with_plotly(region_name, bounds):
    print(f"正在处理: {region_name} ...")
    lat_min, lat_max, lon_min, lon_max = bounds

    # --- 1. 读取地形 ---
    try:
        ds = xr.open_dataset(NC_PATH)
        pad = 0.1
        subset = ds.sel(lat=slice(lat_min-pad, lat_max+pad), lon=slice(lon_min-pad, lon_max+pad))
        scale = 800 
        step_lat = max(1, int(subset.dims['lat'] / scale))
        step_lon = max(1, int(subset.dims['lon'] / scale))
        subset = subset.isel(lat=slice(None, None, step_lat), lon=slice(None, None, step_lon))
        z = subset['elevation'].values
        lon = subset['lon'].values
        lat = subset['lat'].values
        z_smooth = gaussian_filter(z, sigma=1.0)
    except Exception as e:
        print(f"地形读取失败: {e}"); return

    # --- 2. 读取 CSV ---
    csv_path = DATA_PATHS.get(region_name)
    pts_lon, pts_lat = [], []
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        mask = (df['Center_Lat'] >= lat_min) & (df['Center_Lat'] <= lat_max) & \
               (df['Center_Lon'] >= lon_min) & (df['Center_Lon'] <= lon_max)
        pts_lon, pts_lat = df[mask]['Center_Lon'], df[mask]['Center_Lat']
        print(f"  > 找到 {len(pts_lon)} 个内波点")

    # --- 3. Plotly 绘图 ---
    fig = go.Figure()
    fig.add_trace(go.Surface(
        z=z_smooth, x=lon, y=lat, 
        colorscale=BATHYMETRY_COLORSCALE,
        cmin=-6000, cmax=2000,
        lighting=dict(roughness=0.5, fresnel=0.5, ambient=0.4), 
        showscale=False
    ))

    if len(pts_lon) > 0:
        fig.add_trace(go.Scatter3d(
            x=pts_lon, y=pts_lat, z=np.zeros(len(pts_lon)) + 10, mode='markers',
            marker=dict(size=3, color=REGION_COLOR[region_name], opacity=0.9, line=dict(width=0))
        ))

    xx, yy = np.meshgrid(np.linspace(lon_min, lon_max, 2), np.linspace(lat_min, lat_max, 2))
    fig.add_trace(go.Surface(z=np.zeros_like(xx), x=xx, y=yy, colorscale=[[0, 'cyan'], [1, 'cyan']], opacity=0.1, showscale=False))

    # --- 布局与相机 ---
    lon_range, lat_range = lon_max - lon_min, lat_max - lat_min
    max_range = max(lon_range, lat_range)
    ratio_x, ratio_y, ratio_z = lon_range/max_range, lat_range/max_range, 0.3
    camera = CAMERA_CONFIG.get(region_name) or dict(eye=dict(x=-1.5*ratio_x, y=-1.5*ratio_y, z=1.5))

    # 【统一公共边框样式】：彻底关闭所有显示，去掉外边框
    common_axis_style = dict(
        visible=False  # <--- 关键修改：一键隐藏整个坐标轴及其线框
    )

    fig.update_layout(
        font=dict(family="Times New Roman", size=18, color="black"),
        scene=dict(
            xaxis=dict(**common_axis_style),
            yaxis=dict(**common_axis_style),
            zaxis=dict(range=[np.min(z_smooth), 500], **common_axis_style),
            
            aspectmode='manual',
            aspectratio=dict(x=ratio_x, y=ratio_y, z=ratio_z),
            camera=camera,
            dragmode='orbit' 
        ),
        margin=dict(l=0, r=0, b=0, t=0), 
        autosize=True,
        width=1400,
        height=900,
        # =================【透明背景】=================
        paper_bgcolor='rgba(0,0,0,0)', 
        plot_bgcolor='rgba(0,0,0,0)'
    )

    # --- 5. 保存 HTML ---
    html_path = os.path.join(OUTPUT_DIR, f"{region_name}.html")
    raw_html = fig.to_html(full_html=True, include_plotlyjs='cdn').replace('</body>', CAMERA_PANEL_INJECTION + '\n</body>')
    with open(html_path, 'w', encoding='utf-8') as f: f.write(raw_html)
    print(f"✅ HTML 生成: {html_path}")

    # --- 6. 保存 PNG/PDF ---
    img_path = os.path.join(OUTPUT_DIR, f"{region_name}.png")
    try:
        fig.write_image(img_path, format='png', scale=2, width=1400, height=900)
        print(f"✅ 图片 生成: {img_path}")
    except Exception as e:
        print(f"⚠ 图片 保存失败: {e}")

if __name__ == "__main__":
    for name, bounds in REGIONS.items():
        plot_with_plotly(name, bounds)