import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import json
import os
import xarray as xr
from scipy.ndimage import gaussian_filter
from matplotlib import font_manager
import matplotlib as mpl
# 设置图像的地理范围
lon_min, lon_max = 112.4, 121.32
lat_min, lat_max = 18.32, 23.19

FONT_PATH = "/root/data/TIMES.TTF"
font_prop = font_manager.FontProperties(fname=FONT_PATH)
font_manager.fontManager.addfont(FONT_PATH)
FONT_NAME = font_prop.get_name()

mpl.rcParams.update({
    'font.family': FONT_NAME,
    'mathtext.fontset': 'stix',
    'axes.unicode_minus': False,
    'axes.labelsize': 40,
    'xtick.labelsize': 40,
    'ytick.labelsize': 40
})

# 加载已保存的累积数据
accumulated_file = './data/accumulated_mask.npy'
metadata_file = './data/accumulated_mask_metadata.json'

if not os.path.exists(accumulated_file):
    print("错误：未找到 accumulated_mask.npy 文件！")
    print("请先运行原始代码生成累积数据。")
    exit()

print("正在加载累积数据...")
accumulated_mask = np.load(accumulated_file)
with open(metadata_file, 'r') as f:
    metadata = json.load(f)

ny, nx = metadata['shape']
num_masks = metadata['num_masks']
print(f"已加载：{num_masks} 个mask，图像尺寸：{ny} x {nx}")

# 创建经纬度网格
lons = np.linspace(lon_min, lon_max, nx)
lats = np.linspace(lat_max, lat_min, ny)
lon_grid, lat_grid = np.meshgrid(lons, lats)

# 配色方案
cmap = plt.cm.viridis

# ==================== 读取GEBCO水深数据 ====================
gebco_file = r'/root/data/GEBCO_15_Dec_2025_6d47b43e62e6/gebco_2025_n23.689_s-1.2161_w80.4424_e128.6421.nc'

print(f"\n正在读取GEBCO水深数据：{gebco_file}")
try:
    # 读取netCDF文件
    bathy_data = xr.open_dataset(gebco_file)
    
    if 'elevation' in bathy_data:
        depth_var = 'elevation'
    elif 'z' in bathy_data:
        depth_var = 'z'
    else:
        depth_var = list(bathy_data.data_vars)[0]
    
    # 获取经纬度和水深数据
    if 'lon' in bathy_data.coords:
        bathy_lon = bathy_data.lon.values
        bathy_lat = bathy_data.lat.values
    elif 'longitude' in bathy_data.coords:
        bathy_lon = bathy_data.longitude.values
        bathy_lat = bathy_data.latitude.values
    else:
        bathy_lon = bathy_data[list(bathy_data.coords)[0]].values
        bathy_lat = bathy_data[list(bathy_data.coords)[1]].values
    
    bathy_depth = bathy_data[depth_var].values
    
    # 创建经纬度网格
    bathy_lon_grid, bathy_lat_grid = np.meshgrid(bathy_lon, bathy_lat)
    
except Exception as e:
    print(f"读取数据出错：{e}")
    exit()

# ==================== 绘制图形 ====================
fig = plt.figure(figsize=(18, 12))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())

# 计算合适的vmax
vmax_percentile = np.percentile(accumulated_mask[accumulated_mask > 0], 95)
vmax_value = max(10, min(vmax_percentile, 30))

# 应用平滑
display_data = gaussian_filter(accumulated_mask.astype(float), sigma=2.0)
masked_data = np.ma.masked_where(display_data < 0.5, display_data)

# 绘制密度图 (底图)
im = ax.pcolormesh(
    lon_grid, lat_grid, masked_data,
    cmap=cmap,
    transform=ccrs.PlateCarree(),
    vmin=0, 
    vmax=vmax_value,
    shading='gouraud',
    alpha=0.88,
    zorder=1
)

# 绘制等深线 (去掉了数值标签)
depth_levels = [-4000, -3000, -2000, -1000, -500, -200, -100, -50]
cs = ax.contour(bathy_lon_grid, bathy_lat_grid, bathy_depth,
                levels=depth_levels,
                colors='#666666',
                linewidths=0.8,
                linestyles='--',
                alpha=0.6,
                transform=ccrs.PlateCarree(),
                zorder=5)

# 绘制200米等深线 (大陆架边缘)
cs_shelf = ax.contour(bathy_lon_grid, bathy_lat_grid, bathy_depth,
                      levels=[-200],
                      colors='dimgray',
                      linewidths=1.2,
                      linestyles='dashed',
                      alpha=0.7,
                      transform=ccrs.PlateCarree(),
                      zorder=5)

# 添加海岸线和陆地
ax.add_feature(cfeature.LAND.with_scale('10m'), facecolor='white', edgecolor='none', zorder=2)
ax.add_feature(cfeature.COASTLINE.with_scale('10m'), linewidth=0.8, edgecolor='black', zorder=3)

# 添加岛屿
try:
    islands = cfeature.NaturalEarthFeature(
        category='physical',
        name='minor_islands',
        scale='10m',
        edgecolor='black',
        facecolor='white',
        linewidth=0.6,
        alpha=0.7
    )
    ax.add_feature(islands, zorder=2)
except Exception as e:
    pass

# ================= 保留并调整经纬度网格标签 =================
# draw_labels=True 重新开启经纬度文字
# ================= 保留并调整经纬度网格标签 =================
gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', 
                  alpha=0.5, linestyle='--', zorder=4)
gl.top_labels = False     
gl.right_labels = False   

# 终极放大法：改用 'fontsize' 并且直接拉到 50，加入 fontweight='bold' 加粗
gl.xlabel_style = {'fontsize': 34, 'fontweight': 'bold', 'color': 'black'}  
gl.ylabel_style = {'fontsize': 34, 'fontweight': 'bold', 'color': 'black'}  

plt.tight_layout()
# 强制保存为一个新的测试文件名，确保你绝不会看错旧图
plt.savefig('scs_overlay.png', dpi=300, bbox_inches='tight', pad_inches=0.05)
print("\nImage saved: scs_overlay.png")

# 关闭数据集
bathy_data.close()

# ==================== 输出统计信息 ====================
print("\n=== Data Statistics ===")
print(f"Bathymetry range: {np.nanmin(bathy_depth):.1f}m to {np.nanmax(bathy_depth):.1f}m")
print(f"Maximum sampling density: {accumulated_mask.max()} overlaps")

# 只在控制台输出水深区域占比，图表上不再保留任何文本
depth_ranges = [
    (-50, 0, "0-50m Shallow water"),
    (-200, -50, "50-200m Continental shelf"),
    (-1000, -200, "200-1000m Continental slope"),
    (-5000, -1000, "1000m+ Deep sea")
]

print("\nArea percentage by depth zone:")
for d_min, d_max, label in depth_ranges:
    mask_depth = (bathy_depth >= d_min) & (bathy_depth < d_max)
    if mask_depth.sum() > 0:
        area_pct = mask_depth.sum() / (bathy_depth.size) * 100
        print(f"  {label}: {area_pct:.1f}%")