import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
import os
import glob
from matplotlib.colors import ListedColormap
from PIL import Image
from scipy.ndimage import zoom, gaussian_filter
from osgeo import gdal
import xarray as xr

# === 强制指向本地数据路径 ===
import cartopy
cartopy.config['pre_existing_data_dir'] = os.path.expanduser('~/.local/share/cartopy')
cartopy.config['data_dir'] = os.path.expanduser('~/.local/share/cartopy')

# ================= 配置 =================
sar_dir = r'/home/xiaobowen/project/internal_wave_detection_project/analysis/sar'
mask_dir = r'/home/xiaobowen/project/internal_wave_detection_project/analysis/mask'
corrected_mask_dir = r'/home/xiaobowen/project/internal_wave_detection_project/analysis/mask_corrected'
gebco_file = r'/home/xiaobowen/project/internal_wave_detection_project/analysis/GEBCO_15_Dec_2025_6d47b43e62e6/gebco_2025_n23.689_s-1.2161_w80.4424_e128.6421.nc' 
land_shp = r'/home/xiaobowen/.local/share/cartopy/shapefiles/natural_earth/physical/ne_10m_land.shp'

os.makedirs(corrected_mask_dir, exist_ok=True)

gebco_lon_min, gebco_lon_max = 117, 121.5
gebco_lat_min, gebco_lat_max = 4.5, 10
grid_res = 0.002 

# ================= 顶刊配色定义 =================
colors_low_max = [
    (0.00, '#F0F9FF'), # 背景
    (0.15, '#B3E5FC'), 
    (0.25, '#29B6F6'), # 单层
    (0.26, '#FFF59D'), # 转折
    (0.40, '#FFB74D'), # 双层
    (0.60, '#FF7043'), 
    (0.80, '#D32F2F'), 
    (1.00, '#B71C1C')  # 峰值
]
cmap_custom = mcolors.LinearSegmentedColormap.from_list("low_max_blue_fire", colors_low_max)

LAND_COLOR = '#FAFAFA'
LAND_EDGE = '#888888'
CONTOUR_COLOR = '#999999'

# ================= 辅助函数 =================
def load_gebco_data(gebco_path, lon_min, lon_max, lat_min, lat_max):
    try:
        ds = xr.open_dataset(gebco_path)
        lon_var = 'lon' if 'lon' in ds.coords else 'longitude'
        lat_var = 'lat' if 'lat' in ds.coords else 'latitude'
        elev_var = 'elevation' if 'elevation' in ds.data_vars else 'z'
        data = ds.sel({lon_var: slice(lon_min, lon_max), lat_var: slice(lat_min, lat_max)})
        lons, lats, elevation = data[lon_var].values, data[lat_var].values, data[elev_var].values
        ds.close()
        return lons, lats, elevation
    except: return None, None, None

def correct_and_save_mask(tiff_path, mask_path, output_path):
    ds = gdal.Open(tiff_path)
    meta = ds.GetMetadata()
    orbit_dir = meta.get('OrbitDirection', '').upper()
    ascending = (orbit_dir == 'ASCENDING')
    mask = (np.array(Image.open(mask_path).convert('L')) > 0).astype(np.uint8)
    if ascending:
        mask = np.rot90(np.fliplr(mask), 2)
    else:
        mask = np.fliplr(mask)
    Image.fromarray((mask * 255).astype(np.uint8)).save(output_path)
    return {'corrected_path': output_path, 'min_lon': float(meta.get('min_lon', 0)), 
            'max_lon': float(meta.get('max_lon', 0)), 'min_lat': float(meta.get('min_lat', 0)), 
            'max_lat': float(meta.get('max_lat', 0))}

# ================= 批处理与网格生成 =================
mask_info_list = []
for tiff_path in glob.glob(os.path.join(sar_dir, '*.tiff')):
    base = os.path.splitext(os.path.basename(tiff_path))[0]
    mask_p = os.path.join(mask_dir, base + '.png')
    if os.path.exists(mask_p):
        mask_info_list.append(correct_and_save_mask(tiff_path, mask_p, os.path.join(corrected_mask_dir, base + '_corrected.png')))

if not mask_info_list:
    print("未找到有效的 mask 数据！")
    exit()

all_min_lon = min(m['min_lon'] for m in mask_info_list)
all_max_lon = max(m['max_lon'] for m in mask_info_list)
all_min_lat = min(m['min_lat'] for m in mask_info_list)
all_max_lat = max(m['max_lat'] for m in mask_info_list)

n_lon, n_lat = int((all_max_lon - all_min_lon) / grid_res) + 1, int((all_max_lat - all_min_lat) / grid_res) + 1
lon_grid, lat_grid = np.meshgrid(np.linspace(all_min_lon, all_max_lon, n_lon), np.linspace(all_min_lat, all_max_lat, n_lat))
accumulated = np.zeros((n_lat, n_lon), dtype=np.int16)

for info in mask_info_list:
    mask = (np.array(Image.open(info['corrected_path']).convert('L')) > 0).astype(np.uint8)
    j0, j1 = int((info['min_lon'] - all_min_lon) / grid_res), int((info['max_lon'] - all_min_lon) / grid_res)
    i0, i1 = int((info['min_lat'] - all_min_lat) / grid_res), int((info['max_lat'] - all_min_lat) / grid_res)
    h, w = max(0, min(n_lat, i1) - max(0, i0)), max(0, min(n_lon, j1) - max(0, j0))
    if h > 0 and w > 0:
        resized = zoom(mask.astype(float), (h / mask.shape[0], w / mask.shape[1]), order=0)
        accumulated[max(0, i0):max(0, i0)+h, max(0, j0):max(0, j0)+w] += np.flipud(resized).astype(np.int16)

bathy_lons, bathy_lats, bathy_elevation = load_gebco_data(gebco_file, gebco_lon_min, gebco_lon_max, gebco_lat_min, gebco_lat_max)

# ================= 绘图部分 (强化字体) =================
fig = plt.figure(figsize=(18, 12))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([gebco_lon_min, gebco_lon_max, gebco_lat_min, gebco_lat_max])

display_data = gaussian_filter(accumulated.astype(float), sigma=0.6)
masked_data = np.ma.masked_where(display_data < 0.1, display_data)

# 核心热力图
im = ax.pcolormesh(lon_grid, lat_grid, masked_data, 
                   cmap=cmap_custom, 
                   transform=ccrs.PlateCarree(), 
                   vmin=0, vmax=5, 
                   shading='nearest', alpha=0.9, zorder=1)

# 1. 绘制等深线 (加大字号至 10)
if bathy_lons is not None:
    bathy_lon_grid, bathy_lat_grid = np.meshgrid(bathy_lons, bathy_lats)
    contours = ax.contour(bathy_lon_grid, bathy_lat_grid, bathy_elevation,
                          levels=[-4000, -2000, -1000, -500, -200], 
                          colors=CONTOUR_COLOR, 
                          linewidths=0.6, linestyles='--', alpha=0.5, transform=ccrs.PlateCarree(), zorder=5)
    ax.clabel(contours, inline=True, fontsize=16, fmt='%d m', colors=CONTOUR_COLOR)
# 2. 绘制陆地
land_feat = cfeature.ShapelyFeature(shpreader.Reader(land_shp).geometries(), ccrs.PlateCarree())
ax.add_feature(land_feat, facecolor=LAND_COLOR, edgecolor=LAND_EDGE, linewidth=0.6, zorder=3)

# 3. 坐标轴网格标签 (加大字号至 14)
gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.3, zorder=4)
gl.top_labels = False    # 关闭顶部标签
gl.right_labels = False  # 关闭右侧标签
gl.xlabel_style = {'size': 36, 'color': 'black', 'weight': 'normal'}
gl.ylabel_style = {'size': 36, 'color': 'black', 'weight': 'normal'}

# 4. 色标调整 (加大标签至 18，刻度至 14)
plt.tight_layout()
pos = ax.get_position()
# cax 位置: [左, 下, 宽, 高]
cax = fig.add_axes([pos.x1 + 0.02, pos.y0, 0.02, pos.height])
cbar = plt.colorbar(im, cax=cax, orientation='vertical')
cbar.set_label('Overlap Count', fontsize=18, fontweight='bold', labelpad=15)
cbar.ax.tick_params(labelsize=14) 
cbar.set_ticks(range(0, 6, 1))

# 保存图片
save_name = 'Sulu.png'
plt.savefig(save_name, dpi=300, bbox_inches='tight')
plt.close()

print(f"处理完成：已保存 {save_name}。字号已大幅调优。")