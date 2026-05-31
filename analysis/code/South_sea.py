import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
import json
import os
import xarray as xr
from scipy.ndimage import gaussian_filter

# ================= 配置与路径 =================
land_shp = r'/home/xiaobowen/.local/share/cartopy/shapefiles/natural_earth/physical/ne_10m_land.shp'
gebco_file = r'./GEBCO_14_Dec_2025_b048ba5b0116/gebco_2025_n23.19_s18.32_w112.4_e121.32.nc'
accumulated_file = 'accumulated_mask.npy'
metadata_file = 'accumulated_mask_metadata.json'

lon_min, lon_max = 112.4, 121.32
lat_min, lat_max = 18.32, 23.19

# ================= 最终配色 =================
colors_faint_short_blue = [
    (0.00, '#F0F9FF'), (0.08, '#B3E5FC'), (0.16, '#29B6F6'),
    (0.17, '#FFF59D'), (0.30, '#FFB74D'), (0.50, '#FF7043'),
    (0.75, '#D32F2F'), (1.00, '#B71C1C')
]
cmap_final = mcolors.LinearSegmentedColormap.from_list("faint_short_blue", colors_faint_short_blue)

LAND_COLOR = '#F5F5F5'
LAND_EDGE = '#888888'
CONTOUR_COLOR = '#999999'

# ================= 数据加载 =================
if not os.path.exists(accumulated_file):
    print("错误：未找到数据文件！")
    exit()

accumulated_mask = np.load(accumulated_file)
with open(metadata_file, 'r') as f:
    metadata = json.load(f)

ny, nx = metadata['shape']
lons = np.linspace(lon_min, lon_max, nx)
lats = np.linspace(lat_max, lat_min, ny)
lon_grid, lat_grid = np.meshgrid(lons, lats)

try:
    ds = xr.open_dataset(gebco_file)
    lon_var = 'lon' if 'lon' in ds.coords else 'longitude'
    lat_var = 'lat' if 'lat' in ds.coords else 'latitude'
    elev_var = 'elevation' if 'elevation' in ds.data_vars else 'z'
    bathy_lon, bathy_lat = ds[lon_var].values, ds[lat_var].values
    bathy_elevation = ds[elev_var].values
    bathy_lon_grid, bathy_lat_grid = np.meshgrid(bathy_lon, bathy_lat)
    ds.close()
except:
    bathy_elevation = None

# ================= 绘图部分 =================
fig = plt.figure(figsize=(20, 14))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([lon_min, lon_max, lat_min, lat_max])

display_data = gaussian_filter(accumulated_mask.astype(float), sigma=1.0)
masked_data = np.ma.masked_where(display_data < 0.1, display_data)

im = ax.pcolormesh(lon_grid, lat_grid, masked_data, 
                   cmap=cmap_final, transform=ccrs.PlateCarree(), 
                   vmin=0, vmax=30, shading='auto', alpha=0.95, zorder=1)

# 1. 修正后的等深线标注 (移除报错参数)
if bathy_elevation is not None:
    contours = ax.contour(bathy_lon_grid, bathy_lat_grid, bathy_elevation,
                          levels=[-4000, -2000, -1000, -500, -200], 
                          colors=CONTOUR_COLOR, linewidths=0.8, 
                          linestyles='--', alpha=0.5, transform=ccrs.PlateCarree(), zorder=5)
    # 彻底修复：只保留基础支持的参数
    ax.clabel(contours, inline=True, fontsize=14, fmt='%d m', colors=CONTOUR_COLOR)

# 2. 陆地
if os.path.exists(land_shp):
    land_feat = cfeature.ShapelyFeature(shpreader.Reader(land_shp).geometries(), ccrs.PlateCarree())
    ax.add_feature(land_feat, facecolor=LAND_COLOR, edgecolor=LAND_EDGE, linewidth=0.7, zorder=3)
else:
    ax.add_feature(cfeature.LAND.with_scale('10m'), facecolor=LAND_COLOR, edgecolor=LAND_EDGE, zorder=3)

# 3. 坐标轴 22 号字
gl = ax.gridlines(draw_labels=True, linewidth=0.6, color='gray', alpha=0.3, zorder=4)
gl.top_labels = False 
gl.right_labels = False 
gl.xlabel_style = {'size': 42, 'color': 'black', 'weight': 'normal'}
gl.ylabel_style = {'size': 42, 'color': 'black', 'weight': 'normal'}

# 4. 布局与色标
plt.tight_layout()
pos = ax.get_position()
cax = fig.add_axes([pos.x1 + 0.06, pos.y0, 0.015, pos.height]) # 增加间距到 0.06

cbar = plt.colorbar(im, cax=cax, orientation='vertical')
cbar.set_label('Overlap Count', fontsize=22, fontweight='bold', labelpad=20)
cbar.ax.tick_params(labelsize=18)

save_name = 'South_sea_Fixed_Font22.png'
plt.savefig(save_name, dpi=300, bbox_inches='tight')
plt.close()

print(f"成功修复！图片已保存为 {save_name}，经纬度字号 22。")