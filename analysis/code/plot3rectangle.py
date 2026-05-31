import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl  # 引入 mpl 以配置字体
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
import xarray as xr
from matplotlib.colors import LinearSegmentedColormap
import os

# ================= 字体配置 (关键修改) =================
# 尝试设置 Times New Roman
# 如果你在服务器上且有 TIMES.TTF 文件，Matplotlib 会自动优先寻找系统中的新罗马
mpl.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'], # 优先使用新罗马
    'mathtext.fontset': 'stix',        # 数学公式也用类似新罗马的字体
    'axes.unicode_minus': False,
    'axes.labelsize': 12,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12
})

# ================= 配置 =================
gebco_file = r'./GEBCO_15_Dec_2025_6d47b43e62e6/gebco_2025_n23.689_s-1.2161_w80.4424_e128.6421.nc'
land_shp_path = r'/home/xiaobowen/.local/share/cartopy/shapefiles/natural_earth/physical/ne_10m_land.shp'

plot_extent = [88, 128.6421, 3, 23.689] 

# ================= 读取数据 =================
def load_gebco_data(gebco_path, extent, buffer=0.5):
    lon_min, lon_max, lat_min, lat_max = extent
    try:
        ds = xr.open_dataset(gebco_path)
        lon_var = 'lon' if 'lon' in ds.coords else list(ds.coords.keys())[0]
        lat_var = 'lat' if 'lat' in ds.coords else list(ds.coords.keys())[1]
        elev_var = 'elevation' if 'elevation' in ds.data_vars else list(ds.data_vars.keys())[0]
        
        # 读取范围稍大一点，防止边缘白边
        data = ds.sel({
            lon_var: slice(lon_min - buffer, lon_max + buffer), 
            lat_var: slice(lat_min - buffer, lat_max + buffer)
        })
        
        lons, lats, elevation = data[lon_var].values, data[lat_var].values, data[elev_var].values
        ds.close()
        return lons, lats, elevation
    except Exception as e:
        print(f"读取GEBCO数据失败: {e}"); return None, None, None

bathy_lons, bathy_lats, bathy_elevation = load_gebco_data(gebco_file, plot_extent)

# ================= 配色方案 =================
ocean_colors = ['#003366', '#004d7a', '#0066a3', '#0080cc', '#4da6d9', '#99ccee']
ocean_cmap = LinearSegmentedColormap.from_list('ocean', ocean_colors)

land_colors = ['#4A7C59', '#6B9A5F', '#8FB569', '#B8C77A', '#D4C19C', '#C8A882', '#B08968', '#8B6F47']
land_cmap = LinearSegmentedColormap.from_list('land_terrain', land_colors)

# ================= 绘图核心 =================
fig = plt.figure(figsize=(18, 12), facecolor='white')
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent(plot_extent, crs=ccrs.PlateCarree())

# 1. 绘制海洋
if bathy_lons is not None:
    bathy_lon_grid, bathy_lat_grid = np.meshgrid(bathy_lons, bathy_lats)
    ocean_data = np.ma.masked_where(bathy_elevation >= 0, bathy_elevation)
    ax.pcolormesh(bathy_lon_grid, bathy_lat_grid, ocean_data,
                  cmap=ocean_cmap, vmin=-5000, vmax=0,
                  transform=ccrs.PlateCarree(), zorder=0, alpha=0.8, shading='auto')
    
    contours = ax.contour(bathy_lon_grid, bathy_lat_grid, bathy_elevation,
                          levels=[-5000, -4000, -3000, -2000, -1000, -500, -200], 
                          colors='#2c3e50', linewidths=0.5, linestyles='-', alpha=0.4, 
                          transform=ccrs.PlateCarree(), zorder=1)
    ax.clabel(contours, inline=True, fontsize=8, fmt='%d m', colors='#2c3e50')

# 2. 绘制陆地
if bathy_lons is not None:
    land_data = np.ma.masked_where(bathy_elevation <= 0, bathy_elevation)
    ax.pcolormesh(bathy_lon_grid, bathy_lat_grid, land_data,
                  cmap=land_cmap, vmin=0, vmax=3000,
                  transform=ccrs.PlateCarree(), zorder=2, alpha=0.85, shading='auto')
    
    ax.contour(bathy_lon_grid, bathy_lat_grid, land_data,
               levels=[500, 1000, 1500, 2000, 2500], colors='#4A4A4A', 
               linewidths=0.4, linestyles='-', alpha=0.3, 
               transform=ccrs.PlateCarree(), zorder=3)

# 3. 陆地边界
# 注意：如果 shapereader 读取失败，这行可能会报错，建议加个 try-except 或确保路径正确
try:
    land_feature = cfeature.ShapelyFeature(shpreader.Reader(land_shp_path).geometries(), 
                                           ccrs.PlateCarree(), facecolor='none', 
                                           edgecolor='#2C3E50', linewidth=0.8)
    ax.add_feature(land_feature, zorder=4)
except Exception as e:
    print(f"Warning: Failed to load local shapefile. Using default. Error: {e}")
    ax.add_feature(cfeature.LAND, facecolor='none', edgecolor='#2C3E50', linewidth=0.8, zorder=4)

# 4. 指北针
x, y, arrow_len = 0.95, 0.92, 0.05
ax.annotate('N', xy=(x, y), xytext=(x, y-arrow_len),
            arrowprops=dict(facecolor='black', width=3, headwidth=10, edgecolor='white', linewidth=0.5),
            ha='center', va='bottom', fontsize=18, fontweight='bold',
            xycoords='axes fraction', zorder=10,
            bbox=dict(boxstyle='circle', facecolor='white', edgecolor='black', linewidth=1.5, pad=0.3))

# 5. 绘制矩形区域及标签
regions = [
    # A: 顶部
    {'name': 'Region A', 'lon_min': 90.0, 'lon_max': 100.0, 'lat_min': 5.0, 'lat_max': 10.0, 'color': '#E74C3C', 'pos': 'top'},
    # B: 左侧
    {'name': 'Region B', 'lon_min': 112.4, 'lon_max': 121.32, 'lat_min': 18.32, 'lat_max': 23.19, 'color': '#3498DB', 'pos': 'left'},
    # C: 顶部
    {'name': 'Region C', 'lon_min': 117.0, 'lon_max': 121.5, 'lat_min': 4.5, 'lat_max': 12.5, 'color': '#27AE60', 'pos': 'top'}
]

for region in regions:
    l_min, l_max, t_min, t_max = region['lon_min'], region['lon_max'], region['lat_min'], region['lat_max']
    
    # 矩形框
    ax.plot([l_min, l_max, l_max, l_min, l_min], [t_min, t_min, t_max, t_max, t_min], 
            color=region['color'], linewidth=3, linestyle='-', 
            transform=ccrs.PlateCarree(), zorder=6)
    
    # 根据 pos 设置文字位置
    if region['pos'] == 'left':
        # 左侧布局 (Region B)
        text_x = l_min - 0.5
        text_y = (t_min + t_max) / 2
        ha_val = 'right'
        va_val = 'center'
    else:
        # 顶部布局 (Region A, C)
        text_x = l_min
        text_y = t_max + 0.3
        ha_val = 'left'
        va_val = 'bottom'

    # 这里的字体会继承全局的 Times New Roman
    ax.text(text_x, text_y, region['name'], 
            color=region['color'], fontsize=13, fontweight='bold', 
            transform=ccrs.PlateCarree(),
            ha=ha_val, va=va_val,
            bbox=dict(facecolor='white', alpha=0.9, edgecolor=region['color'], 
                      linewidth=1.5, pad=3, boxstyle='round,pad=0.5'), zorder=7)

# 6. 【已删除】内波源地星星标记

# 7. 坐标轴设置
gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, zorder=4, linestyle='--')
gl.top_labels = False   
gl.right_labels = False 
gl.left_labels = True   
gl.bottom_labels = True 

# 【关键修改】：这里显式加入 'family': 'serif'，确保经纬度数字也是 Times New Roman
gl.xlabel_style = {'size': 12, 'color': 'black', 'weight': 'bold', 'family': 'serif'}
gl.ylabel_style = {'size': 12, 'color': 'black', 'weight': 'bold', 'family': 'serif'}

# 8. 边框
ax.spines['geo'].set_linewidth(2.5)
ax.spines['geo'].set_edgecolor('#2C3E50')

# 9. 【已删除】标题

# 10. 数据来源
ax.text(0.02, 0.02, '© GEBCO Bathymetry Data', 
        transform=ax.transAxes, fontsize=8, color='gray',
        va='bottom', ha='left', style='italic') # style='italic' 对应斜体 Times New Roman

plt.tight_layout()
plt.savefig('bathymetry_map_clean_times.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✓ 优化完成：已应用 Times New Roman 字体")
plt.close()