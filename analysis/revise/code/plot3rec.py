import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import font_manager
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
import xarray as xr
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter, zoom, label as nd_label
from PIL import Image
from osgeo import gdal
import glob
import os
import json
import gc  # 用于手动释放内存

# ================= 字体配置 =================
FONT_PATH = "/root/data/TIMES.TTF"
font_prop = font_manager.FontProperties(fname=FONT_PATH)
font_manager.fontManager.addfont(FONT_PATH)
FONT_NAME = font_prop.get_name()

mpl.rcParams.update({
    'font.family': FONT_NAME,
    'mathtext.fontset': 'stix',
    'axes.unicode_minus': False,
    'axes.labelsize': 28,
    'xtick.labelsize': 28,
    'ytick.labelsize': 28
})

# ================= 路径配置 =================
gebco_file_bg   = r'/root/data/GEBCO_15_Dec_2025_6d47b43e62e6/gebco_2025_n23.689_s-1.2161_w80.4424_e128.6421.nc'
land_shp_path   = r'/root/data/physical/ne_10m_land.shp'

# Region B (South China Sea) — 读取 npy
region_b_accumulated_file = r'/root/data/accumulated_mask.npy'
region_b_metadata_file    = r'/root/data/accumulated_mask_metadata.json'

# Region C (Sulu/Celebes Sea) — 实时读取 SAR
region_c_sar_dir            = r'/root/data/sar'
region_c_mask_dir           = r'/root/data/mask'
region_c_corrected_mask_dir = r'/root/data/mask_corrected'

# Region A (Andaman Sea) — 实时读取 SAR
region_a_sar_dir            = region_c_sar_dir
region_a_mask_dir           = region_c_mask_dir
region_a_corrected_mask_dir = region_c_corrected_mask_dir

os.makedirs(region_c_corrected_mask_dir, exist_ok=True)

# 大背景地图范围
plot_extent = [88, 128.6421, 3, 23.689]

# 三个 Region 范围
regions_def = {
    'A': {'lon_min': 90.0,   'lon_max': 100.0,   'lat_min': 5.0,   'lat_max': 10.0,  'color': '#E74C3C', 'pos': 'top',  'label': 'Region A'},
    'B': {'lon_min': 112.4,  'lon_max': 121.32,  'lat_min': 18.32, 'lat_max': 23.19, 'color': '#3498DB', 'pos': 'left', 'label': 'Region B'},
    'C': {'lon_min': 117.0,  'lon_max': 121.5,   'lat_min': 4.5,   'lat_max': 12.5,  'color': '#27AE60', 'pos': 'top',  'label': 'Region C'},
}

grid_res = 0.002  # °

# ================= 工具函数 =================

def load_gebco_data(gebco_path, lon_min, lon_max, lat_min, lat_max, buffer=0.0, stride=1):
    """加入 stride 降采样，防止读取全量高分数据爆内存"""
    try:
        ds = xr.open_dataset(gebco_path)
        lon_var  = 'lon'       if 'lon'       in ds.coords else list(ds.coords.keys())[0]
        lat_var  = 'lat'       if 'lat'       in ds.coords else list(ds.coords.keys())[1]
        elev_var = 'elevation' if 'elevation' in ds.data_vars else list(ds.data_vars.keys())[0]

        data = ds.sel({
            lon_var: slice(lon_min - buffer, lon_max + buffer),
            lat_var: slice(lat_min - buffer, lat_max + buffer)
        })
        
        # 降采样
        if stride > 1:
            data = data.isel({lon_var: slice(None, None, stride), lat_var: slice(None, None, stride)})

        lons, lats, elev = data[lon_var].values, data[lat_var].values, data[elev_var].values
        ds.close()
        return lons, lats, elev
    except Exception as e:
        print(f"读取GEBCO数据失败: {e}")
        return None, None, None

def correct_and_save_mask(tiff_path, mask_path, output_path):
    ds_gdal = gdal.Open(tiff_path)
    if ds_gdal is None:
        raise RuntimeError(f'无法打开 TIFF: {tiff_path}')

    meta      = ds_gdal.GetMetadata()
    orbit_dir = meta.get('OrbitDirection', '').upper()
    if orbit_dir not in ['ASCENDING', 'DESCENDING']:
        ds_gdal = None # 及时释放句柄
        raise RuntimeError(f'OrbitDirection 缺失或非法: {tiff_path}')

    ascending = (orbit_dir == 'ASCENDING')
    mask = (np.array(Image.open(mask_path).convert('L')) > 0).astype(np.uint8)

    min_lon = float(meta.get('min_lon', 0))
    max_lon = float(meta.get('max_lon', 0))
    min_lat = float(meta.get('min_lat', 0))
    max_lat = float(meta.get('max_lat', 0))

    if ascending:
        mask = np.fliplr(mask)
        mask = np.rot90(mask, 2)
    else:
        mask = np.fliplr(mask)

    Image.fromarray((mask * 255).astype(np.uint8)).save(output_path)
    
    # 彻底释放 GDAL 句柄防止内存泄漏
    ds_gdal = None 

    return {'corrected_path': output_path,
            'min_lon': min_lon, 'max_lon': max_lon,
            'min_lat': min_lat, 'max_lat': max_lat,
            'orbit_dir': orbit_dir}

def build_accumulated_from_sar(sar_dir, mask_dir, corrected_mask_dir,
                               lon_min, lon_max, lat_min, lat_max):
    mask_info_list = []
    for tiff_path in glob.glob(os.path.join(sar_dir, '*.tiff')):
        base       = os.path.splitext(os.path.basename(tiff_path))[0]
        mask_path  = os.path.join(mask_dir, base + '.png')
        if not os.path.exists(mask_path):
            continue
        corrected_path = os.path.join(corrected_mask_dir, base + '_corrected.png')
        try:
            info = correct_and_save_mask(tiff_path, mask_path, corrected_path)
            if (info['max_lon'] < lon_min or info['min_lon'] > lon_max or
                    info['max_lat'] < lat_min or info['min_lat'] > lat_max):
                continue
            test = np.array(Image.open(corrected_path).convert('L'))
            if test.sum() > 0:
                mask_info_list.append(info)
        except Exception as e:
            print(f"  处理失败 {base}: {e}")

    if not mask_info_list:
        print("  警告：没有有效 mask，返回空网格")
        n_lon = int((lon_max - lon_min) / grid_res) + 1
        n_lat = int((lat_max - lat_min) / grid_res) + 1
        lons      = np.linspace(lon_min, lon_max, n_lon)
        lats      = np.linspace(lat_min, lat_max, n_lat)
        lon_grid, lat_grid = np.meshgrid(lons, lats)
        return lon_grid, lat_grid, np.zeros((n_lat, n_lon), dtype=np.int16)

    all_min_lon = max(lon_min, min(m['min_lon'] for m in mask_info_list))
    all_max_lon = min(lon_max, max(m['max_lon'] for m in mask_info_list))
    all_min_lat = max(lat_min, min(m['min_lat'] for m in mask_info_list))
    all_max_lat = min(lat_max, max(m['max_lat'] for m in mask_info_list))

    all_min_lon = max(all_min_lon, lon_min)
    all_max_lon = min(all_max_lon, lon_max)
    all_min_lat = max(all_min_lat, lat_min)
    all_max_lat = min(all_max_lat, lat_max)

    n_lon = int((all_max_lon - all_min_lon) / grid_res) + 1
    n_lat = int((all_max_lat - all_min_lat) / grid_res) + 1
    lons      = np.linspace(all_min_lon, all_max_lon, n_lon)
    lats      = np.linspace(all_min_lat, all_max_lat, n_lat)
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    accumulated = np.zeros((n_lat, n_lon), dtype=np.int16)

    for info in mask_info_list:
        mask = (np.array(Image.open(info['corrected_path']).convert('L')) > 0).astype(np.uint8)
        j0 = int((info['min_lon'] - all_min_lon) / grid_res)
        j1 = int((info['max_lon'] - all_min_lon) / grid_res)
        i0 = int((info['min_lat'] - all_min_lat) / grid_res)
        i1 = int((info['max_lat'] - all_min_lat) / grid_res)
        i0, i1 = max(0, i0), min(n_lat, i1)
        j0, j1 = max(0, j0), min(n_lon, j1)
        h, w = i1 - i0, j1 - j0
        if h <= 0 or w <= 0:
            continue
        resized = zoom(mask.astype(float),
                       (h / mask.shape[0], w / mask.shape[1]), order=0)
        accumulated[i0:i1, j0:j1] += np.flipud(resized[:h, :w]).astype(np.int16)

    return lon_grid, lat_grid, accumulated

def overlay_region_data(ax, lon_grid, lat_grid, accumulated,
                        bathy_lons, bathy_lats, bathy_elevation,
                        vmax=5, sigma=0.6, contour_levels=None,
                        cmap='viridis'):

    display_data = gaussian_filter(accumulated.astype(float), sigma=sigma)
    masked_data  = np.ma.masked_where(display_data == 0, display_data)

    im = ax.pcolormesh(
        lon_grid, lat_grid, masked_data,
        cmap=cmap,                          
        transform=ccrs.PlateCarree(),
        vmin=0, vmax=vmax,
        shading='nearest',
        alpha=0.85,
        zorder=5,
        rasterized=True  # ★ 压缩体积：将内波热力图光栅化
    )

    if bathy_lons is not None and bathy_elevation is not None:
        if contour_levels is None:
            contour_levels = [-4000, -3000, -2000, -1000, -500, -200]
        bathy_lon_grid, bathy_lat_grid = np.meshgrid(bathy_lons, bathy_lats)
        cs = ax.contour(
            bathy_lon_grid, bathy_lat_grid, bathy_elevation,
            levels=contour_levels,
            colors='#666666',
            linewidths=0.6,
            linestyles='--',
            alpha=0.6,
            transform=ccrs.PlateCarree(),
            zorder=6
        )
        ax.clabel(cs, inline=True, fontsize=6, fmt='%d m', colors='#666666')
    
    # 手动释放内存
    del display_data, masked_data
    gc.collect()

    return im

# ================= 第一步：读取大背景数据 =================
print("读取大背景 GEBCO 数据...")
# ★ 解决OOM & 体积问题：stride=20 进行大幅度降采样
bg_lons, bg_lats, bg_elev = load_gebco_data(gebco_file_bg, plot_extent[0], plot_extent[1],
                                             plot_extent[2], plot_extent[3], buffer=0.5, stride=20)

# ================= 第二步：准备三个 Region 数据 =================

LOCAL_STRIDE = 2 # 局部区域由于面积小，使用较细的 stride 保证等深线质量

print("加载 Region B accumulated_mask.npy ...")
rb = regions_def['B']
if os.path.exists(region_b_accumulated_file):
    rb_accumulated = np.load(region_b_accumulated_file)
    with open(region_b_metadata_file, 'r') as f:
        rb_meta = json.load(f)
    rb_ny, rb_nx = rb_meta['shape']
    rb_lons     = np.linspace(rb['lon_min'], rb['lon_max'], rb_nx)
    rb_lats     = np.linspace(rb['lat_max'], rb['lat_min'], rb_ny) 
    rb_lon_grid, rb_lat_grid = np.meshgrid(rb_lons, rb_lats)
    print(f"  Region B: {rb_meta['num_masks']} 个 mask 已加载")
else:
    print("  警告：未找到 accumulated_mask.npy，Region B 将为空")
    rb_ny, rb_nx = 100, 100
    rb_lons     = np.linspace(rb['lon_min'], rb['lon_max'], rb_nx)
    rb_lats     = np.linspace(rb['lat_max'], rb['lat_min'], rb_ny)
    rb_lon_grid, rb_lat_grid = np.meshgrid(rb_lons, rb_lats)
    rb_accumulated = np.zeros((rb_ny, rb_nx), dtype=np.int16)

rb_bathy_lons, rb_bathy_lats, rb_bathy_elev = load_gebco_data(
    gebco_file_bg, rb['lon_min'], rb['lon_max'], rb['lat_min'], rb['lat_max'], stride=LOCAL_STRIDE)

print("构建 Region C accumulated ...")
rc = regions_def['C']
rc_lon_grid, rc_lat_grid, rc_accumulated = build_accumulated_from_sar(
    region_c_sar_dir, region_c_mask_dir, region_c_corrected_mask_dir,
    rc['lon_min'], rc['lon_max'], rc['lat_min'], rc['lat_max'])

rc_bathy_lons, rc_bathy_lats, rc_bathy_elev = load_gebco_data(
    gebco_file_bg, rc['lon_min'], rc['lon_max'], rc['lat_min'], rc['lat_max'], stride=LOCAL_STRIDE)

print("构建 Region A accumulated ...")
ra = regions_def['A']
ra_lon_grid, ra_lat_grid, ra_accumulated = build_accumulated_from_sar(
    region_a_sar_dir, region_a_mask_dir, region_a_corrected_mask_dir,
    ra['lon_min'], ra['lon_max'], ra['lat_min'], ra['lat_max'])

ra_bathy_lons, ra_bathy_lats, ra_bathy_elev = load_gebco_data(
    gebco_file_bg, ra['lon_min'], ra['lon_max'], ra['lat_min'], ra['lat_max'], stride=LOCAL_STRIDE)

# ================= 第三步：绘图 =================
print("开始绘图...")

ocean_colors = ['#003366', '#004d7a', '#0066a3', '#0080cc', '#4da6d9', '#99ccee']
ocean_cmap   = LinearSegmentedColormap.from_list('ocean', ocean_colors)
land_colors  = ['#4A7C59', '#6B9A5F', '#8FB569', '#B8C77A', '#D4C19C', '#C8A882', '#B08968', '#8B6F47']
land_cmap    = LinearSegmentedColormap.from_list('land_terrain', land_colors)

fig = plt.figure(figsize=(18, 12), facecolor='white')
ax  = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent(plot_extent, crs=ccrs.PlateCarree())

# --- 1. 背景海洋 ---
if bg_lons is not None:
    bg_lon_grid, bg_lat_grid = np.meshgrid(bg_lons, bg_lats)

    ocean_data = np.ma.masked_where(bg_elev >= 0, bg_elev)
    ax.pcolormesh(bg_lon_grid, bg_lat_grid, ocean_data,
                  cmap=ocean_cmap, vmin=-5000, vmax=0,
                  transform=ccrs.PlateCarree(), zorder=0, alpha=0.8, 
                  shading='nearest', rasterized=True) # ★ 压缩体积：背景光栅化

    bg_contours = ax.contour(bg_lon_grid, bg_lat_grid, bg_elev,
                             levels=[-5000, -4000, -3000, -2000, -1000, -500, -200],
                             colors='#2c3e50', linewidths=0.5, linestyles='-', alpha=0.4,
                             transform=ccrs.PlateCarree(), zorder=1)
    ax.clabel(bg_contours, inline=True, fontsize=8, fmt='%d m', colors='#2c3e50')
    
    # 释放海洋层内存
    del ocean_data
    gc.collect()

    # --- 2. 背景陆地 ---
    land_data = np.ma.masked_where(bg_elev <= 0, bg_elev)
    ax.pcolormesh(bg_lon_grid, bg_lat_grid, land_data,
                  cmap=land_cmap, vmin=0, vmax=3000,
                  transform=ccrs.PlateCarree(), zorder=2, alpha=0.85, 
                  shading='nearest', rasterized=True) # ★ 压缩体积：背景光栅化

    ax.contour(bg_lon_grid, bg_lat_grid, land_data,
               levels=[500, 1000, 1500, 2000, 2500], colors='#4A4A4A',
               linewidths=0.4, linestyles='-', alpha=0.3,
               transform=ccrs.PlateCarree(), zorder=3)
               
    # 释放陆地层内存
    del land_data, bg_lon_grid, bg_lat_grid, bg_elev
    gc.collect()

# --- 3. 陆地边界 ---
try:
    land_feature = cfeature.ShapelyFeature(
        shpreader.Reader(land_shp_path).geometries(),
        ccrs.PlateCarree(), facecolor='none', edgecolor='#2C3E50', linewidth=0.8)
    ax.add_feature(land_feature, zorder=4)
except Exception as e:
    print(f"Warning: 使用默认海岸线. Error: {e}")
    ax.add_feature(cfeature.LAND, facecolor='none', edgecolor='#2C3E50', linewidth=0.8, zorder=4)

# --- 4. 叠加三个 Region 的内波密度 + 等深线 ---
print("叠加 Region A ...")
overlay_region_data(ax, ra_lon_grid, ra_lat_grid, ra_accumulated,
                    ra_bathy_lons, ra_bathy_lats, ra_bathy_elev,
                    vmax=5, sigma=0.6, cmap='YlOrRd')

print("叠加 Region B ...")
rb_vmax = max(10, min(np.percentile(rb_accumulated[rb_accumulated > 0], 95)
                      if (rb_accumulated > 0).any() else 10, 30))

THRESHOLD = 2
rb_accumulated_filtered = rb_accumulated.copy()
rb_accumulated_filtered[rb_accumulated_filtered < THRESHOLD] = 0

MIN_AREA = 25 
binary = (rb_accumulated_filtered > 0).astype(np.uint8)
labeled, num_features = nd_label(binary)
for region_id in range(1, num_features + 1):
    if (labeled == region_id).sum() < MIN_AREA:
        rb_accumulated_filtered[labeled == region_id] = 0

overlay_region_data(ax, rb_lon_grid, rb_lat_grid, rb_accumulated_filtered,
                    rb_bathy_lons, rb_bathy_lats, rb_bathy_elev,
                    vmax=rb_vmax, sigma=2.0, cmap='plasma')

print("叠加 Region C ...")
overlay_region_data(ax, rc_lon_grid, rc_lat_grid, rc_accumulated,
                    rc_bathy_lons, rc_bathy_lats, rc_bathy_elev,
                    vmax=5, sigma=0.6, cmap='YlOrRd')

# --- 5. 绘制三个矩形框 + 标签 ---
for key, region in regions_def.items():
    l_min, l_max = region['lon_min'], region['lon_max']
    t_min, t_max = region['lat_min'], region['lat_max']
    color = region['color']

    ax.plot([l_min, l_max, l_max, l_min, l_min],
            [t_min, t_min, t_max, t_max, t_min],
            color=color, linewidth=3, linestyle='-',
            transform=ccrs.PlateCarree(), zorder=8)

    if region['pos'] == 'left':
        text_x, text_y = l_min - 0.5, (t_min + t_max) / 2
        ha_val, va_val = 'right', 'center'
    else:
        text_x, text_y = l_min, t_max + 0.3
        ha_val, va_val = 'left', 'bottom'

    ax.text(text_x, text_y, region['label'],
            color=color, fontsize=13, fontweight='bold',
            transform=ccrs.PlateCarree(), ha=ha_val, va=va_val,
            fontproperties=font_prop,
            bbox=dict(facecolor='white', alpha=0.9, edgecolor=color,
                      linewidth=1.5, pad=3, boxstyle='round,pad=0.5'),
            zorder=9)

# --- 6. 指北针 ---
x, y, arrow_len = 0.95, 0.92, 0.05
ax.annotate('N', xy=(x, y), xytext=(x, y - arrow_len),
            arrowprops=dict(facecolor='black', width=3, headwidth=10,
                            edgecolor='white', linewidth=0.5),
            ha='center', va='bottom', fontsize=18, fontweight='bold',
            xycoords='axes fraction', zorder=10, fontproperties=font_prop,
            bbox=dict(boxstyle='circle', facecolor='white', edgecolor='black',
                      linewidth=1.5, pad=0.3))

# --- 7. 坐标轴 ---
gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray',
                  alpha=0.5, zorder=4, linestyle='--')
gl.top_labels    = False
gl.right_labels  = False
gl.left_labels   = True
gl.bottom_labels = True
gl.xlabel_style  = {'size': 28, 'color': 'black', 'weight': 'bold', 'family': FONT_NAME}
gl.ylabel_style  = {'size': 28, 'color': 'black', 'weight': 'bold', 'family': FONT_NAME}

# --- 9. 边框 ---
ax.spines['geo'].set_linewidth(2.5)
ax.spines['geo'].set_edgecolor('#2C3E50')

# --- 10. 数据来源 ---
ax.text(0.02, 0.02, '© GEBCO Bathymetry Data',
        transform=ax.transAxes, fontsize=8, color='gray',
        va='bottom', ha='left', style='italic', fontproperties=font_prop)

plt.tight_layout()
output_name = 'integrated_bathymetry_internal_wave_map.svg'
fig.canvas.draw()
# ★ 压缩体积：将 dpi 设为 150。对于光栅化（rasterized）的背景已经足够清晰，而文字和线条依然是纯矢量不会模糊。
plt.savefig(output_name, dpi=150, bbox_inches='tight', facecolor='white')
print(f"✓ 完成，已保存：{output_name}")
plt.close()