import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from PIL import Image
from osgeo import gdal
import glob
import os
from scipy.ndimage import gaussian_filter
import xarray as xr
import gc
from matplotlib import font_manager
import matplotlib as mpl

# ================= 配置 =================
sar_dir = r'/root/data/sar'
mask_dir = r'/root/data/mask_sar'
corrected_mask_dir = r'/root/data/mask_corrected'
gebco_file = r'/root/data/GEBCO_15_Dec_2025_6d47b43e62e6/gebco_2025_n23.689_s-1.2161_w80.4424_e128.6421.nc'

os.makedirs(corrected_mask_dir, exist_ok=True)

# ================= 字体配置 =================
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

gebco_lon_min, gebco_lon_max = 117.0, 121.5
gebco_lat_min, gebco_lat_max = 4.5, 12.5

grid_res = 0.002  # °

# ================= 读取GEBCO数据 =================
def load_gebco_data(gebco_path, lon_min, lon_max, lat_min, lat_max):
    print("正在读取GEBCO数据...")
    try:
        ds = xr.open_dataset(gebco_path)

        if 'lon' in ds.coords: lon_var = 'lon'
        elif 'longitude' in ds.coords: lon_var = 'longitude'
        else: lon_var = list(ds.coords.keys())[0]

        if 'lat' in ds.coords: lat_var = 'lat'
        elif 'latitude' in ds.coords: lat_var = 'latitude'
        else: lat_var = list(ds.coords.keys())[1]

        if 'elevation' in ds.data_vars: elev_var = 'elevation'
        elif 'z' in ds.data_vars: elev_var = 'z'
        else: elev_var = list(ds.data_vars.keys())[0]

        data = ds.sel(
            {lon_var: slice(lon_min, lon_max),
             lat_var: slice(lat_min, lat_max)}
        )

        lons = data[lon_var].values
        lats = data[lat_var].values
        elevation = data[elev_var].values

        print(f"成功读取GEBCO数据: {lons.shape[0]} x {lats.shape[0]} 网格点")
        ds.close()
        return lons, lats, elevation
    except Exception as e:
        print(f"读取GEBCO数据失败: {e}")
        return None, None, None

# ================= 第一步：修正并保存 mask =================
def correct_and_save_mask(tiff_path, mask_path, output_path):
    ds = gdal.Open(tiff_path)
    if ds is None: raise RuntimeError(f'无法打开 TIFF: {tiff_path}')

    meta = ds.GetMetadata()
    orbit_dir = meta.get('OrbitDirection', '').upper()
    if orbit_dir not in ['ASCENDING', 'DESCENDING']:
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

    return {
        'corrected_path': output_path,
        'min_lon': min_lon,
        'max_lon': max_lon,
        'min_lat': min_lat,
        'max_lat': max_lat,
        'orbit_dir': orbit_dir
    }

print("开始处理 Mask 数据...")
mask_info_list = []
for tiff_path in glob.glob(os.path.join(sar_dir, '*.tiff')):
    base = os.path.splitext(os.path.basename(tiff_path))[0]
    mask_path = os.path.join(mask_dir, base + '.png')

    if not os.path.exists(mask_path): continue

    corrected_path = os.path.join(corrected_mask_dir, base + '_corrected.png')
    try:
        info = correct_and_save_mask(tiff_path, mask_path, corrected_path)
        test_mask = np.array(Image.open(corrected_path).convert('L'))
        if test_mask.sum() > 0:
            mask_info_list.append(info)
    except Exception as e:
        print(f"处理失败 {base}: {e}")
        continue

if not mask_info_list: raise RuntimeError("没有有效 mask")
print(f"已处理 {len(mask_info_list)} 个 mask")

# ================= 第二步：累积密度计算 =================
all_min_lon = gebco_lon_min
all_max_lon = gebco_lon_max
all_min_lat = gebco_lat_min
all_max_lat = gebco_lat_max

print(f"网格已锁定至目标范围: lon [{all_min_lon:.4f}, {all_max_lon:.4f}], lat [{all_min_lat:.4f}, {all_max_lat:.4f}]")

n_lon = int((all_max_lon - all_min_lon) / grid_res) + 1
n_lat = int((all_max_lat - all_min_lat) / grid_res) + 1

lons = np.linspace(all_min_lon, all_max_lon, n_lon)
lats = np.linspace(all_min_lat, all_max_lat, n_lat)
lon_grid, lat_grid = np.meshgrid(lons, lats)

accumulated = np.zeros((n_lat, n_lon), dtype=np.int16)

print("正在计算空间重叠密度...")
for idx, info in enumerate(mask_info_list):
    if (info['max_lon'] < all_min_lon or info['min_lon'] > all_max_lon or
        info['max_lat'] < all_min_lat or info['min_lat'] > all_max_lat):
        continue

    mask_arr = (np.array(Image.open(info['corrected_path']).convert('L')) > 0).astype(np.uint8)

    j0 = int((info['min_lon'] - all_min_lon) / grid_res)
    j1 = int((info['max_lon'] - all_min_lon) / grid_res)
    i0 = int((info['min_lat'] - all_min_lat) / grid_res)
    i1 = int((info['max_lat'] - all_min_lat) / grid_res)

    h_target, w_target = i1 - i0, j1 - j0
    if h_target <= 0 or w_target <= 0:
        continue

    mask_img = Image.fromarray(mask_arr)
    resized_img = mask_img.resize((w_target, h_target), resample=Image.NEAREST)
    resized_arr = np.array(resized_img, dtype=np.int16)
    resized_arr = np.flipud(resized_arr)

    valid_i0, valid_i1 = max(0, i0), min(n_lat, i1)
    valid_j0, valid_j1 = max(0, j0), min(n_lon, j1)

    patch_i0 = valid_i0 - i0
    patch_i1 = patch_i0 + (valid_i1 - valid_i0)
    patch_j0 = valid_j0 - j0
    patch_j1 = patch_j0 + (valid_j1 - valid_j0)

    accumulated[valid_i0:valid_i1, valid_j0:valid_j1] += resized_arr[patch_i0:patch_i1, patch_j0:patch_j1]

    del mask_arr, mask_img, resized_img, resized_arr

    if idx % 50 == 0:
        gc.collect()

# ================= 读取GEBCO测深数据 =================
bathy_lons, bathy_lats, bathy_elevation = load_gebco_data(
    gebco_file, gebco_lon_min, gebco_lon_max, gebco_lat_min, gebco_lat_max
)

# ================= 绘图（完全对齐代码2格式） =================
print("正在生成高分辨率可视化图表...")
cmap = plt.cm.viridis

fig = plt.figure(figsize=(18, 12))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([gebco_lon_min, gebco_lon_max, gebco_lat_min, gebco_lat_max], crs=ccrs.PlateCarree())

# 计算vmax
vmax_percentile = np.percentile(accumulated[accumulated > 0], 95) if (accumulated > 0).any() else 5
vmax_value = max(10, min(vmax_percentile, 30))

# 平滑处理
display_data = gaussian_filter(accumulated.astype(float), sigma=2.0)
masked_data = np.ma.masked_where(display_data < 0.5, display_data)

# 绘制密度图（无colorbar，对齐代码2）
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

# 绘制等深线
if bathy_lons is not None and bathy_elevation is not None:
    bathy_lon_grid, bathy_lat_grid = np.meshgrid(bathy_lons, bathy_lats)

    contour_levels = [-4000, -3000, -2000, -1000, -500, -200, -100, -50]
    ax.contour(
        bathy_lon_grid, bathy_lat_grid, bathy_elevation,
        levels=contour_levels,
        colors='#666666',
        linewidths=0.8,
        linestyles='--',
        alpha=0.6,
        transform=ccrs.PlateCarree(),
        zorder=5
    )

    # 200米等深线单独加粗
    ax.contour(
        bathy_lon_grid, bathy_lat_grid, bathy_elevation,
        levels=[-200],
        colors='dimgray',
        linewidths=1.2,
        linestyles='dashed',
        alpha=0.7,
        transform=ccrs.PlateCarree(),
        zorder=5
    )

# 地图要素
ax.add_feature(cfeature.LAND.with_scale('10m'), facecolor='white', edgecolor='none', zorder=2)
ax.add_feature(cfeature.COASTLINE.with_scale('10m'), linewidth=0.8, edgecolor='black', zorder=3)

try:
    islands = cfeature.NaturalEarthFeature(
        category='physical', name='minor_islands', scale='10m',
        edgecolor='black', facecolor='white', linewidth=0.6, alpha=0.7
    )
    ax.add_feature(islands, zorder=2)
except Exception as e:
    pass

# 经纬度网格标签（完全对齐代码2）
gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--', zorder=4)
gl.top_labels = False
gl.right_labels = False
gl.xlabel_style = {'fontsize': 34, 'fontweight': 'bold', 'color': 'black'}
gl.ylabel_style = {'fontsize': 34, 'fontweight': 'bold', 'color': 'black'}

plt.tight_layout()
plt.savefig('sulu.png', dpi=600, bbox_inches='tight', pad_inches=0.05)
plt.close()

print(f"大功告成！已保存: sulu.png")
print(f"数据统计: 最大重叠={accumulated.max()}, 平均重叠={accumulated[accumulated > 0].mean() if (accumulated > 0).any() else 0:.2f}")