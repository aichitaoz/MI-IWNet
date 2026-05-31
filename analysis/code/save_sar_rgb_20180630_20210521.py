import rasterio
import numpy as np
from rasterio.windows import from_bounds
from rasterio.enums import Resampling
from pathlib import Path

# ======================
# 配置
# ======================
sar_tif = "sar/s1a-iw-grd-vv-20180630t102501-20180630t102530-022585-027255-001.tiff"
ref_tif = "rgb/MODIS_TrueColor_2018-06-30_1Terra.tiff"

out_dir = Path("processed_tiff")
out_dir.mkdir(exist_ok=True)

sar_out = out_dir / Path(sar_tif).name
ref_out = out_dir / "rgb_clipped_by_sar_extent.tif"

TARGET_SIZE = 1024  # 目标尺寸

# ======================
# 工具函数
# ======================
def flip_and_rotate(img, orbit_direction):
    img = np.fliplr(img)
    if orbit_direction.upper() == "ASCENDING":
        img = np.rot90(img, 2)
    return img


def resize_image(img, target_size):
    """将图像resize到target_size x target_size"""
    from scipy.ndimage import zoom
    
    if img.ndim == 2:  # 单通道
        h, w = img.shape
        zoom_factor = (target_size / h, target_size / w)
        return zoom(img, zoom_factor, order=1)  # order=1 为双线性插值
    else:  # 多通道 (bands, h, w)
        bands, h, w = img.shape
        zoom_factor = (1, target_size / h, target_size / w)
        return zoom(img, zoom_factor, order=1)


# ======================
# 第一部分：SAR 方向校正、resize并保存
# ======================
with rasterio.open(sar_tif) as src:
    tags = src.tags()
    orbit_direction = tags.get("OrbitDirection", "UNKNOWN")

    sar_lon_min = float(tags["min_lon"])
    sar_lon_max = float(tags["max_lon"])
    sar_lat_min = float(tags["min_lat"])
    sar_lat_max = float(tags["max_lat"])

    sar_img = src.read(1)
    sar_meta = src.meta.copy()

sar_img = flip_and_rotate(sar_img, orbit_direction)
sar_img_resized = resize_image(sar_img, TARGET_SIZE)

# 更新元数据
sar_meta.update({
    "height": TARGET_SIZE,
    "width": TARGET_SIZE
})

with rasterio.open(sar_out, "w", **sar_meta) as dst:
    dst.write(sar_img_resized, 1)
    dst.update_tags(**tags)

print(f"✅ Oriented SAR (1024×1024) saved: {sar_out}")


# ======================
# 第二部分：RGB TIFF 裁剪并resize
# ======================
with rasterio.open(ref_tif) as src:
    window = from_bounds(
        left=sar_lon_min,
        bottom=sar_lat_min+0.75,
        right=sar_lon_max-0.4,
        top=sar_lat_max-0.3,
        transform=src.transform
    )

    clipped = src.read(window=window)   # shape = (3, H, W)
    clipped_transform = src.window_transform(window)

    out_meta = src.meta.copy()

# Resize RGB
clipped_resized = resize_image(clipped, TARGET_SIZE)

# 更新元数据
out_meta.update({
    "height": TARGET_SIZE,
    "width": TARGET_SIZE,
    "transform": clipped_transform
})

with rasterio.open(ref_out, "w", **out_meta) as dst:
    dst.write(clipped_resized)

print(f"✅ RGB clipped TIFF (1024×1024) saved: {ref_out}")