import os
import json
import numpy as np
import cv2
from scipy.interpolate import UnivariateSpline

# ====== 路径设置 ======
LABEL_DIR = '/home/xiaobowen/project/internal_wave_detection_project/IW_data/annotations_sar'
OUTPUT_DIR = '/home/xiaobowen/project/internal_wave_detection_project/IW_data/masks'

# ====== 核心参数 ======
N_CURVE_POINTS = 3000
OVERSAMPLE = 4
WIDTH = 3  # 固定线宽（像素），按需调整

DEFAULT_H, DEFAULT_W = 1024, 1024


def ultra_smooth_curve(points, n_points=N_CURVE_POINTS, smooth_factor=0.001):
    points = np.array(points)
    if len(points) < 4:
        return None
    deltas = np.diff(points, axis=0)
    dist = np.sqrt((deltas ** 2).sum(axis=1))
    dist[dist == 0] = 1e-6
    t = np.hstack(([0], np.cumsum(dist)))
    t /= t[-1]
    x_spline = UnivariateSpline(t, points[:, 0], s=smooth_factor)
    y_spline = UnivariateSpline(t, points[:, 1], s=smooth_factor)
    return np.stack([x_spline(np.linspace(0, 1, n_points)),
                     y_spline(np.linspace(0, 1, n_points))], axis=1)


def generate_mask(label_path, h=DEFAULT_H, w=DEFAULT_W):
    with open(label_path, 'r') as f:
        data = json.load(f)

    # 支持JSON里存了图像尺寸
    h = data.get('imageHeight', h)
    w = data.get('imageWidth', w)

    mask = np.zeros((h, w), dtype=np.uint8)
    half_w = WIDTH / 2.0

    for points in data.get('curves', []):
        curve = ultra_smooth_curve(points)
        if curve is None:
            continue

        # 计算曲线bbox，局部超采样保证边缘平滑
        c_min = np.min(curve, axis=0) - (half_w + 2)
        c_max = np.max(curve, axis=0) + (half_w + 2)
        min_x = int(max(0, c_min[0]))
        min_y = int(max(0, c_min[1]))
        max_x = int(min(w, c_max[0]))
        max_y = int(min(h, c_max[1]))
        local_w = max_x - min_x
        local_h = max_y - min_y
        if local_w <= 0 or local_h <= 0:
            continue

        up_w = local_w * OVERSAMPLE
        up_h = local_h * OVERSAMPLE

        # 在超采样空间里画脊线，然后做距离变换
        up_spine = np.zeros((up_h, up_w), dtype=np.uint8)
        up_curve = (curve - [min_x, min_y]) * OVERSAMPLE
        cv2.polylines(up_spine, [up_curve.astype(np.int32)], False, 255, 1)

        dist_field = cv2.distanceTransform(255 - up_spine, cv2.DIST_L2, 5)
        up_mask = (dist_field <= half_w * OVERSAMPLE).astype(np.uint8) * 255

        # 下采样回原分辨率
        local_mask = cv2.resize(up_mask, (local_w, local_h),
                                interpolation=cv2.INTER_AREA)
        _, local_mask = cv2.threshold(local_mask, 127, 255, cv2.THRESH_BINARY)
        mask[min_y:max_y, min_x:max_x] = cv2.bitwise_or(
            mask[min_y:max_y, min_x:max_x], local_mask)

    return mask


def process_all():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    label_files = sorted([f for f in os.listdir(LABEL_DIR) if f.endswith('.json')])

    for lf in label_files:
        base_name = os.path.splitext(lf)[0]
        label_path = os.path.join(LABEL_DIR, lf)
        out_path = os.path.join(OUTPUT_DIR, base_name + '_mask.png')

        try:
            mask = generate_mask(label_path)
            cv2.imwrite(out_path, mask)
            print(f"✓ {base_name}")
        except Exception as e:
            print(f"✗ {base_name}: {e}")


if __name__ == '__main__':
    process_all()