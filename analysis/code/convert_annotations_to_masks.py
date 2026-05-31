import os
import json
import numpy as np
import cv2
from scipy.interpolate import UnivariateSpline
from scipy.ndimage import gaussian_filter1d

# ====== 路径设置 ======
DATA_ROOT_SAR = '/home/xiaobowen/project/internal_wave_detection_project/IW_data/images_sar'
LABEL_DIR = '/home/xiaobowen/project/internal_wave_detection_project/IW_data/annotations_sar'
VIS_ROOT_DIR = '/home/xiaobowen/project/internal_wave_detection_project/IW_data/process_visualization'

# ====== 核心参数 ======
N_CURVE_POINTS = 3000     
MAX_RADIUS = 5            
ENERGY_THRESHOLD = 0.96   
WIDTH_SMOOTH_SIGMA = 40   
OVERSAMPLE = 4            

def get_adaptive_width(img, x, y, nx, ny, max_r=MAX_RADIUS, threshold=ENERGY_THRESHOLD):
    h, w = img.shape
    ix, iy = int(np.clip(x, 0, w-1)), int(np.clip(y, 0, h-1))
    center_val = float(img[iy, ix])
    if center_val <= 0: return 2.0
    def probe(direction):
        for r in range(1, max_r):
            px, py = int(x + direction * r * nx), int(y + direction * r * ny)
            if 0 <= px < w and 0 <= py < h:
                if img[py, px] < center_val * threshold: return r
            else: return r
        return max_r
    return float(probe(1) + probe(-1))

def ultra_smooth_curve(points, n_points=N_CURVE_POINTS, smooth_factor=0.001):
    points = np.array(points)
    if len(points) < 4: return None
    deltas = np.diff(points, axis=0); dist = np.sqrt((deltas ** 2).sum(axis=1))
    dist[dist == 0] = 1e-6; t = np.hstack(([0], np.cumsum(dist))); t /= t[-1]
    x_spline = UnivariateSpline(t, points[:, 0], s=smooth_factor)
    y_spline = UnivariateSpline(t, points[:, 1], s=smooth_factor)
    return np.stack([x_spline(np.linspace(0, 1, n_points)), y_spline(np.linspace(0, 1, n_points))], axis=1)

def process_and_save_stages(sar_img, label_path, save_dir):
    h, w = sar_img.shape
    sar_smooth = cv2.GaussianBlur(sar_img, (5, 5), 0)
    with open(label_path, 'r') as f:
        data = json.load(f)

    # --- 图 0: 原始 SAR 图 ---
    cv2.imwrite(os.path.join(save_dir, "0_original_sar.jpg"), sar_img)

    stage1_points = cv2.cvtColor(sar_img, cv2.COLOR_GRAY2BGR)
    stage2_spine = cv2.cvtColor(sar_img, cv2.COLOR_GRAY2BGR)
    stage3_probing = stage2_spine.copy()
    stage4_final = np.zeros((h, w), dtype=np.uint8)

    control_points_list = data.get('curves', [])

    for points in control_points_list:
        # --- 图 1: 关键点可视化 ---
        for pt in np.array(points, dtype=np.int32):
            cv2.circle(stage1_points, (pt[0], pt[1]), 4, (0, 255, 0), -1) # 绿圆
            cv2.drawMarker(stage1_points, (pt[0], pt[1]), (0, 0, 255), cv2.MARKER_CROSS, 8, 1) # 红叉

        # --- 图 2: 脊线拟合 ---
        curve = ultra_smooth_curve(points)
        if curve is None: continue
        cv2.polylines(stage2_spine, [curve.astype(np.int32)], False, (0, 0, 255), 2)

        # 采样与宽度计算
        raw_widths, normals = [], []
        for i in range(len(curve)):
            idx_next, idx_prev = min(i+1, len(curve)-1), max(i-1, 0)
            tangent = curve[idx_next] - curve[idx_prev]
            dt = np.linalg.norm(tangent)
            nx, ny = (-tangent[1]/dt, tangent[0]/dt) if dt > 1e-6 else (0, 0)
            raw_widths.append(get_adaptive_width(sar_smooth, curve[i,0], curve[i,1], nx, ny))
            normals.append([nx, ny])
        smoothed_widths = gaussian_filter1d(raw_widths, sigma=WIDTH_SMOOTH_SIGMA)

        # --- 图 3: 探测块 (高亮青色 + 圆润处理) ---
        probing_overlay = np.zeros_like(stage3_probing)
        bright_cyan = (255, 255, 0) # 亮青色
        
        # 3.1 绘制分段探测块
        for i in range(0, len(curve) - 1, 2): 
            p1, p2 = curve[i], curve[i+1]
            nx, ny = normals[i]
            r1, r2 = smoothed_widths[i]/2.0, smoothed_widths[i+1]/2.0
            quad = np.array([p1 + r1*np.array([nx,ny]), p1 - r1*np.array([nx,ny]),
                             p2 - r2*np.array([nx,ny]), p2 + r2*np.array([nx,ny])], dtype=np.int32)
            cv2.fillPoly(probing_overlay, [quad], bright_cyan)
        
        # 3.2 绘制图 3 端点圆头 (解决图 3 的切割感)
        s_r3 = int(smoothed_widths[0]/2.0)
        e_r3 = int(smoothed_widths[-1]/2.0)
        cv2.circle(probing_overlay, tuple(curve[0].astype(np.int32)), s_r3, bright_cyan, -1, lineType=cv2.LINE_AA)
        cv2.circle(probing_overlay, tuple(curve[-1].astype(np.int32)), e_r3, bright_cyan, -1, lineType=cv2.LINE_AA)
        
        # 透明融合
        cv2.addWeighted(probing_overlay, 0.6, stage3_probing, 1.0, 0, stage3_probing)

        # --- 图 4: 最终丝滑 Mask (圆润化 + 外推) ---
        pad = MAX_RADIUS + 10
        c_min, c_max = np.min(curve, axis=0)-pad, np.max(curve, axis=0)+pad
        mx, my, Mx, My = int(max(0,c_min[0])), int(max(0,c_min[1])), int(min(w,c_max[0])), int(min(h,c_max[1]))
        
        up_w, up_h = (Mx-mx)*OVERSAMPLE, (My-my)*OVERSAMPLE
        up_mask = np.zeros((up_h, up_w), dtype=np.uint8)
        up_curve = (curve - [mx, my]) * OVERSAMPLE
        
        # 绘制主干距离场
        up_spine = np.zeros((up_h, up_w), dtype=np.uint8)
        cv2.polylines(up_spine, [up_curve.astype(np.int32)], False, 255, 1)
        dist_field = cv2.distanceTransform(255 - up_spine, cv2.DIST_L2, 5)
        
        up_width_map = np.zeros((up_h, up_w), dtype=np.float32)
        for i in range(len(up_curve)):
            cx, cy = int(np.clip(up_curve[i,0], 0, up_w-1)), int(np.clip(up_curve[i,1], 0, up_h-1))
            up_width_map[cy, cx] = (smoothed_widths[i] * OVERSAMPLE) / 2.0
        up_width_map = cv2.dilate(up_width_map, np.ones((int(5*OVERSAMPLE), int(5*OVERSAMPLE)), np.uint8))
        up_mask = (dist_field <= up_width_map).astype(np.uint8) * 255

        # 【核心优化】端点外推逻辑，确保圆头圆润且不缩水
        # 计算两端的延长趋势向量
        s_vec = (curve[0] - curve[10]) if len(curve)>10 else (curve[0]-curve[1])
        e_vec = (curve[-1] - curve[-11]) if len(curve)>10 else (curve[-1]-curve[-2])
        s_vec /= (np.linalg.norm(s_vec) + 1e-6)
        e_vec /= (np.linalg.norm(e_vec) + 1e-6)

        # 在超采样图上绘制外推圆
        s_pt = (up_curve[0] + s_vec * OVERSAMPLE * 1.5).astype(np.int32)
        e_pt = (up_curve[-1] + e_vec * OVERSAMPLE * 1.5).astype(np.int32)
        s_r = int(max(smoothed_widths[0], 2.5) * OVERSAMPLE / 2.0)
        e_r = int(max(smoothed_widths[-1], 2.5) * OVERSAMPLE / 2.0)
        
        cv2.circle(up_mask, tuple(s_pt), s_r, 255, -1, lineType=cv2.LINE_AA)
        cv2.circle(up_mask, tuple(e_pt), e_r, 255, -1, lineType=cv2.LINE_AA)
        
        # 降采样回原尺寸，并平滑边缘
        local_mask = cv2.resize(up_mask, (Mx-mx, My-my), interpolation=cv2.INTER_AREA)
        _, local_mask = cv2.threshold(local_mask, 110, 255, cv2.THRESH_BINARY)
        stage4_final[my:My, mx:Mx] = cv2.bitwise_or(stage4_final[my:My, mx:Mx], local_mask)

    # --- 统一保存 ---
    cv2.imwrite(os.path.join(save_dir, "1_control_points.jpg"), stage1_points)
    cv2.imwrite(os.path.join(save_dir, "2_spine_fitting.jpg"), stage2_spine)
    cv2.imwrite(os.path.join(save_dir, "3_adaptive_probing_boxes.jpg"), stage3_probing)
    cv2.imwrite(os.path.join(save_dir, "4_final_mask.png"), stage4_final)

def process_all():
    os.makedirs(VIS_ROOT_DIR, exist_ok=True)
    lfs = sorted([f for f in os.listdir(LABEL_DIR) if f.endswith('.json')])
    for lf in lfs:
        name = os.path.splitext(lf)[0]
        p = os.path.join(DATA_ROOT_SAR, name + '.jpg')
        if not os.path.exists(p): p = os.path.join(DATA_ROOT_SAR, name + '.tif')
        if not os.path.exists(p): continue
        img = cv2.imread(p, 0)
        if img is None: continue
        sdir = os.path.join(VIS_ROOT_DIR, name); os.makedirs(sdir, exist_ok=True)
        try:
            process_and_save_stages(img, os.path.join(LABEL_DIR, lf), sdir)
            print(f"Success: {name}")
        except Exception as e: print(f"Error {name}: {e}")

if __name__ == '__main__':
    process_all()