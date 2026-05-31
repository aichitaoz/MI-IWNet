import cv2
import numpy as np
import pandas as pd
import os
from skimage import measure, morphology
from scipy.spatial.distance import pdist, squareform
from scipy.stats import skew, kurtosis
from collections import defaultdict, deque
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

# ================= 路径配置保持不变 =================
CONFIG = {
    "SAR": {
        "mask_dir": "/home/xiaobowen/project/internal_wave_detection_project/IW_data/masks_sar",
        "img_dir": "/home/xiaobowen/project/internal_wave_detection_project/IW_data/images_sar",
        "sensor_name": "SAR"
    },
    "MODIS": {
        "mask_dir": "/home/xiaobowen/project/internal_wave_detection_project/IW_data/masks_rgb",
        "img_dir": "/home/xiaobowen/project/internal_wave_detection_project/IW_data/images_rgb",
        "sensor_name": "MODIS"
    }
}

def calculate_curvature(pts):
    if len(pts) < 10: return 0
    p1, p2 = pts[0], pts[-1]
    chord_len = np.linalg.norm(p1 - p2)
    if chord_len < 2: return 0
    v_chord = p2 - p1
    v_pts = pts - p1
    cross_product = np.abs(v_pts[:,0] * v_chord[1] - v_pts[:,1] * v_chord[0])
    return (np.max(cross_product) / chord_len) / chord_len

def extract_normalized_features(mask_path, img_dir, sensor_name):
    try:
        mask = cv2.imread(mask_path, 0)
        if mask is None: return []
        h, w = mask.shape
        diag_len = np.sqrt(h**2 + w**2)
        
        stem = Path(mask_path).stem.replace("_mask", "")
        img_gray = None
        for ext in ['.png', '.jpg', '.jpeg', '.tiff']:
            p = os.path.join(img_dir, stem + ext)
            if os.path.exists(p):
                img_gray = cv2.imread(p, 0)
                break

        _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        labels = measure.label(binary)
        regions = measure.regionprops(labels)
        
        stripes = []
        for p in regions:
            if p.area < 30: continue
            m = (labels == p.label).astype(np.uint8)
            skel = morphology.skeletonize(m > 0)
            pts = np.column_stack(np.where(skel))[:, [1,0]]
            if len(pts) < 5: continue
            
            # --- 核心：多维度深度特征挖掘 ---
            feat_dict = {'curvature': calculate_curvature(pts)}
            
            if img_gray is not None:
                # 提取目标像素区域
                s_px = img_gray[m > 0].astype(np.float32)
                # 提取背景区域
                kernel = np.ones((5,5), np.uint8)
                dilated = cv2.dilate(m, kernel)
                bg_mask = cv2.subtract(dilated, m)
                b_px = img_gray[bg_mask > 0].astype(np.float32)
                
                # A. 统计分布维度 (Statistical Moments)
                feat_dict['pixel_mean'] = np.mean(s_px)
                feat_dict['pixel_std'] = np.std(s_px)
                feat_dict['pixel_skew'] = skew(s_px) if len(s_px)>10 else 0
                feat_dict['pixel_kurt'] = kurtosis(s_px) if len(s_px)>10 else 0
                feat_dict['pixel_cv'] = np.std(s_px) / (np.mean(s_px) + 1e-5) # 变异系数
                
                # B. 相对辐射维度 (Radiometric)
                if len(b_px) > 5:
                    mu_s, mu_b = np.mean(s_px), np.mean(b_px)
                    feat_dict['snr'] = abs(mu_s - mu_b) / (np.std(b_px) + 1e-5)
                    feat_dict['contrast'] = abs(mu_s - mu_b) / (mu_s + mu_b + 1e-5)
                    feat_dict['bg_std'] = np.std(b_px)
                
                # C. 梯度/边缘维度 (Gradient Consistency)
                # 使用局部切片的 Sobel 梯度
                min_r, min_c, max_r, max_c = p.bbox
                crop = img_gray[min_r:max_r, min_c:max_c]
                grad_x = cv2.Sobel(crop, cv2.CV_64F, 1, 0, ksize=3)
                grad_y = cv2.Sobel(crop, cv2.CV_64F, 0, 1, ksize=3)
                feat_dict['grad_mag'] = np.mean(np.sqrt(grad_x**2 + grad_y**2))
                
                # D. 结构复杂度 (Entropy-like)
                hist = np.histogram(s_px, bins=16, range=(0, 255))[0]
                hist = hist / hist.sum()
                feat_dict['entropy'] = -np.sum(hist * np.log2(hist + 1e-7))

            stripes.append({
                'centroid': p.centroid,
                'pts': pts,
                **feat_dict
            })

        # --- 聚类与包络分析 ---
        n = len(stripes)
        if n == 0: return []
        adj = defaultdict(list)
        dist_thresh = diag_len * 0.05
        for i in range(n):
            for j in range(i+1, n):
                d = np.min(np.linalg.norm(stripes[i]['pts'][:,None] - stripes[j]['pts'][None,:], axis=2))
                if d < dist_thresh: adj[i].append(j); adj[j].append(i)
        
        visited, packets = set(), []
        for i in range(n):
            if i not in visited:
                q = deque([i]); visited.add(i); cluster = []
                while q:
                    curr = q.popleft(); cluster.append(curr)
                    for nbr in adj[curr]:
                        if nbr not in visited: visited.add(nbr); q.append(nbr)
                packets.append(cluster)

        results = []
        for c in packets:
            c_stripes = [stripes[idx] for idx in c]
            all_pts = np.vstack([s['pts'] for s in c_stripes])
            rect = cv2.minAreaRect(all_pts.astype(np.float32))
            
            # 计算包络和间距
            area_ratio = (rect[1][0] * rect[1][1]) / (h * w)
            norm_spacing = 0
            if len(c) > 1:
                cents = np.array([s['centroid'] for s in c_stripes])
                d_mat = squareform(pdist(cents))
                np.fill_diagonal(d_mat, np.inf)
                norm_spacing = np.min(d_mat) / diag_len

            # 汇总该波包内所有条纹的特征
            row = {
                'sensor': sensor_name,
                'file_name': os.path.basename(mask_path),
                'stripe_count': len(c),
                'area_ratio': area_ratio,
                'aspect_ratio': max(rect[1]) / (min(rect[1]) + 1e-5),
                'norm_spacing': norm_spacing
            }
            
            # 自动聚合所有在 feat_dict 中定义的指标（取平均值）
            keys_to_avg = ['curvature', 'pixel_skew', 'pixel_kurt', 'pixel_cv', 
                           'snr', 'contrast', 'grad_mag', 'entropy', 'bg_std']
            for k in keys_to_avg:
                vals = [s[k] for s in c_stripes if k in s]
                row[f'mean_{k}'] = np.mean(vals) if vals else 0
                
            results.append(row)
        return results
    except Exception:
        return []

if __name__ == "__main__":
    tasks = []
    for k, v in CONFIG.items():
        if os.path.exists(v['mask_dir']):
            for f in os.listdir(v['mask_dir']):
                if f.lower().endswith('.png'):
                    tasks.append((os.path.join(v['mask_dir'], f), v['img_dir'], v['sensor_name']))
    
    final_data = []
    with ProcessPoolExecutor() as ex:
        res_list = list(tqdm(ex.map(extract_normalized_features, *zip(*tasks)), total=len(tasks)))
        for r in res_list: final_data.extend(r)
    
    pd.DataFrame(final_data).to_csv("internal_wave_feature_mine.csv", index=False)
    print(f"挖掘任务完成! 得到 {len(final_data)} 个波包，特征已保存至 internal_wave_feature_mine.csv")