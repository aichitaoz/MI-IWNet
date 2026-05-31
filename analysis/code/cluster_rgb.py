import os
import cv2
import numpy as np
import pandas as pd
from osgeo import gdal
from skimage import measure, morphology
from scipy.spatial.distance import pdist, squareform
from collections import defaultdict, deque

# ================= 路径配置 =================
SAR_DIR = "/home/xiaobowen/project/internal_wave_detection_project/IW_data/images_rgb"
MASK_DIR = "/home/xiaobowen/project/internal_wave_detection_project/IW_data/masks_rgb"
OUTPUT_CSV = "/home/xiaobowen/project/internal_wave_detection_project/analysis/iw_clusters_geoinfo_rgb.csv"

# ================= 筛选策略参数 =================
MIN_AREA = 50          # 过滤杂波面积
MIN_ASPECT_RATIO = 3.0 # 条纹必须足够细长 (长/宽 > 3)
MIN_BRANCH = 30        # 骨架修剪长度
PROXIMITY_THRESHOLD = 60 # 簇分类距离
DIRECTION_TOLERANCE = 20 # 簇分类方向差异
MIN_STRIPES_PER_CLUSTER = 2 # 【核心修改】每个簇至少包含2条条纹，否则剔除

# ================= 核心算法 =================

def prune_skeleton(skel, min_len=MIN_BRANCH):
    skel = skel.copy()
    kernel = np.array([[1,1,1],[1,10,1],[1,1,1]], np.uint8)
    for _ in range(5):
        counts = cv2.filter2D(skel.astype(np.uint8), -1, kernel)
        endpoints = ((counts - 10) == 1) & (skel > 0)
        if not endpoints.any(): break
        for y, x in np.argwhere(endpoints):
            branch, pos, visited = [(y,x)], (y,x), {(y,x)}
            while True:
                nbrs = [(pos[0]+dy, pos[1]+dx) for dy,dx in [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
                        if 0<=pos[0]+dy<skel.shape[0] and 0<=pos[1]+dx<skel.shape[1] 
                        and skel[pos[0]+dy,pos[1]+dx] and (pos[0]+dy,pos[1]+dx) not in visited]
                if len(nbrs) != 1: break
                pos = nbrs[0]; visited.add(pos); branch.append(pos)
            if len(branch) < min_len:
                for py, px in branch: skel[py,px] = 0
    return skel

def extract_features(region_mask, props):
    # 策略1：长宽比过滤（剔除圆润的噪声）
    major = props.major_axis_length
    minor = props.minor_axis_length
    if minor == 0 or (major / minor) < MIN_ASPECT_RATIO:
        return None

    skel_img = morphology.skeletonize(region_mask > 0)
    pruned = prune_skeleton(skel_img)
    pts = np.column_stack(np.where(pruned))[:, [1, 0]]
    if len(pts) < 5: return None # 过于破碎的骨架也剔除
    
    dists = squareform(pdist(pts))
    i, j = np.unravel_index(dists.argmax(), dists.shape)
    dx, dy = pts[j] - pts[i]
    direction = np.arctan2(dy, dx) * 180 / np.pi
    if direction < 0: direction += 180
    
    return {
        'centroid': pts.mean(axis=0),
        'direction': direction,
        'points': pts,
        'bbox': np.array([props.bbox[1], props.bbox[0], props.bbox[3], props.bbox[2]]),
        'label': props.label
    }

def get_converter(tiff_path):
    ds = gdal.Open(tiff_path)
    if not ds: return None, 0, 0
    meta = ds.GetMetadata()
    min_lon, max_lon = float(meta.get('min_lon', 0)), float(meta.get('max_lon', 0))
    min_lat, max_lat = float(meta.get('min_lat', 0)), float(meta.get('max_lat', 0))
    orbit_dir = meta.get('OrbitDirection', '').upper()
    W, H = ds.RasterXSize, ds.RasterYSize
    
    def to_geo(px, py):
        # 严格补偿轨道反转
        if orbit_dir == 'ASCENDING':
            x_corr, y_corr = px, H - 1 - py
        elif orbit_dir == 'DESCENDING':
            x_corr, y_corr = W - 1 - px, py
        else:
            x_corr, y_corr = px, py
        lon = min_lon + (x_corr / W) * (max_lon - min_lon)
        lat = max_lat - (y_corr / H) * (max_lat - min_lat)
        return lon, lat
    return to_geo

# ================= 主程序 =================

def main():
    results = []
    mask_files = sorted([f for f in os.listdir(MASK_DIR) if f.endswith('.png')])
    
    for mask_name in mask_files:
        base = mask_name.replace('.png', '')
        tiff_path = os.path.join(SAR_DIR, f"{base}.tiff")
        converter = get_converter(tiff_path)
        if not converter: continue
        
        mask_img = cv2.imread(os.path.join(MASK_DIR, mask_name), 0)
        _, binary = cv2.threshold(mask_img, 127, 255, cv2.THRESH_BINARY)
        labeled = measure.label(binary, connectivity=2)
        regions = [r for r in measure.regionprops(labeled) if r.area >= MIN_AREA]
        
        features = []
        for reg in regions:
            feat = extract_features((labeled == reg.label).astype(np.uint8)*255, reg)
            if feat: features.append(feat)
        
        if not features: continue
        
        # 聚类逻辑
        n = len(features)
        graph = defaultdict(list)
        for i in range(n):
            for j in range(i+1, n):
                # 距离与方向双约束
                pts1, pts2 = features[i]['points'][::5], features[j]['points'][::5]
                min_dist = np.min(np.linalg.norm(pts1[:, None] - pts2[None, :], axis=2))
                diff = abs(features[i]['direction'] - features[j]['direction'])
                if min_dist < PROXIMITY_THRESHOLD and min(diff, 180-diff) < DIRECTION_TOLERANCE:
                    graph[i].append(j); graph[j].append(i)
        
        visited, clusters = set(), []
        for start in range(n):
            if start in visited: continue
            cluster, queue = [], deque([start])
            visited.add(start); 
            while queue:
                node = queue.popleft(); cluster.append(node)
                for nb in graph[node]:
                    if nb not in visited: visited.add(nb); queue.append(nb)
            
            # 【筛选层级 3】只有当簇内条纹数 >= 2 时才保留
            if len(cluster) >= MIN_STRIPES_PER_CLUSTER:
                clusters.append(cluster)
        
        # 记录结果
        for cid, indices in enumerate(clusters):
            all_pts = np.vstack([features[idx]['points'] for idx in indices])
            lon_c, lat_c = converter(*all_pts.mean(axis=0))
            rect = cv2.minAreaRect(all_pts.astype(np.float32))
            geo_box = [converter(*p) for p in cv2.boxPoints(rect)]
            
            results.append({
                'FileName': base, 'ClusterID': cid, 'StripeCount': len(indices),
                'Center_Lon': lon_c, 'Center_Lat': lat_c,
                'OBB_P1_Lon': geo_box[0][0], 'OBB_P1_Lat': geo_box[0][1],
                'OBB_P2_Lon': geo_box[1][0], 'OBB_P2_Lat': geo_box[1][1],
                'OBB_P3_Lon': geo_box[2][0], 'OBB_P3_Lat': geo_box[2][1],
                'OBB_P4_Lon': geo_box[3][0], 'OBB_P4_Lat': geo_box[3][1]
            })
        print(f"✓ 处理完成: {base} (保留簇数: {len(clusters)})")

    pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False)
    print(f"\n结果已保存至: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()