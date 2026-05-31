import os
import cv2
import math
import numpy as np
import pandas as pd
from osgeo import gdal
from skimage import measure, morphology
from scipy.spatial.distance import pdist, squareform
from collections import defaultdict, deque

# ================= 1. 配置路径与参数 =================
SAR_DIR = "/root/data/sar"
MASK_DIR = "/root/data/sar1"
OUTPUT_CSV = "/root/data/iw_clusters_geoinfo_sar_new.csv"

# 算法阈值
MIN_AREA = 50          
MIN_BRANCH = 30        
PROXIMITY_THRESHOLD = 60 
DIRECTION_TOLERANCE = 15 

# 源头坐标配置 (用于消除 ±90° 传播方向的歧义，例如 Sibutu Passage)
SRC_LON = 119.5  
SRC_LAT = 4.8    

# ================= 2. 核心条纹分析算法 =================

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
                pos = nbrs[0]
                visited.add(pos)
                branch.append(pos)
            if len(branch) < min_len:
                for py, px in branch: skel[py,px] = 0
    return skel

# 【关键修复】引入 converter，确保计算出的是真实地球上的方向
def extract_features(region_mask, props, converter):
    skel_img = morphology.skeletonize(region_mask > 0)
    pruned = prune_skeleton(skel_img)
    pts = np.column_stack(np.where(pruned))[:, [1, 0]] 
    if len(pts) < 2: return None
    
    dists = squareform(pdist(pts))
    i, j = np.unravel_index(dists.argmax(), dists.shape)
    
    # 提取骨架两端点的像素坐标
    px1, py1 = pts[i]
    px2, py2 = pts[j]
    
    # 转为真实的经纬度坐标
    lon1, lat1 = converter(px1, py1)
    lon2, lat2 = converter(px2, py2)
    
    # 基于地理坐标计算绝对真实的走向
    dx_geo = lon2 - lon1
    dy_geo = lat2 - lat1
    geo_direction = math.degrees(math.atan2(dy_geo, dx_geo)) % 180.0
    
    return {
        'centroid': pts.mean(axis=0),
        'direction': geo_direction, # 现在的 direction 已经是地理意义上的无向角度了
        'points': pts,
        'bbox': np.array([props.bbox[1], props.bbox[0], props.bbox[3], props.bbox[2]]),
        'label': props.label
    }

def build_graph_and_cluster(features):
    n = len(features)
    graph = defaultdict(list)
    
    def compute_prox(f1, f2):
        x_gap = max(0, max(f1['bbox'][0] - f2['bbox'][2], f2['bbox'][0] - f1['bbox'][2]))
        y_gap = max(0, max(f1['bbox'][1] - f2['bbox'][3], f2['bbox'][1] - f1['bbox'][3]))
        pts1 = f1['points'][::max(1, len(f1['points'])//20)]
        pts2 = f2['points'][::max(1, len(f2['points'])//20)]
        min_dist = np.min(np.linalg.norm(pts1[:, None] - pts2[None, :], axis=2))
        return min(np.sqrt(x_gap**2 + y_gap**2), min_dist)

    for i in range(n):
        for j in range(i+1, n):
            diff = abs(features[i]['direction'] - features[j]['direction'])
            dir_diff = min(diff, 180 - diff)
            if compute_prox(features[i], features[j]) < PROXIMITY_THRESHOLD and dir_diff < DIRECTION_TOLERANCE:
                graph[i].append(j)
                graph[j].append(i)
    
    visited, clusters = set(), []
    for start in range(n):
        if start in visited: continue
        cluster, queue = [], deque([start])
        visited.add(start)
        while queue:
            node = queue.popleft()
            cluster.append(node)
            for nb in graph[node]:
                if nb not in visited:
                    visited.add(nb)
                    queue.append(nb)
        clusters.append(cluster)
    return clusters

# ================= 3. 地理坐标转换与数学工具 =================

def get_converter(tiff_path):
    ds = gdal.Open(tiff_path)
    if not ds: return None, 0, 0
    meta = ds.GetMetadata()
    
    min_lon = float(meta.get('min_lon', 0))
    max_lon = float(meta.get('max_lon', 0))
    min_lat = float(meta.get('min_lat', 0))
    max_lat = float(meta.get('max_lat', 0))
    orbit_dir = meta.get('OrbitDirection', '').upper()
    
    W, H = ds.RasterXSize, ds.RasterYSize
    
    def to_geo(px, py):
        if orbit_dir == 'ASCENDING':
            x_corr, y_corr = px, H - 1 - py
        elif orbit_dir == 'DESCENDING':
            x_corr, y_corr = W - 1 - px, py
        else:
            x_corr, y_corr = px, py

        lon = min_lon + (x_corr / W) * (max_lon - min_lon)
        lat = max_lat - (y_corr / H) * (max_lat - min_lat) 
        return lon, lat
    
    return to_geo, W, H

def ang_diff(a, b):
    diff = abs(a - b) % 360
    return diff if diff <= 180 else 360 - diff

# ================= 4. 主执行流程 =================

def main():
    results = []
    mask_files = sorted([f for f in os.listdir(MASK_DIR) if f.endswith('.png')])
    print(f"开始批量处理，共计 {len(mask_files)} 个文件...")

    for mask_name in mask_files:
        base = mask_name.replace('.png', '')
        tiff_path = os.path.join(SAR_DIR, f"{base}.tiff")
        mask_path = os.path.join(MASK_DIR, mask_name)
        
        if not os.path.exists(tiff_path): continue

        converter, _, _ = get_converter(tiff_path)
        if converter is None: continue
        
        mask_img = cv2.imread(mask_path, 0)
        if mask_img is None: continue
        
        _, binary = cv2.threshold(mask_img, 127, 255, cv2.THRESH_BINARY)
        labeled = measure.label(binary, connectivity=2)
        regions = [r for r in measure.regionprops(labeled) if r.area >= MIN_AREA]
        
        features = []
        for reg in regions:
            # 传入 converter！
            feat = extract_features((labeled == reg.label).astype(np.uint8)*255, reg, converter)
            if feat: features.append(feat)
        
        if not features: continue
        
        clusters = build_graph_and_cluster(features)
        
        for cid, indices in enumerate(clusters):
            cluster_pts = np.vstack([features[idx]['points'] for idx in indices])
            
            # 1. 计算地理中心点
            px_c, py_c = cluster_pts.mean(axis=0)
            lon_c, lat_c = converter(px_c, py_c)
            
            # 2. 提取波峰条纹的真实圆周平均走向 (0-180) - 此刻已经是纯地理上的走向了
            stripe_angles = [features[idx]['direction'] for idx in indices]
            sin_sum = sum(math.sin(math.radians(a * 2)) for a in stripe_angles)
            cos_sum = sum(math.cos(math.radians(a * 2)) for a in stripe_angles)
            mean_stripe_dir = (math.degrees(math.atan2(sin_sum, cos_sum)) / 2.0) % 180.0
            
            # 3. 计算垂直于波峰的候选传播方向 (正向或反向)
            cand_a = (mean_stripe_dir + 90.0) % 360.0
            cand_b = (mean_stripe_dir - 90.0) % 360.0
            
            # 4. 计算从源头到波包的物理辐射方向 (只做校验针)
            dlon = lon_c - SRC_LON
            dlat = lat_c - SRC_LAT
            from_src_dir = math.degrees(math.atan2(dlat, dlon)) % 360.0
            
            # 5. 取出符合物理常识的一端
            if ang_diff(cand_a, from_src_dir) <= ang_diff(cand_b, from_src_dir):
                final_propagation_dir = cand_a
            else:
                final_propagation_dir = cand_b
            
            rect = cv2.minAreaRect(cluster_pts.astype(np.float32))
            geo_box = [converter(p[0], p[1]) for p in cv2.boxPoints(rect)]
            
            results.append({
                'FileName': base,
                'ClusterID': cid,
                'StripeCount': len(indices),
                'Center_Lon': lon_c,
                'Center_Lat': lat_c,
                'Mean_Stripe_Dir': mean_stripe_dir,            
                'Propagation_Dir': final_propagation_dir,      
                'From_Source_Dir': from_src_dir,               
                'OBB_P1_Lon': geo_box[0][0], 'OBB_P1_Lat': geo_box[0][1],
                'OBB_P2_Lon': geo_box[1][0], 'OBB_P2_Lat': geo_box[1][1],
                'OBB_P3_Lon': geo_box[2][0], 'OBB_P3_Lat': geo_box[2][1],
                'OBB_P4_Lon': geo_box[3][0], 'OBB_P4_Lat': geo_box[3][1]
            })
            
        print(f"✓ 完成处理: {base} (识别到 {len(clusters)} 个内波群)")

    if results:
        df = pd.DataFrame(results)
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"\n{'='*50}\n地理信息已导出至：{OUTPUT_CSV}\n{'='*50}")

if __name__ == "__main__":
    main()