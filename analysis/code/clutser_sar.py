import os
import cv2
import numpy as np
import pandas as pd
import glob
from osgeo import gdal
from skimage import measure, morphology
from scipy.spatial.distance import pdist, squareform
from collections import defaultdict, deque

# ================= 1. 配置路径与参数 =================
SAR_DIR = "/home/xiaobowen/project/internal_wave_detection_project/analysis/sar"
MASK_DIR = "/home/xiaobowen/project/internal_wave_detection_project/analysis/sar_masks"
OUTPUT_CSV = "/home/xiaobowen/project/internal_wave_detection_project/analysis/iw_clusters_geoinfo_sar.csv"

# 算法阈值（可根据需要微调）
MIN_AREA = 50          # 过滤杂波的最小面积
MIN_BRANCH = 30        # 骨架化修剪的最小分支长度
PROXIMITY_THRESHOLD = 60 # 簇分类的邻近像素阈值
DIRECTION_TOLERANCE = 20 # 簇分类的方向一致性阈值（度）

# ================= 2. 核心条纹分析算法 =================

def prune_skeleton(skel, min_len=MIN_BRANCH):
    """修剪骨架化后的短毛刺，保证主干提取的严谨性"""
    skel = skel.copy()
    kernel = np.array([[1,1,1],[1,10,1],[1,1,1]], np.uint8)
    for _ in range(5):
        counts = cv2.filter2D(skel.astype(np.uint8), -1, kernel)
        endpoints = ((counts - 10) == 1) & (skel > 0)
        if not endpoints.any(): break
        for y, x in np.argwhere(endpoints):
            branch, pos, visited = [(y,x)], (y,x), {(y,x)}
            while True:
                # 8邻域搜索
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

def extract_features(region_mask, props):
    """提取单个条纹的几何中心、方向和像素点集"""
    skel_img = morphology.skeletonize(region_mask > 0)
    pruned = prune_skeleton(skel_img)
    pts = np.column_stack(np.where(pruned))[:, [1, 0]] # 转为 [x, y]
    if len(pts) < 2: return None
    
    # 使用骨架最远两点计算主方向
    dists = squareform(pdist(pts))
    i, j = np.unravel_index(dists.argmax(), dists.shape)
    dx, dy = pts[j] - pts[i]
    direction = np.arctan2(dy, dx) * 180 / np.pi
    if direction < 0: direction += 180
    
    return {
        'centroid': pts.mean(axis=0),
        'direction': direction,
        'points': pts,
        'bbox': np.array([props.bbox[1], props.bbox[0], props.bbox[3], props.bbox[2]]), # [min_x, min_y, max_x, max_y]
        'label': props.label
    }

def build_graph_and_cluster(features):
    """基于距离和方向一致性构建连通图并进行簇分类"""
    n = len(features)
    graph = defaultdict(list)
    
    def compute_prox(f1, f2):
        # 综合考虑BBox间隙和点集最小距离
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

# ================= 3. 严谨地理坐标转换逻辑 =================

def get_converter(tiff_path):
    """读取TIFF元数据并根据轨道方向创建坐标转换器"""
    ds = gdal.Open(tiff_path)
    if not ds: return None, 0, 0
    meta = ds.GetMetadata()
    
    # 从Tags读取地理范围真值
    min_lon = float(meta.get('min_lon', 0))
    max_lon = float(meta.get('max_lon', 0))
    min_lat = float(meta.get('min_lat', 0))
    max_lat = float(meta.get('max_lat', 0))
    orbit_dir = meta.get('OrbitDirection', '').upper()
    
    W, H = ds.RasterXSize, ds.RasterYSize
    
    def to_geo(px, py):
        """将Mask像素坐标转换回真实经纬度，补偿轨道导致的镜像反转"""
        # 逆转‘North-up’校正逻辑以对应TIFF原始元数据
        if orbit_dir == 'ASCENDING':
            # 升轨在原图中通常是上下颠倒的，还原 y 轴映射
            x_corr, y_corr = px, H - 1 - py
        elif orbit_dir == 'DESCENDING':
            # 降轨在原图中通常是左右镜像的，还原 x 轴映射
            x_corr, y_corr = W - 1 - px, py
        else:
            x_corr, y_corr = px, py

        # 线性插值计算经纬度
        lon = min_lon + (x_corr / W) * (max_lon - min_lon)
        lat = max_lat - (y_corr / H) * (max_lat - min_lat) # 维度从上往下减小
        return lon, lat
    
    return to_geo, W, H

# ================= 4. 主执行流程 =================

def main():
    results = []
    # 获取并排序所有Mask文件
    mask_files = sorted([f for f in os.listdir(MASK_DIR) if f.endswith('.png')])
    print(f"开始批量处理，共计 {len(mask_files)} 个文件...")

    for mask_name in mask_files:
        base = mask_name.replace('.png', '')
        tiff_path = os.path.join(SAR_DIR, f"{base}.tiff")
        mask_path = os.path.join(MASK_DIR, mask_name)
        
        if not os.path.exists(tiff_path):
            print(f"警告：未找到对应的 TIFF 文件，跳过 {base}")
            continue

        # 初始化地理转换器
        converter, _, _ = get_converter(tiff_path)
        if converter is None: continue
        
        # 加载并分析 Mask
        mask_img = cv2.imread(mask_path, 0)
        if mask_img is None: continue
        
        _, binary = cv2.threshold(mask_img, 127, 255, cv2.THRESH_BINARY)
        labeled = measure.label(binary, connectivity=2)
        regions = [r for r in measure.regionprops(labeled) if r.area >= MIN_AREA]
        
        features = []
        for reg in regions:
            feat = extract_features((labeled == reg.label).astype(np.uint8)*255, reg)
            if feat: features.append(feat)
        
        if not features: continue
        
        # 条纹聚类
        clusters = build_graph_and_cluster(features)
        
        # 遍历每个簇并转换坐标
        for cid, indices in enumerate(clusters):
            # 获取簇内所有像素点
            cluster_pts = np.vstack([features[idx]['points'] for idx in indices])
            
            # 1. 计算地理中心点
            px_c, py_c = cluster_pts.mean(axis=0)
            lon_c, lat_c = converter(px_c, py_c)
            
            # 2. 计算最小外接矩形(OBB)并转为地理坐标
            # OBB 对计算内波群长度、走向至关重要
            rect = cv2.minAreaRect(cluster_pts.astype(np.float32))
            box_pixels = cv2.boxPoints(rect)
            geo_box = [converter(p[0], p[1]) for p in box_pixels]
            
            # 保存结果到列表
            results.append({
                'FileName': base,
                'ClusterID': cid,
                'StripeCount': len(indices),
                'Center_Lon': lon_c,
                'Center_Lat': lat_c,
                'OBB_P1_Lon': geo_box[0][0], 'OBB_P1_Lat': geo_box[0][1],
                'OBB_P2_Lon': geo_box[1][0], 'OBB_P2_Lat': geo_box[1][1],
                'OBB_P3_Lon': geo_box[2][0], 'OBB_P3_Lat': geo_box[2][1],
                'OBB_P4_Lon': geo_box[3][0], 'OBB_P4_Lat': geo_box[3][1]
            })
        
        print(f"✓ 完成处理: {base} (识别到 {len(clusters)} 个内波群)")

    # 结果持久化
    if results:
        df = pd.DataFrame(results)
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"\n{'='*50}\n处理成功！地理信息已导出至：\n{OUTPUT_CSV}\n{'='*50}")
    else:
        print("未识别到任何符合条件的内波条纹。")

if __name__ == "__main__":
    main()