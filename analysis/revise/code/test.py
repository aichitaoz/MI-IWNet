import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure, morphology
from scipy.spatial.distance import pdist, squareform
from collections import defaultdict, deque
import seaborn as sns

# ================= 配置路径 =================
MASK_DIR = "/root/data/mask_rgb"  # 你的SAR Mask路径
MIN_AREA = 50
MIN_BRANCH = 30

# 测试的超参数网格
TEST_DISTS = [20, 40, 60, 80, 100]
TEST_ANGS = [5, 10, 15, 20, 30]

# ================= 复用你的核心提取算法 =================
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
    skel_img = morphology.skeletonize(region_mask > 0)
    pruned = prune_skeleton(skel_img)
    pts = np.column_stack(np.where(pruned))[:, [1, 0]]
    if len(pts) < 2: return None
    
    dists = squareform(pdist(pts))
    i, j = np.unravel_index(dists.argmax(), dists.shape)
    dx, dy = pts[j] - pts[i]
    direction = np.arctan2(dy, dx) * 180 / np.pi
    if direction < 0: direction += 180
    return {'direction': direction, 'points': pts, 
            'bbox': np.array([props.bbox[1], props.bbox[0], props.bbox[3], props.bbox[2]])}

def build_graph_and_cluster(features, prox_thresh, dir_thresh):
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
            if compute_prox(features[i], features[j]) < prox_thresh and dir_diff < dir_thresh:
                graph[i].append(j); graph[j].append(i)
                
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
                    visited.add(nb); queue.append(nb)
        clusters.append(cluster)
    return clusters

# ================= 主执行：统计不同阈值下的结果 =================
def run_ablation():
    mask_files = sorted([f for f in os.listdir(MASK_DIR) if f.endswith('.png')])[:3000] # 取前100张图跑统计即可
    print(f"Running ablation study on {len(mask_files)} images...")

    # 预先提取所有图像的特征（避免重复计算，加速运算）
    all_image_features = []
    for f in mask_files:
        img = cv2.imread(os.path.join(MASK_DIR, f), 0)
        _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        labeled = measure.label(binary, connectivity=2)
        regions = [r for r in measure.regionprops(labeled) if r.area >= MIN_AREA]
        
        feats = []
        for reg in regions:
            feat = extract_features((labeled == reg.label).astype(np.uint8)*255, reg)
            if feat: feats.append(feat)
        if feats:
            all_image_features.append(feats)

    # 记录结果
    results_dist = []
    results_ang = []

    # 1. 固定 Angle = 15，遍历 Distance
    fixed_ang = 15
    for dist in TEST_DISTS:
        total_clusters = 0
        total_valid_clusters = 0
        for feats in all_image_features:
            clusters = build_graph_and_cluster(feats, dist, fixed_ang)
            total_clusters += len(clusters)
            # 记录包含 >=2 条纹的簇数（有效波包）
            total_valid_clusters += sum(1 for c in clusters if len(c) >= 2)
        
        avg_clusters = total_clusters / len(all_image_features)
        results_dist.append((dist, avg_clusters))
        print(f"Dist={dist}, Ang={fixed_ang} -> Avg Clusters: {avg_clusters:.2f}")

    # 2. 固定 Distance = 60，遍历 Angle
    fixed_dist = 60
    for ang in TEST_ANGS:
        stripes_in_cluster = []
        for feats in all_image_features:
            clusters = build_graph_and_cluster(feats, fixed_dist, ang)
            for c in clusters:
                if len(c) >= 2: # 只统计有效波包内的条纹数
                    stripes_in_cluster.append(len(c))
        
        avg_stripes = np.mean(stripes_in_cluster) if stripes_in_cluster else 0
        results_ang.append((ang, avg_stripes))
        print(f"Dist={fixed_dist}, Ang={ang} -> Avg Stripes per Packet: {avg_stripes:.2f}")

    # ================= 绘图 =================
    sns.set_theme(style="whitegrid")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # 图1：Distance 敏感性
    dists, avg_cls = zip(*results_dist)
    ax1.plot(dists, avg_cls, marker='o', linewidth=2.5, color='#E64B35', markersize=8)
    ax1.axvline(x=60, color='gray', linestyle='--', alpha=0.7)
    ax1.set_xlabel('Proximity Threshold (pixels)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Average Number of Clusters / Image', fontsize=12, fontweight='bold')
    ax1.set_title(f'Ablation on Distance (Fixed Angle={fixed_ang}°)', fontsize=14)
    ax1.text(62, max(avg_cls)*0.9, 'Optimal Threshold (60)', color='#333333', fontweight='bold')

    # 图2：Angle 敏感性
    angs, avg_strps = zip(*results_ang)
    ax2.plot(angs, avg_strps, marker='s', linewidth=2.5, color='#3C5488', markersize=8)
    ax2.axvline(x=15, color='gray', linestyle='--', alpha=0.7)
    ax2.set_xlabel('Direction Tolerance (°)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Average Stripes per Packet', fontsize=12, fontweight='bold')
    ax2.set_title(f'Ablation on Angle (Fixed Dist={fixed_dist} px)', fontsize=14)
    ax2.text(16, min(avg_strps) + (max(avg_strps)-min(avg_strps))*0.1, 'Optimal Threshold (15°)', color='#333333', fontweight='bold')

    plt.tight_layout()
    plt.savefig('Hyperparameter_Ablation.png', dpi=300)
    print("✓ 已生成支撑图表：Hyperparameter_Ablation.png")

if __name__ == "__main__":
    run_ablation()