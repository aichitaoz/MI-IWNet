import cv2
import numpy as np
from skimage import measure, morphology
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
from collections import defaultdict, deque

# ==================== 1. 配置与初始化 ====================
MASK_DIR = "/home/xiaobowen/project/internal_wave_detection_project/IW_data/test_masks"
IMAGE_DIR = "/home/xiaobowen/project/internal_wave_detection_project/IW_data/test_images"
OUTPUT_DIR = "./output_cluster"
FONT_PATH = "/home/xiaobowen/project/internal_wave_detection_project/analysis/code/TIMES.TTF"

os.makedirs(OUTPUT_DIR, exist_ok=True)

try:
    prop = fm.FontProperties(fname=FONT_PATH)
    plt.rcParams['font.family'] = prop.get_name()
    print(f"✓ Loaded font: {FONT_PATH}")
except:
    print("⚠ Font not found, using default sans-serif.")
    plt.rcParams['font.family'] = 'sans-serif'

MIN_AREA = 50

plt.rcParams['axes.linewidth'] = 0
plt.rcParams['xtick.bottom'] = False
plt.rcParams['ytick.left'] = False

# ==================== 2. 核心算法 ====================
def prune_skeleton(skel, min_len=30):
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
    skel = prune_skeleton(skel_img)
    pts = np.column_stack(np.where(skel))[:, [1,0]]
    if len(pts) < 5: return None
    
    dists = squareform(pdist(pts))
    i, j = np.unravel_index(dists.argmax(), dists.shape)
    p_start, p_end = pts[i], pts[j]
    
    tangent_dx, tangent_dy = p_end[0] - p_start[0], p_end[1] - p_start[1]
    normal_dx, normal_dy = -tangent_dy, tangent_dx 
    
    chord_midpoint = (p_start + p_end) / 2
    actual_centroid = pts.mean(axis=0)
    bulge_dx, bulge_dy = actual_centroid[0] - chord_midpoint[0], actual_centroid[1] - chord_midpoint[1]
    
    if normal_dx * bulge_dx + normal_dy * bulge_dy < 0:
        normal_dx, normal_dy = -normal_dx, -normal_dy
        
    direction = np.arctan2(normal_dy, normal_dx) * 180 / np.pi
    if direction < 0: direction += 360
    
    return {'centroid': actual_centroid, 'direction': direction, 'points': pts,
            'bbox': np.array([props.bbox[1], props.bbox[0], props.bbox[3], props.bbox[2]]), 'label': props.label}

def build_graph_and_cluster(features, proximity_threshold, direction_tolerance):
    n = len(features)
    graph = defaultdict(list)
    for i in range(n):
        for j in range(i+1, n):
            f1, f2 = features[i], features[j]
            x_gap = max(0, max(f1['bbox'][0] - f2['bbox'][2], f2['bbox'][0] - f1['bbox'][2]))
            y_gap = max(0, max(f1['bbox'][1] - f2['bbox'][3], f2['bbox'][1] - f1['bbox'][3]))
            dist = np.sqrt(x_gap**2 + y_gap**2)
            diff = abs(f1['direction'] - f2['direction'])
            angle_diff = min(diff, 360 - diff)
            effective_angle_diff = min(angle_diff, 180 - angle_diff)
            
            if dist < proximity_threshold and effective_angle_diff < direction_tolerance:
                graph[i].append(j); graph[j].append(i)
                
    visited, clusters = set(), []
    for start in range(n):
        if start in visited: continue
        c, q = [], deque([start]); visited.add(start)
        while q:
            u = q.popleft(); c.append(u)
            for v in graph[u]:
                if v not in visited: visited.add(v); q.append(v)
        clusters.append(c)
    return graph, clusters

def compute_cluster_obb(cluster, features):
    pts = np.vstack([features[i]['points'] for i in cluster])
    rect = cv2.minAreaRect(pts.astype(np.float32))
    return cv2.boxPoints(rect).astype(int)

# ==================== 3. 独立绘图函数 ====================

def save_single_plot(fig, output_path):
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0,0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

def process_paper_figures(img, clusters, features, graph, basename, labels_img, sub_dir):
    save_dir = os.path.join(OUTPUT_DIR, sub_dir, basename)
    os.makedirs(save_dir, exist_ok=True)
    
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    orig_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    high_impact_colors = ['#FF0000', '#00FFFF', '#FFFF00', '#FF00FF']

    # --- Figure 1: Skeletonization ---
    fig1 = plt.figure(figsize=(6, 8))
    plt.imshow(gray_img, cmap='gray')
    for feat in features:
        pts = feat['points']
        plt.scatter(pts[:,0], pts[:,1], color='red', s=15, edgecolors='none')
    plt.axis('off')
    save_single_plot(fig1, os.path.join(save_dir, '01_skeleton.pdf'))

    # --- Figure 2: Physics (Propagation Vectors) ---
    fig2 = plt.figure(figsize=(6, 8))
    plt.gca().set_facecolor('white')
    plt.xlim(0, img.shape[1])
    plt.ylim(img.shape[0], 0)
    plt.gca().set_aspect('equal')

    CUSTOM_BLUE = '#3C5D7B'

    for feat in features:
        cx, cy = feat['centroid']
        pts = feat['points']
        
        dists = np.linalg.norm(pts - np.array([cx, cy]), axis=1)
        closest_idx = np.argmin(dists)
        anchor_x, anchor_y = pts[closest_idx]

        ang = np.deg2rad(feat['direction'])
        
        plt.scatter(pts[:,0], pts[:,1], color=CUSTOM_BLUE, s=15, edgecolors='none', alpha=1.0)
        plt.arrow(anchor_x, anchor_y, 20*np.cos(ang), 20*np.sin(ang), 
                  color='red', head_width=6, lw=2.0, zorder=5)
                  
    plt.axis('off')
    save_single_plot(fig2, os.path.join(save_dir, '02_propagation.pdf'))

    # --- Figure 3: Graph Connectivity ---
    fig3 = plt.figure(figsize=(6, 8))
    plt.gca().set_facecolor('white')
    plt.xlim(0, img.shape[1])
    plt.ylim(img.shape[0], 0)
    plt.gca().set_aspect('equal')
    
    # 画线：底层连接依然保留，展示图的结构
    for i in range(len(features)):
        cx, cy = features[i]['centroid']
        for j in graph[i]:
            if i < j:
                px, py = features[j]['centroid']
                plt.plot([cx, px], [cy, py], color='gray', lw=1.5, ls='--', alpha=0.6)
    
    # 画点：只为有效聚类（多于1条线的）画高亮的大圆点
    for c_idx, cluster in enumerate(clusters):
        c_color = high_impact_colors[c_idx % len(high_impact_colors)]
        for i in cluster:
            cx, cy = features[i]['centroid']
            plt.scatter(cx, cy, c=[c_color], s=150, edgecolors='black', zorder=5)
            
    plt.axis('off')
    save_single_plot(fig3, os.path.join(save_dir, '03_graph.pdf'))

    # --- Figure 4: Final Result (Instance Segmentation Overlay) ---
    fig4 = plt.figure(figsize=(6, 8))
    
    plt.imshow(orig_rgb)
    
    h, w = img.shape[:2]
    final_overlay = np.zeros((h, w, 4), dtype=np.uint8)
    
    # 只有有效聚类才会被加上彩色蒙版和外接矩形框
    for c_idx, cluster in enumerate(clusters):
        c_hex = high_impact_colors[c_idx % len(high_impact_colors)]
        c_hex = c_hex.lstrip('#')
        c_rgb = [int(c_hex[i:i+2], 16) for i in (0, 2, 4)]
        c_rgba = c_rgb + [120] 

        for i in cluster:
            lbl_id = features[i]['label']
            mask_indices = (labels_img == lbl_id)
            final_overlay[mask_indices] = c_rgba

        obb = compute_cluster_obb(cluster, features)
        obb_poly = np.vstack([obb, obb[0]])
        plt.plot(obb_poly[:,0], obb_poly[:,1], color='#'+c_hex, lw=4)

    plt.imshow(final_overlay)
    plt.axis('off')
    save_single_plot(fig4, os.path.join(save_dir, '04_result.pdf'))


# ==================== 主程序 ====================
if __name__ == "__main__":
    files = sorted([f for f in os.listdir(MASK_DIR) if f.endswith('.png')])
    
    # 你指定的 5 组十字交叉消融配置 (dist, ang)
    configs = [
        (40, 15),
        (60, 10),
        (60, 15), # 基准点 Baseline
        (60, 20),
        (80, 15)
    ]
    
    print(f"Starting specific config processing for {len(files)} files...")
    
    for t_dist, t_ang in configs:
        sub_dir_name = f"dist{t_dist}_ang{t_ang}"
        print(f"\n--- Running Config: Tau_dist={t_dist}, Tau_ang={t_ang} ---")
        
        for f in files:
            try:
                mask_path = os.path.join(MASK_DIR, f)
                basename = os.path.splitext(f)[0]
                img_path = os.path.join(IMAGE_DIR, basename + ".jpg")
                
                mask = cv2.imread(mask_path, 0)
                if mask is None: 
                    print(f"Warning: Could not read mask {f}")
                    continue
                    
                original_img = cv2.imread(img_path)
                if original_img is None: 
                    original_img = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                
                _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
                labels = measure.label(binary, connectivity=2)
                regions = [p for p in measure.regionprops(labels) if p.area >= MIN_AREA]
                
                features = []
                for reg in regions:
                    feat = extract_features((labels == reg.label).astype(np.uint8)*255, reg)
                    if feat: features.append(feat)
                
                if features:
                    graph, clusters = build_graph_and_cluster(features, t_dist, t_ang)
                    
                    # =============== 【关键修改点】 ===============
                    # 过滤掉只有一条线的孤立类，只保留有多条线拼接的波浪簇
                    valid_clusters = [c for c in clusters if len(c) > 1]
                    # ============================================
                    
                    process_paper_figures(original_img, valid_clusters, features, graph, basename, labels, sub_dir_name)
                    
            except Exception as e:
                print(f"Error on {f} with config {sub_dir_name}: {e}")