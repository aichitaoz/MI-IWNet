import cv2
import numpy as np
import os
import glob
from scipy.stats import skew

def calculate_dataset_metrics(folder_path, extension="*.png", use_morph_close=False):
    search_pattern = os.path.join(folder_path, extension)
    image_paths = glob.glob(search_pattern)
    
    if not image_paths:
        print(f"错误: 找不到图片，请检查路径: {folder_path}")
        return

    print(f"开始处理，共找到 {len(image_paths)} 张图片...")
    all_refined_areas = []

    # 如果开启形态学缝合，定义一个 3x3 或 5x5 的卷积核
    # 核越大，能缝合的断裂间隙就越大
    kernel = np.ones((5, 5), np.uint8)

    for idx, img_path in enumerate(image_paths):
        # 以灰度模式读取
        mask = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        
        # 【修复 1：强制鲁棒二值化】
        # 无论 mask 是 0/1 还是 0/255，只要大于 0 的全变成 255
        binary_mask = np.where(mask > 0, 255, 0).astype(np.uint8)

        # 【修复 2：缝合断裂的内波条纹】
        if use_morph_close:
            # 闭运算：先膨胀后腐蚀，连接断开的邻近碎片，防止一个内波被算成多个
            binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)

        # 进行连通域分析
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
        
        if num_labels > 1: 
            # 提取面积，stats[0] 永远是背景，所以从 1 开始切片
            areas = stats[1:, cv2.CC_STAT_AREA]
            
            # 过滤面积
            valid_areas = areas[(areas >= 50) & (areas <= 5000)]
            all_refined_areas.extend(valid_areas)

        if (idx + 1) % 500 == 0 or (idx + 1) == len(image_paths):
            print(f"  已处理: {idx + 1} / {len(image_paths)}")

    all_refined_areas = np.array(all_refined_areas)
    count = len(all_refined_areas)

    if count == 0:
        print("\n计算结果为 0！可能是因为没有波包落在这个面积区间，或者 mask 都是全黑的。")
        return

    # 区间统计
    bins = [50, 200, 500, 1000, 2000, 5001] 
    hist, _ = np.histogram(all_refined_areas, bins=bins)
    percentages = (hist / count) * 100

    # 特征计算
    med_area = np.median(all_refined_areas)
    mean_area = np.mean(all_refined_areas)
    skew_val = skew(all_refined_areas)

    print(f"\n========== 整个数据集统计结果 ==========")
    print(f"Count (内波组件总数) : {count:,}")
    print("-" * 38)
    print("Refined Area Distribution (%):")
    print(f"  50–200 px  : {percentages[0]:.1f}")
    print(f"  200–500 px : {percentages[1]:.1f}")
    print(f"  500–1k px  : {percentages[2]:.1f}")
    print(f"  1k–2k px   : {percentages[3]:.1f}")
    print(f"  2k–5k px   : {percentages[4]:.1f}")
    print("-" * 38)
    print("Descriptive Stats (px):")
    print(f"  Med. (中位数): {med_area:.0f}")
    print(f"  Mean (均值)  : {mean_area:.1f}")
    print(f"  Skew.(偏度)  : {skew_val:.2f}")
    print("========================================")

if __name__ == "__main__":
    dataset_folder_path = '/root/data/mask_sar' 
    
    # 💡 关键点：
    # 如果你发现原代码算出来的 Count 大得不正常，请把 use_morph_close 设为 True
    calculate_dataset_metrics(dataset_folder_path, extension="*.png", use_morph_close=False)