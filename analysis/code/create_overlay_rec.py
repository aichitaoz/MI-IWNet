import cv2
import numpy as np

def draw_roi_on_tiff(tiff_path, output_path):
    # 1. 定义大图地理范围 (Region A)
    lon_min, lon_max = 112.4, 121.32
    lat_min, lat_max = 18.32, 23.19
    
    # 2. 定义小框四个点的十进制经纬度 (Region B)
    # 按顺序连接：点1 -> 点2 -> 点3 -> 点4
    roi_coords = [
        (114.462206, 20.626392), # Point 1
        (116.879281, 21.055369), # Point 2
        (116.537742, 22.822886), # Point 3
        (114.087064, 22.380981)  # Point 4
    ]

    # 3. 读取 TIFF 原图
    # 注意：如果文件很大，可以用 cv2.IMREAD_UNCHANGED
    img = cv2.imread(tiff_path)
    if img is None:
        print(f"无法读取文件: {tiff_path}")
        return
    
    h, w = img.shape[:2]
    print(f"大图尺寸: {w}x{h}")

    # 4. 经纬度转像素坐标
    pixel_pts = []
    for lon, lat in roi_coords:
        px = int((lon - lon_min) / (lon_max - lon_min) * w)
        # 图像 Y 是从上往下的，所以用 lat_max 减去当前 lat
        py = int((lat_max - lat) / (lat_max - lat_min) * h)
        pixel_pts.append([px, py])
    
    pts = np.array(pixel_pts, np.int32)
    pts = pts.reshape((-1, 1, 2))

    # 5. 在图上画框 (亮红色, 线宽根据图的大小自动调整)
    thickness = max(2, int(w / 1000)) 
    cv2.polylines(img, [pts], isClosed=True, color=(0, 0, 255), thickness=thickness)

    # 6. 保存结果
    cv2.imwrite(output_path, img)
    print(f"✅ 框图已保存至: {output_path}")

# 运行
input_tiff = "/home/xiaobowen/project/internal_wave_detection_project/analysis/code/MODIS_TrueColor_2022-05-28_2Aqua.tiff"
output_jpg = "/home/xiaobowen/project/internal_wave_detection_project/analysis/MODIS_ROI_Visual.jpg"

draw_roi_on_tiff(input_tiff, output_jpg)