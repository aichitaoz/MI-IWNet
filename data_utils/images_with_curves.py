"""
SDG多光谱图像曲线可视化工具
处理7通道TIFF图像，并在假彩色合成图上绘制贝塞尔曲线
"""

import os
import json
import numpy as np
import cv2
import math
import warnings
warnings.filterwarnings('ignore')

# 导入tifffile处理多通道TIFF
try:
    import tifffile
    HAS_TIFFFILE = True
except ImportError:
    HAS_TIFFFILE = False
    print("⚠️  Warning: tifffile not installed. Install with: pip install tifffile")


# 配置路径
DATA_ROOT = '/home/xiaobowen/project/internal_wave_detection_project/IW_data'
IMG_DIR = os.path.join(DATA_ROOT, 'images_sdg')
LABEL_DIR = os.path.join(DATA_ROOT, 'annotation_sdg')
VISUAL_SAVE_DIR = os.path.join(DATA_ROOT, 'images_with_curves_sdg')

# 可视化参数
LINE_THICKNESS = 3
LINE_COLOR = (0, 255, 255)  # 黄色 BGR (在RGB上显示为青色)


def bezier_curve(points, n_points=200):
    """
    生成贝塞尔曲线点
    Args:
        points: 控制点数组 [(x1,y1), (x2,y2), ...]
        n_points: 生成的曲线点数量
    Returns:
        curve_points: 曲线点数组
    """
    t = np.linspace(0, 1, n_points)[:, None]
    n = len(points) - 1
    curve = np.zeros((n_points, 2))
    
    for i in range(n + 1):
        binomial = math.comb(n, i)
        coef = binomial * ((1 - t) ** (n - i)) * (t ** i)
        curve += coef * points[i]
    
    return curve


def load_multispectral_image(image_path, composite_type='false_color'):
    """
    加载多光谱SDG图像并生成RGB合成图
    Args:
        image_path: TIFF图像路径
        composite_type: 合成类型
            - 'false_color': 假彩色 (NIR2-Red-Green)
            - 'natural': 自然色 (Red-Green-Blue)
            - 'nir': 近红外合成 (NIR2-NIR1-Red)
    Returns:
        rgb_image: 8-bit RGB图像 (H, W, 3)
    """
    if not HAS_TIFFFILE:
        raise ImportError("tifffile is required. Install with: pip install tifffile")
    
    # 加载7通道图像
    image = tifffile.imread(image_path).astype(np.float32)
    
    # 归一化到0-1
    if image.max() > 1.0:
        image = image / 255.0
    
    # SDG波段索引: 0:Coastal, 1:Blue, 2:Green, 3:Red, 4:RE1, 5:NIR1, 6:NIR2
    
    if composite_type == 'false_color':
        # 假彩色合成 (NIR2, Red, Green) -> (R, G, B)
        rgb = np.stack([
            image[:, :, 6],  # NIR2 -> R
            image[:, :, 3],  # Red -> G
            image[:, :, 2]   # Green -> B
        ], axis=-1)
    
    elif composite_type == 'natural':
        # 自然色合成 (Red, Green, Blue)
        rgb = np.stack([
            image[:, :, 3],  # Red -> R
            image[:, :, 2],  # Green -> G
            image[:, :, 1]   # Blue -> B
        ], axis=-1)
    
    elif composite_type == 'nir':
        # 近红外合成 (NIR2, NIR1, Red)
        rgb = np.stack([
            image[:, :, 6],  # NIR2 -> R
            image[:, :, 5],  # NIR1 -> G
            image[:, :, 3]   # Red -> B
        ], axis=-1)
    
    else:
        raise ValueError(f"Unknown composite_type: {composite_type}")
    
    # 对比度拉伸 (2%线性拉伸)
    rgb_stretched = np.zeros_like(rgb)
    for i in range(3):
        channel = rgb[:, :, i]
        p2, p98 = np.percentile(channel, (2, 98))
        rgb_stretched[:, :, i] = np.clip((channel - p2) / (p98 - p2 + 1e-8), 0, 1)
    
    # 转换为8-bit BGR (OpenCV格式)
    rgb_8bit = (rgb_stretched * 255).astype(np.uint8)
    bgr_8bit = cv2.cvtColor(rgb_8bit, cv2.COLOR_RGB2BGR)
    
    return bgr_8bit


def overlay_curves_on_sdg_image(image_path, label_path, 
                                 composite_type='false_color',
                                 thickness=LINE_THICKNESS, 
                                 color=LINE_COLOR):
    """
    在SDG多光谱图像上叠加贝塞尔曲线
    Args:
        image_path: SDG TIFF图像路径
        label_path: JSON标注文件路径
        composite_type: RGB合成类型
        thickness: 线条粗细
        color: 线条颜色 (BGR格式)
    Returns:
        image_with_curves: 叠加曲线后的图像
    """
    # 加载多光谱图像并生成RGB合成图
    try:
        image = load_multispectral_image(image_path, composite_type=composite_type)
        print(f"  ✅ 加载图像: {image.shape}")
    except Exception as e:
        print(f"  ⚠️ 无法读取图像: {image_path}")
        print(f"     错误: {e}")
        return None
    
    # 加载标注曲线
    if not os.path.exists(label_path):
        print(f"  ⚠️ 标注文件不存在: {label_path}")
        return None
    
    with open(label_path, 'r') as f:
        data = json.load(f)
    
    control_points_list = data.get('curves', [])
    print(f"  📍 找到 {len(control_points_list)} 条曲线")
    
    # 在图像上绘制贝塞尔曲线
    for idx, control_points_pixels in enumerate(control_points_list):
        control_points = np.array(control_points_pixels)
        
        # 生成贝塞尔曲线
        curve_points = bezier_curve(control_points, n_points=200).astype(np.int32)
        
        # 绘制曲线
        for i in range(len(curve_points) - 1):
            pt1 = tuple(curve_points[i])
            pt2 = tuple(curve_points[i + 1])
            cv2.line(image, pt1, pt2, color=color, thickness=thickness)
        
        # 绘制控制点 (可选)
        for cp in control_points:
            cv2.circle(image, tuple(cp.astype(int)), radius=5, 
                      color=(255, 0, 0), thickness=-1)  # 蓝色控制点
    
    return image


def create_comparison_image(image_path, label_path, save_path):
    """
    创建多种合成方式的对比图
    Args:
        image_path: SDG图像路径
        label_path: 标注路径
        save_path: 保存路径
    """
    composite_types = ['false_color', 'natural', 'nir']
    titles = ['False Color (NIR2-R-G)', 'Natural Color (R-G-B)', 'NIR Composite (NIR2-NIR1-R)']
    
    images = []
    for comp_type in composite_types:
        img = overlay_curves_on_sdg_image(image_path, label_path, 
                                          composite_type=comp_type)
        if img is not None:
            # 添加标题
            img_with_title = cv2.copyMakeBorder(img, 50, 0, 0, 0, 
                                               cv2.BORDER_CONSTANT, 
                                               value=(0, 0, 0))
            images.append(img_with_title)
    
    if len(images) > 0:
        # 横向拼接
        combined = np.hstack(images)
        cv2.imwrite(save_path, combined)
        print(f"  ✅ 对比图保存: {save_path}")


def process_all_visualizations(label_dir=LABEL_DIR, 
                               img_dir=IMG_DIR, 
                               save_dir=VISUAL_SAVE_DIR,
                               composite_type='false_color',
                               create_comparison=False):
    """
    批量处理所有标注文件，生成可视化结果
    Args:
        label_dir: 标注目录
        img_dir: 图像目录
        save_dir: 保存目录
        composite_type: 合成类型
        create_comparison: 是否创建对比图
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"✅ 创建保存目录: {save_dir}")
    
    label_files = [f for f in os.listdir(label_dir) if f.endswith('.json')]
    print(f'\n共找到 {len(label_files)} 个标注文件')
    print('='*60)
    
    success_count = 0
    fail_count = 0
    
    for lf in label_files:
        print(f'\n处理标注文件: {lf}')
        label_path = os.path.join(label_dir, lf)
        
        # SDG图像为TIFF格式
        image_name = os.path.splitext(lf)[0] + '.tif'
        image_path = os.path.join(img_dir, image_name)
        
        if not os.path.exists(image_path):
            # 尝试.tiff扩展名
            image_name = os.path.splitext(lf)[0] + '.tiff'
            image_path = os.path.join(img_dir, image_name)
        
        if not os.path.exists(image_path):
            print(f'  ❌ 图像文件不存在: {image_name}')
            fail_count += 1
            continue
        
        # 生成单一合成图
        output_image = overlay_curves_on_sdg_image(image_path, label_path, 
                                                   composite_type=composite_type)
        
        if output_image is not None:
            save_path = os.path.join(save_dir, 
                                    os.path.splitext(lf)[0] + f'_{composite_type}.jpg')
            cv2.imwrite(save_path, output_image)
            print(f'  ✅ 可视化保存: {save_path}')
            success_count += 1
            
            # 可选：创建对比图
            if create_comparison:
                comparison_path = os.path.join(save_dir, 
                                              os.path.splitext(lf)[0] + '_comparison.jpg')
                create_comparison_image(image_path, label_path, comparison_path)
        else:
            print(f'  ❌ 处理失败')
            fail_count += 1
    
    print('\n' + '='*60)
    print(f'✅ 处理完成: 成功 {success_count} 个, 失败 {fail_count} 个')
    print(f'📁 结果保存在: {save_dir}')


def main():
    """主函数"""
    print("SDG多光谱图像曲线可视化工具")
    print("="*60)
    
    # 检查依赖
    if not HAS_TIFFFILE:
        print("❌ 需要安装 tifffile: pip install tifffile")
        return
    
    # 检查目录
    if not os.path.exists(IMG_DIR):
        print(f"❌ 图像目录不存在: {IMG_DIR}")
        return
    
    if not os.path.exists(LABEL_DIR):
        print(f"❌ 标注目录不存在: {LABEL_DIR}")
        return
    
    # 执行批量处理
    # composite_type 可选: 'false_color', 'natural', 'nir'
    process_all_visualizations(
        composite_type='false_color',  # 推荐使用假彩色，内波更明显
        create_comparison=False  # 设为True可生成3种合成方式的对比图
    )


if __name__ == '__main__':
    main()