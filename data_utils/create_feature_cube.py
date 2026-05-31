import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse


def load_feature_channels(folder_path, selected_channels=None, downsample=4):
    """
    加载指定文件夹中的特征图通道
    
    Args:
        folder_path: 包含channel_xxx.png的文件夹路径
        selected_channels: 要选择的通道列表,如[0,10,20,30],None表示全部加载
        downsample: 下采样倍数,用于加速渲染
    
    Returns:
        channels: 形状为(N, H, W, 3)的numpy数组,N为通道数
        channel_indices: 实际加载的通道索引
    """
    # 获取所有通道文件
    all_files = sorted([f for f in os.listdir(folder_path) if f.startswith('channel_') and f.endswith('.png')])
    
    if not all_files:
        raise ValueError(f"No channel files found in {folder_path}")
    
    # 如果指定了选择的通道
    if selected_channels is not None:
        selected_files = [f"channel_{i:03d}.png" for i in selected_channels]
        files_to_load = [f for f in selected_files if f in all_files]
        channel_indices = selected_channels[:len(files_to_load)]
    else:
        files_to_load = all_files
        channel_indices = [int(f.split('_')[1].split('.')[0]) for f in all_files]
    
    print(f"📂 Loading {len(files_to_load)} channels from {folder_path}")
    
    # 加载第一张图像获取原始尺寸
    first_img = cv2.imread(os.path.join(folder_path, files_to_load[0]))
    orig_h, orig_w = first_img.shape[:2]
    
    # 计算下采样后的尺寸
    new_h, new_w = orig_h // downsample, orig_w // downsample
    print(f"📐 Original size: {orig_w}x{orig_h}, Downsampled to: {new_w}x{new_h} (factor={downsample})")
    
    # 加载并下采样图像
    channels = []
    for i, fname in enumerate(files_to_load):
        img_path = os.path.join(folder_path, fname)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR -> RGB
        
        # 下采样加速
        if downsample > 1:
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        channels.append(img)
        
        if (i + 1) % 20 == 0:
            print(f"   Loaded {i+1}/{len(files_to_load)} channels...")
    
    channels = np.array(channels)  # (N, H, W, 3)
    
    print(f"✅ Loaded shape: {channels.shape}")
    return channels, channel_indices


def create_feature_cube_front_view(channels, channel_indices, save_path, 
                                   spacing=0.3, figsize=(16, 12), dpi=100, stride=4,
                                   elevation=15, azimuth=-60):
    """
    修正版：实现真正的“正常视角” + 透明背景
    """
    num_channels, height, width, _ = channels.shape
    
    # 1. 初始化：背景设为完全透明
    fig = plt.figure(figsize=figsize, dpi=dpi, facecolor='none')
    ax = fig.add_subplot(111, projection='3d', facecolor='none')
    
    # 2. 坐标网格：Z轴向上，但映射为图片从上到下的顺序
    x = np.arange(0, width, stride)
    z = np.arange(0, height, stride)
    X, Z = np.meshgrid(x, z)
    
    # 计算总深度
    total_depth = (num_channels - 1) * spacing * width
    
    for i, (channel_img, ch_idx) in enumerate(zip(channels, channel_indices)):
        # Y轴代表深度，从 0 延伸到 total_depth
        Y = np.ones_like(X) * (i * spacing * width)
        
        # 核心：Matplotlib 3D 绘图 Z轴 0 在下，图片 0 在上，所以必须翻转图片内容
        channel_img_flipped = np.flipud(channel_img)
        colors_downsampled = channel_img_flipped[::stride, ::stride, :].astype(np.float32) / 255.0
        
        ax.plot_surface(X, Y, Z, 
                        facecolors=colors_downsampled,
                        rstride=1, cstride=1,
                        antialiased=False,
                        shade=False,
                        alpha=0.95, 
                        linewidth=0)

    # 3. 视角设置：elev=15 (稍微俯视), azim=-60 (左侧透视)
    ax.view_init(elev=elevation, azim=azimuth)
    
    # 4. 彻底去掉背景、面板、网格线
    ax.set_axis_off()
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.pane.fill = False
        axis.pane.set_edgecolor('none')
    ax.grid(False)
    
    # 5. 关键：修正比例，防止图像被拉伸或压缩
    # 我们希望 X 和 Z 的比例是 1:1，Y 的比例由层数决定
    ax.set_box_aspect([width, total_depth, height]) 
    
    # 6. 设置显示范围
    ax.set_xlim(0, width)
    ax.set_ylim(0, max(total_depth, 1)) # 防止单层报错
    ax.set_zlim(0, height)
    
    plt.tight_layout(pad=0)
    
    # 7. 导出透明图片
    print(f"💾 Saving corrected view to {save_path}...")
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Create front-facing 3D feature cube visualization')
    parser.add_argument('--input_dir', type=str, default='/home/xiaobowen/project/internal_wave_detection_project/IW_data/features_onlysar/ConvNeXt_real_size/stage1_256x256/10',
                       help='Input directory containing channel_xxx.png files')
    parser.add_argument('--output_dir', type=str, default='./feature_cubes',
                       help='Output directory for cube visualizations')
    parser.add_argument('--channels', type=str, default='2-4',
                       help='Selected channels (e.g., "0,10,20,30" or "0-79" or "None" for all)')
    parser.add_argument('--spacing', type=float, default=0.3,
                       help='Spacing between layers (default=0.3, smaller=more compact)')
    parser.add_argument('--figsize', type=str, default='16,12',
                       help='Figure size as "width,height"')
    parser.add_argument('--dpi', type=int, default=100,
                       help='Output DPI (lower = faster)')
    parser.add_argument('--downsample', type=int, default=1,
                       help='Image downsample factor (higher = faster, default=4)')
    parser.add_argument('--stride', type=int, default=1,
                       help='Grid stride for rendering (higher = faster but coarser, default=4)')
    parser.add_argument('--elevation', type=float, default=10,
                       help='Camera elevation angle (default=10, nearly horizontal)')
    parser.add_argument('--azimuth', type=float, default=105,
                       help='Camera azimuth angle (default=105 for slight angle view)')
    parser.add_argument('--output_name', type=str, default=None,
                       help='Output filename (default: auto-generated from input_dir)')
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("🧊 Front-Facing Feature Cube Visualization")
    print("="*80)
    
    # 解析通道选择
    selected_channels = None
    if args.channels and args.channels.lower() != 'none':
        if '-' in args.channels:
            start, end = map(int, args.channels.split('-'))
            selected_channels = list(range(start, end + 1))
        else:
            selected_channels = [int(x.strip()) for x in args.channels.split(',')]
        print(f"🎯 Selected {len(selected_channels)} channels: {selected_channels}")
    else:
        print(f"🎯 Loading all available channels")
    
    # 加载特征通道
    channels, channel_indices = load_feature_channels(
        args.input_dir, 
        selected_channels,
        downsample=args.downsample
    )
    
    # 解析图像尺寸
    figsize = tuple(map(float, args.figsize.split(',')))
    
    # 确定输出文件名
    if args.output_name:
        output_name = args.output_name
    else:
        base_name = os.path.basename(args.input_dir.rstrip('/'))
        output_name = f"{base_name}_cube.png"
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    save_path = os.path.join(args.output_dir, output_name)
    
    # 生成可视化
    create_feature_cube_front_view(
        channels, channel_indices, save_path,
        spacing=args.spacing, figsize=figsize, dpi=args.dpi, stride=args.stride,
        elevation=args.elevation, azimuth=args.azimuth
    )
    
    print("\n" + "="*80)
    print(f"🎉 Done! Saved to: {save_path}")
    print("="*80 + "\n")


if __name__ == '__main__':
    print("\n📖 Example Usage:")
    print("-" * 80)
    print("# 快速预览模式 (推荐):")
    print("python create_feature_cube.py \\")
    print("  --input_dir /path/to/channel_folder \\")
    print("  --channels 0,10,20,30,40,50,60,70 \\")
    print("  --downsample 4 --stride 4 --spacing 0.3")
    print()
    print("# 选择通道范围:")
    print("python create_feature_cube.py \\")
    print("  --input_dir /path/to/channel_folder \\")
    print("  --channels 0-39 \\")
    print("  --spacing 0.5")
    print()
    print("# 高质量模式 (较慢):")
    print("python create_feature_cube.py \\")
    print("  --input_dir /path/to/channel_folder \\")
    print("  --downsample 2 --stride 2 --dpi 150")
    print()
    print("# 调整层间距 (spacing越小越紧凑):")
    print("python create_feature_cube.py \\")
    print("  --input_dir /path/to/channel_folder \\")
    print("  --spacing 0.2  # 更紧凑")
    print("-" * 80 + "\n")
    
    main()