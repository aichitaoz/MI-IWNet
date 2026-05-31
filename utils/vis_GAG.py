import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# =========================
# 配置
# =========================
feature_dir = r"/home/xiaobowen/project/internal_wave_detection_project/GAG_stage_0_input_x"
max_slices = 32                   
alpha = 0.6
elev = 15                         # 🔥 降低仰角，更平视（原来是30）
azim = -60                        

# =========================
# 1️⃣ 读取 PNG feature maps
# =========================
png_files = sorted([
    f for f in os.listdir(feature_dir)
    if f.lower().endswith(".png")
])

assert len(png_files) > 0, "❌ 文件夹里没有 PNG 文件"

feature_list = []

for fname in png_files:
    path = os.path.join(feature_dir, fname)
    img = Image.open(path).convert("RGB")
    fmap = np.array(img, dtype=np.float32) / 255.0
    feature_list.append(fmap)

feature_cube = np.stack(feature_list, axis=0)
D, H, W, C = feature_cube.shape
print(f"✅ Feature cube shape: {feature_cube.shape}")

# =========================
# 2️⃣ 下采样 depth
# =========================
if D > max_slices:
    idx = np.linspace(0, D - 1, max_slices).astype(int)
    feature_cube = feature_cube[idx]
    D = feature_cube.shape[0]

# =========================
# 3️⃣ 3D 立方体可视化（往后堆叠）
# =========================
fig = plt.figure(figsize=(12, 10))
fig.patch.set_alpha(0)  # 🔥 设置整个图形背景透明

ax = fig.add_subplot(111, projection="3d")
ax.patch.set_alpha(0)   # 🔥 设置3D坐标轴背景透明

x = np.arange(W)
z = np.arange(H)
X, Z = np.meshgrid(x, z)

for y in range(D):
    Y = np.full_like(X, y, dtype=float)
    
    ax.plot_surface(
        X, Y, Z,
        rstride=1,
        cstride=1,
        facecolors=np.flip(feature_cube[y], axis=0),
        shade=False,
        alpha=alpha,
        antialiased=True
    )

# 🔥 更平视的视角
ax.view_init(elev=elev, azim=azim)

# 去掉坐标轴
ax.set_axis_off()

# 🔥 去掉3D坐标系的背景面板
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

ax.set_title("Oblique View of Feature Cube", fontsize=14, pad=20)

plt.tight_layout()

# 🔥 保存为透明背景PNG
plt.savefig("feature_cube_oblique.png", dpi=300, bbox_inches='tight', 
            transparent=True, facecolor='none')
plt.show()