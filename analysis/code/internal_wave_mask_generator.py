import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from scipy.ndimage import gaussian_filter1d
from matplotlib.patches import Polygon, Circle
import matplotlib.patheffects as pe

# 设置论文风格
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['axes.edgecolor'] = 'lightgray'

# ==================== 生成数据 ====================
n_points_curve = 400
t_base = np.linspace(0, 1, n_points_curve)

x_base = 5 + t_base * 290
y_base = 75 + 60 * np.sin(np.pi * t_base)
y_base += 3 * np.sin(8 * np.pi * t_base) * np.exp(-((t_base - 0.5)**2) / 0.1)

sample_indices = [0, 50, 120, 200, 280, 350, n_points_curve-1]
control_points = np.array([[x_base[i], y_base[i]] for i in sample_indices])

def ultra_smooth_curve(points, n_points=n_points_curve, smooth_factor=0.01):
    points = np.array(points)
    if len(points) < 2: return points
    
    deltas = np.diff(points, axis=0)
    dist = np.sqrt(np.sum(deltas**2, axis=1))
    dist[dist == 0] = 1e-6

    t = np.hstack(([0], np.cumsum(dist)))
    t /= t[-1]

    x_spline = UnivariateSpline(t, points[:, 0], s=smooth_factor)
    y_spline = UnivariateSpline(t, points[:, 1], s=smooth_factor)
    t_new = np.linspace(0, 1, n_points)
    return np.stack([x_spline(t_new), y_spline(t_new)], axis=1)

curve = ultra_smooth_curve(control_points)

normals = []
widths = []
base_width = 15
width_amplitude = 8
for i in range(len(curve)):
    if i == 0:
        tangent = curve[1] - curve[0]
    elif i == len(curve) - 1:
        tangent = curve[-1] - curve[-2]
    else:
        tangent = curve[i+1] - curve[i-1]
        
    dt = np.linalg.norm(tangent)
    if dt > 1e-6:
        nx, ny = -tangent[1]/dt, tangent[0]/dt
    else:
        nx, ny = 0, 0
    normals.append([nx, ny])
    
    width_variation = width_amplitude * np.sin(2 * np.pi * i / len(curve) + np.pi/4)**2 
    widths.append(base_width + width_variation)

widths = gaussian_filter1d(widths, sigma=15)
normals = np.array(normals)

def generate_tube_boundary(curve, normals, widths):
    upper_boundary = []
    lower_boundary = []
    for i in range(len(curve)):
        pt = curve[i]
        nx, ny = normals[i]
        w_half = widths[i] / 2
        upper_boundary.append([pt[0] + nx*w_half, pt[1] + ny*w_half])
        lower_boundary.append([pt[0] - nx*w_half, pt[1] - ny*w_half])
    return np.array(upper_boundary), np.array(lower_boundary[::-1])

upper_bound, lower_bound = generate_tube_boundary(curve, normals, widths)
tube_polygon_outline = np.vstack([upper_bound, lower_bound])

# ==================== 通用坐标轴设置（无标题、无标签、无图例）====================
def setup_clean_axis(ax):
    ax.set_xlim(0, 300)
    ax.set_ylim(0, 150)
    ax.set_aspect('equal', adjustable='box')
    ax.set_facecolor('#f9f9f9')
    ax.tick_params(labelsize=9, colors='#555555')
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5, color='lightgray')
    # 去掉所有标签和标题
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_xticks([])
    ax.set_yticks([])

# ==================== 图(a) ====================
fig1, ax1 = plt.subplots(figsize=(6, 5))
setup_clean_axis(ax1)

tube_patch_a = Polygon(tube_polygon_outline, facecolor='lightgray', edgecolor='none', 
                       alpha=0.3, zorder=1)
ax1.add_patch(tube_patch_a)

ax1.plot(control_points[:, 0], control_points[:, 1], 
         linestyle=':', color='#ff7f0e', linewidth=1.5, zorder=3,
         path_effects=[pe.Stroke(linewidth=3, foreground='white'), pe.Normal()])

ax1.scatter(control_points[:, 0], control_points[:, 1], 
            c='#d62728', s=100, marker='o', edgecolors='white', 
            linewidths=2, zorder=5)

plt.tight_layout(pad=0)
plt.savefig('fig_a_control_points.pdf', dpi=300, bbox_inches='tight', facecolor='white')
print("✅ 已保存: fig_a_control_points.png")
plt.close()

# ==================== 图(b) ====================
fig2, ax2 = plt.subplots(figsize=(6, 5))
setup_clean_axis(ax2)

tube_patch_b = Polygon(tube_polygon_outline, facecolor='lightgray', edgecolor='none', 
                       alpha=0.3, zorder=1)
ax2.add_patch(tube_patch_b)

ax2.scatter(control_points[:, 0], control_points[:, 1], 
            c='#d62728', s=60, marker='o', alpha=0.5, zorder=3)

ax2.plot(curve[:, 0], curve[:, 1], color='#1f77b4', linewidth=4, 
         zorder=4, path_effects=[pe.Stroke(linewidth=6, foreground='white'), pe.Normal()])

plt.tight_layout(pad=0)
plt.savefig('fig_b_spine_fitting.pdf', dpi=300, bbox_inches='tight', facecolor='white')
print("✅ 已保存: fig_b_spine_fitting.png")
plt.close()

# ==================== 图(c) ====================
fig3, ax3 = plt.subplots(figsize=(6, 5))
setup_clean_axis(ax3)

tube_patch_c = Polygon(tube_polygon_outline, facecolor='lightgray', edgecolor='none', 
                       alpha=0.3, zorder=1)
ax3.add_patch(tube_patch_c)

ax3.plot(curve[:, 0], curve[:, 1], color='#1f77b4', linewidth=2, alpha=0.7, zorder=2)

for i in range(0, len(curve), 25):
    pt = curve[i]
    nx, ny = normals[i]
    w_half = widths[i] / 2
    
    ax3.annotate('', xy=(pt[0] + nx*w_half, pt[1] + ny*w_half), 
                 xytext=(pt[0] - nx*w_half, pt[1] - ny*w_half),
                 arrowprops=dict(arrowstyle='<->', color='crimson', lw=2, 
                                 mutation_scale=20, shrinkA=0, shrinkB=0),
                 zorder=5, alpha=0.8)
    
    ax3.plot([pt[0] - nx*w_half, pt[0] + nx*w_half], 
             [pt[1] - ny*w_half, pt[1] + ny*w_half], 
             'o', color='gold', markersize=5, markeredgecolor='darkgoldenrod',
             markeredgewidth=1, zorder=6)

plt.tight_layout(pad=0)
plt.savefig('fig_c_width_profiling.pdf', dpi=300, bbox_inches='tight', facecolor='white')
print("✅ 已保存: fig_c_width_profiling.png")
plt.close()

# ==================== 图(d) ====================
fig4, ax4 = plt.subplots(figsize=(6, 5))
setup_clean_axis(ax4)

final_mask_polygon = Polygon(tube_polygon_outline, facecolor='#17a2b8', edgecolor='#0d6efd', 
                             linewidth=2, alpha=0.85, zorder=3)
ax4.add_patch(final_mask_polygon)

start_pt = curve[0]
end_pt = curve[-1]
start_cap = Circle(start_pt, widths[0]/2, facecolor='#17a2b8', edgecolor='#0d6efd', 
                   linewidth=2, alpha=0.85, zorder=4)
end_cap = Circle(end_pt, widths[-1]/2, facecolor='#17a2b8', edgecolor='#0d6efd', 
                 linewidth=2, alpha=0.85, zorder=4)
ax4.add_patch(start_cap)
ax4.add_patch(end_cap)

ax4.plot(curve[:, 0], curve[:, 1], color='#ffc107', linestyle='-', linewidth=2.5, 
         alpha=0.9, zorder=6,
         path_effects=[pe.Stroke(linewidth=4, foreground='white'), pe.Normal()])

plt.tight_layout(pad=0)
plt.savefig('fig_d_final_mask.pdf', dpi=300, bbox_inches='tight', facecolor='white')
print("✅ 已保存: fig_d_final_mask.png")
plt.close()

print("\n✅ 所有图片已保存完成！")