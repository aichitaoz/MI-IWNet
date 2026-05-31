import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import json
import os
from scipy.ndimage import zoom

# ================= 1. 数据加载与路径配置 =================
acc_mask_file = 'accumulated_mask.npy'
metadata_file = 'accumulated_mask_metadata.json'
gebco_path = r'./GEBCO_14_Dec_2025_b048ba5b0116/gebco_2025_n23.19_s18.32_w112.4_e121.32.nc'

if not os.path.exists(acc_mask_file):
    print("错误：未找到累积掩膜文件！")
    exit()

acc_mask = np.load(acc_mask_file)
with open(metadata_file, 'r') as f:
    meta = json.load(f)
ny, nx = meta['shape']

# 读取GEBCO数据
ds = xr.open_dataset(gebco_path)
z_var = 'elevation' if 'elevation' in ds.data_vars else 'z'
z_data = ds[z_var].values
ds.close()

# ================= 2. 数据对齐与预处理 =================
print("正在对齐水深网格并提取特征...")
z_res = zoom(z_data, (ny / z_data.shape[0], nx / z_data.shape[1]), order=1)

# 提取有内波且在海洋(z<0)的像素
iw_idx = (acc_mask > 0) & (z_res < 0)
final_d = -z_res[iw_idx]
final_w = acc_mask[iw_idx]

# 过滤无效值
valid_mask = ~np.isnan(final_d)
final_d, final_w = final_d[valid_mask], final_w[valid_mask]

# ================= 3. 统计绘图 (纯色正规版) =================
print("正在绘制分布图表...")
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans'],
    'font.size': 11,
    'axes.linewidth': 1.2,
    'axes.labelweight': 'bold'
})

fig, ax = plt.subplots(figsize=(10, 5), dpi=300)

bin_size = 50
bins = np.arange(0, 4501, bin_size)
hist, _ = np.histogram(final_d, bins=bins, weights=final_w)
total_w = final_w.sum()
pct = (hist / total_w) * 100
centers = (bins[:-1] + bins[1:]) / 2

# 绘图设置：纯色 + 缝隙优化
ax.bar(centers, pct, width=40, color='steelblue', 
       edgecolor='white', linewidth=0.6, alpha=0.9, zorder=3)

ax.set_xlabel('Water Depth (m)', fontweight='bold', labelpad=10)
ax.set_ylabel('Percentage (%)', fontweight='bold', labelpad=10)
ax.set_xlim(-100, 4550)
ax.set_ylim(0, max(pct) * 1.1)
ax.set_xticks(np.arange(0, 4501, 500))
ax.tick_params(direction='out', length=6, width=1.2)
ax.grid(axis='y', linestyle='--', alpha=0.4, zorder=0)
ax.set_axisbelow(True)
ax.text(-0.07, 1.05, '(a)', transform=ax.transAxes, fontsize=16, fontweight='bold', va='bottom')

plt.tight_layout()
plt.savefig('South_sea_water_depth.png', bbox_inches='tight')
print("✓ 图表已保存为 South_sea_water_depth.png")

# ================= 4. 全维度统计分析报告 =================
print("\n" + "="*50)
print("             内波与水深关系统计报告")
print("="*50)

# A. 基础描述统计
avg_depth = np.average(final_d, weights=final_w)
std_depth = np.sqrt(np.average((final_d - avg_depth)**2, weights=final_w))
median_depth = np.median(np.repeat(final_d.astype(int), final_w.astype(int))) # 近似加权中位数

print(f"【核心指标】")
print(f"总IW像素(加权累积): {total_w:>15,.0f}")
print(f"有效统计网格数:     {len(final_d):>15,}")
print(f"平均水深 (μ):        {avg_depth:>15.1f} m")
print(f"标准差 (σ):          {std_depth:>15.1f} m")
print(f"中位数水深:          {median_depth:>15.1f} m")
print(f"最高频水深区间:      {centers[np.argmax(pct)]-25:>6.0f} - {centers[np.argmax(pct)]+25:<.0f} m")

# B. 关键地形分布
shelf_break = (final_d <= 200)
slope_deep = (final_d > 200) & (final_d <= 2000)
abyssal = (final_d > 2000)

print(f"\n【地形区分布占比】")
print(f"大陆架区域 (0-200m):     {final_w[shelf_break].sum()/total_w*100:>10.2f} %")
print(f"坡折与中深海 (200-2000m): {final_w[slope_deep].sum()/total_w*100:>10.2f} %")
print(f"深海盆地 (>2000m):       {final_w[abyssal].sum()/total_w*100:>10.2f} %")

# C. 详细频数分布表
print(f"\n【详细分布表】")
print("-" * 45)
print(f"  {'深度区间 (m)':<15} | {'像素百分比 (%)':>15}")
print("-" * 45)
range_bins = np.arange(0, 5001, 500)
for i in range(len(range_bins)-1):
    d_min, d_max = range_bins[i], range_bins[i+1]
    m = (final_d >= d_min) & (final_d < d_max)
    p = (final_w[m].sum() / total_w) * 100
    print(f"  {d_min:>4d} - {d_max:<4d}      | {p:>15.2f}")
print("-" * 45)
print("="*50)
print("统计完成。请将以上信息发送给分析伙伴。")

plt.show()