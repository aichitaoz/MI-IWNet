import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

# ============================================
# 1. 环境设置与函数定义
# ============================================
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid", {'grid.linestyle': '--', 'grid.alpha': 0.3})

def calculate_overlap_metrics(X, y, k=5):
    """
    计算流形重合度 (Overlap Score)
    逻辑：检查每个样本的 K 个最近邻中是否包含异源传感器样本
    """
    nbrs = NearestNeighbors(n_neighbors=k+1).fit(X)
    _, indices = nbrs.kneighbors(X)
    overlap_count = 0
    for i in range(len(X)):
        current_sensor = y[i]
        # 获取 K 个邻居的标签 (排除自身)
        neighbor_labels = y[indices[i, 1:]]
        # 如果邻居中存在不同传感器的样本，说明在该区域两个域是交织的
        if (current_sensor == 'SAR' and 'MODIS' in neighbor_labels) or \
           (current_sensor == 'MODIS' and 'SAR' in neighbor_labels):
            overlap_count += 1
    return overlap_count / len(X)

# ============================================
# 2. 加载数据与预处理
# ============================================
df = pd.read_csv("internal_wave_feature_mine.csv")

# 使用量化筛选出的 Top 6 最强对齐特征
best_6_features = [
    'mean_pixel_kurt', 'mean_entropy', 'mean_contrast', 
    'stripe_count', 'mean_snr', 'mean_pixel_skew'
]

# 域对齐处理 (Group-wise Normalization)
df_aligned = df.copy()
for sensor in ['SAR', 'MODIS']:
    mask = (df_aligned['sensor'] == sensor)
    for feat in best_6_features:
        scaler = StandardScaler()
        vals = df_aligned.loc[mask, [feat]].values
        if len(vals) > 0:
            # 填补可能的空值并进行标准化
            valid_vals = np.nan_to_num(vals)
            df_aligned.loc[mask, [feat]] = scaler.fit_transform(valid_vals)

df_final = df_aligned.dropna(subset=best_6_features)
X = df_final[best_6_features].values
y = df_final['sensor'].values

# ============================================
# 3. 核心计算 (t-SNE & Metrics)
# ============================================
tsne = TSNE(
    n_components=2, 
    perplexity=45, 
    n_iter=2000, 
    init='pca', 
    random_state=42
)
components = tsne.fit_transform(X)

# 计算量化指标
# A. 质心距离 (越小说明整体对齐越好)
sar_pts = components[y == 'SAR']
modis_pts = components[y == 'MODIS']
sar_centroid = sar_pts.mean(axis=0)
modis_centroid = modis_pts.mean(axis=0)
centroid_dist = np.linalg.norm(sar_centroid - modis_centroid)

# B. 流形重合度 (百分比越高说明数据混合越均匀，越利于 Joint Training)
overlap_score = calculate_overlap_metrics(components, y, k=5)

# ============================================
# 4. 高质量学术风格绘图
# ============================================
fig, ax = plt.subplots(figsize=(11, 8), dpi=120)

# 专业配色
colors = {'SAR': '#3498db', 'MODIS': '#e74c3c'}

# 绘制散点
for sensor in ['SAR', 'MODIS']:
    mask = y == sensor
    ax.scatter(
        components[mask, 0], 
        components[mask, 1],
        c=colors[sensor],
        label=sensor,
        alpha=0.6,
        s=65,
        edgecolors='white',
        linewidths=0.5,
        zorder=2
    )

# 绘制质心 (可选，用于视觉强调)
ax.scatter(sar_centroid[0], sar_centroid[1], c='navy', marker='X', s=200, edgecolors='white', zorder=5)
ax.scatter(modis_centroid[0], modis_centroid[1], c='darkred', marker='X', s=200, edgecolors='white', zorder=5)

# 5. 添加指标标注框 (论文加分项)
stats_text = (
    f"$\mathbf{{Manifold\ Alignment\ Metrics}}$\n"
    f"Centroid Distance: {centroid_dist:.4f}\n"
    f"Domain Overlap: {overlap_score:.2%}\n"
    f"Alignment Features: n=6"
)
props = dict(boxstyle='round,pad=0.5', facecolor='#f8f9fa', alpha=0.9, edgecolor='#bdc3c7')
ax.text(0.03, 0.03, stats_text, transform=ax.transAxes, fontsize=11,
        verticalalignment='bottom', bbox=props, family='monospace', zorder=10)

# 美化轴标签与标题
ax.set_xlabel('t-SNE Dimension 1', fontsize=14, fontweight='bold')
ax.set_ylabel('t-SNE Dimension 2', fontsize=14, fontweight='bold')
ax.set_title('Feature Space Alignment: SAR vs. MODIS', fontsize=16, pad=20, fontweight='bold')

# 美化图例
legend = ax.legend(title='Sensor Type', title_fontsize=12, fontsize=11, 
                   frameon=True, shadow=True, loc='upper right')

# 设置背景
ax.set_facecolor('#ffffff')
fig.patch.set_facecolor('white')

plt.tight_layout()
plt.savefig("t-SNE_Alignment_Analysis.pdf", bbox_inches='tight')
plt.savefig("t-SNE_Alignment_Analysis.png", dpi=300)
plt.show()

print(f"📊 分析完成!")
print(f">> 质心距离: {centroid_dist:.4f} (越小表示分布重心越接近)")
print(f">> 流形重合度: {overlap_score:.2%} (越高表示混合训练潜力越大)")