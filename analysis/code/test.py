import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 加载地形数据
ds = xr.open_dataset("/home/xiaobowen/project/internal_wave_detection_project/analysis/GEBCO_15_Dec_2025_6d47b43e62e6/gebco_2025_n23.689_s-1.2161_w80.4424_e128.6421.nc")

# 2. 读取你的内波 CSV 文件 (假设以南海为例)
df = pd.read_csv('/home/xiaobowen/project/internal_wave_detection_project/analysis/iw_clusters_geoinfo_rgb.csv')

# 3. [关键步骤] 为每一个内波点提取正下方的真实水深
depths = []
for idx, row in df.iterrows():
    # 使用 nearest 方法找距离内波中心最近的地形网格点
    depth_val = ds['elevation'].sel(lon=row['Center_Lon'], lat=row['Center_Lat'], method='nearest').values
    depths.append(float(depth_val))

# 把真实水深作为新的一列加入 dataframe
df['Real_Depth'] = depths

# 4. [核武级证据] 画定量统计图给审稿人看！
plt.figure(figsize=(10, 6))

# 假设你的 CSV 里有波包面积（Area）这一列
# 画一张散点图：横坐标是水深，纵坐标是面积
sns.scatterplot(data=df, x='Real_Depth', y='Area', alpha=0.6)
plt.title("Correlation between Water Depth and Internal Wave Area (South China Sea)")
plt.xlabel("Water Depth (m)")
plt.ylabel("Wave Packet Area (pixels)")

# 算一下皮尔逊相关系数
correlation = df['Real_Depth'].corr(df['Area'])
plt.text(0.05, 0.95, f'Correlation (R) = {correlation:.2f}', transform=plt.gca().transAxes, fontsize=12)

plt.savefig('depth_area_correlation.png')
print("统计图已生成！")