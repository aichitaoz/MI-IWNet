import pandas as pd

# 1. 定义文件路径
input_file = '/root/data/iw_clusters_geoinfo_sar.csv'
output_sulu = '/root/data/iw_clusters_Sulu.csv'
output_andaman = '/root/data/iw_clusters_Andaman.csv'

print("正在读取数据...")
try:
    # 2. 读取 CSV 数据
    df = pd.read_csv(input_file)
    print(f"成功读取数据，共 {len(df)} 条内波记录。")
except FileNotFoundError:
    print(f"错误：未找到文件 {input_file}，请确保你在正确的目录下运行。")
    exit()

# 3. 定义海域的经纬度边界 (大幅加宽版)
# 苏禄海 (Sulu Sea) 宽边界：向西包容巴拉望岛外围，向东到棉兰老岛，向北到民都洛
sulu_lon_min, sulu_lon_max = 115.0, 125.0
sulu_lat_min, sulu_lat_max = 3.5, 13.5

# 安达曼海 (Andaman Sea) 宽边界：向西包容安达曼群岛，向东到泰国海岸，南北延展
andaman_lon_min, andaman_lon_max = 88.0, 101.0
andaman_lat_min, andaman_lat_max = 3.0, 17.0

# 4. 根据 Center_Lon 和 Center_Lat 进行掩码筛选
# 提取苏禄海数据
sulu_mask = (
    (df['Center_Lon'] >= sulu_lon_min) & (df['Center_Lon'] <= sulu_lon_max) &
    (df['Center_Lat'] >= sulu_lat_min) & (df['Center_Lat'] <= sulu_lat_max)
)
sulu_df = df[sulu_mask]

# 提取安达曼海数据
andaman_mask = (
    (df['Center_Lon'] >= andaman_lon_min) & (df['Center_Lon'] <= andaman_lon_max) &
    (df['Center_Lat'] >= andaman_lat_min) & (df['Center_Lat'] <= andaman_lat_max)
)
andaman_df = df[andaman_mask]

# 5. 保存为新的 CSV 文件
sulu_df.to_csv(output_sulu, index=False)
andaman_df.to_csv(output_andaman, index=False)

print("\n分离完成！")
print(f"- 苏禄海 (Sulu Sea): 提取了 {len(sulu_df)} 行，已保存至 {output_sulu}")
print(f"- 安达曼海 (Andaman Sea): 提取了 {len(andaman_df)} 行，已保存至 {output_andaman}")

# 6. 检查是否有遗漏的数据
other_df = df[~sulu_mask & ~andaman_mask]
if not other_df.empty:
    print(f"\n提示：加宽边界后，仍有 {len(other_df)} 行数据不在定义的范围内。")
    # 把剩下的“漏网之鱼”存下来，方便你查看它们到底在哪
    other_df.to_csv('/root/data/iw_clusters_Leftovers.csv', index=False)
    print("已将剩余的未知区域数据保存至 /root/data/iw_clusters_Leftovers.csv，你可以打开看看经纬度。")
else:
    print("\n完美！所有数据已成功归类。")