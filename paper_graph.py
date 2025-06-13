import matplotlib.pyplot as plt
# import numpy as np
#
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 16
#
# # 数据
# categories = ['Louvain', 'Kmeans', 'PhenoGraph', 'scDCC', 'sigDGCNb', 'scCDG', 'DeepScena', 'CAKE', 'scSAG$^2$E']
# # values = [0.3523, 0.3831, 0.5191, 0.5032, 0.502, 0.251, 0.5438, 0.707, 0.707]  #droupout=0.5
# # values = [0.3539,0.3772,0.5432,0.5003,0.5006,0.2518,0.5012,0.707,0.707]  #droupout=0.75
# # values = [0.3558,0.381,0.5599,0.5022,0.5113,0.2516,0.5151,0.707,0.707]  #droupout=1
# # values = [0.3504,0.379,0.5166,0.5001,0.5018,0.2553,0.5215,0.707,0.707]  #droupout=1.25
# values = [0.3504,0.3806,0.5805,0.5009,0.5029,0.2515,0.5251,0.707,0.707]#droupout=1.5
#
#
# # 创建颜色映射
# cmap = plt.get_cmap('viridis')
# colors = cmap(np.linspace(0, 1, len(categories)))
#
#
# # 创建柱状图
# plt.figure(figsize=(10, 6))
# bars = plt.bar(categories, values, color=colors)
#
# # 添加标题和标签
# plt.title('FMI')
# # plt.xlabel('类别')
# # plt.ylabel('值')
# plt.xticks(rotation=20)
#
# # 显示图表
# # plt.show()
#
# plt.savefig('droupout_1.5.pdf')

# z_height = [0.831,0.831,0.831,0.831,0.831,
#             0.831,0.831,0.831,0.831,0.831,
#             0.831,0.831,0.831,0.831,0.831,
#             0.831,0.831,0.831,0.831,0.831,
#             0.831,0.831,0.831,0.831,0.831]


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 设置X轴和Y轴的值
x_values = [10, 20, 30, 40, 50]  # X轴坐标
y_values = [1, 3, 5, 7, 9]       # Y轴坐标

# 创建网格坐标
x_pos, y_pos = np.meshgrid(x_values, y_values)
x_pos = x_pos.flatten()  # 展平为一维数组
y_pos = y_pos.flatten()  # 展平为一维数组
z_pos = np.zeros(25)     # 所有柱子起始高度为0

# 随机生成25个柱子的高度 (Z值)
np.random.seed(42)  # 设置随机种子以便结果可复现
# z_height = np.random.rand(25) * 15  # 高度在0-15之间
z_height = [0.79,0.79,0.79,0.79,0.79,
            0.79,0.79,0.79,0.79,0.79,
            0.79,0.79,0.79,0.79,0.79,
            0.79,0.79,0.79,0.79,0.79,
            0.79,0.79,0.79,0.79,0.79]


# 设置柱子的宽度和深度
dx = [5] * 25  # 每个柱子X方向的宽度
dy = [1] * 25  # 每个柱子Y方向的深度
dz = z_height  # 柱子的高度

# 创建3D图形
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')

# 绘制柱状图，使用不同颜色
colors = plt.cm.plasma(np.linspace(0, 1, 25))  # 使用plasma颜色映射
ax.bar3d(x_pos, y_pos, z_pos, dx, dy, dz, color=colors, shade=True)

# 设置轴标签和刻度
ax.set_xlabel('K',fontweight='bold')
ax.set_ylabel('T',fontweight='bold')
# ax.set_zlabel('Z Value')

# 设置X轴和Y轴刻度
ax.set_xticks(x_values)
ax.set_yticks(y_values)

# 设置标题
# ax.set_title('3D Bar Chart with Custom X/Y Values (25 Bars)')

# 调整视角
ax.view_init(elev=30, azim=45)  # 仰角30度，方位角45度

plt.tight_layout()
# plt.show()

plt.savefig('FMI.pdf')