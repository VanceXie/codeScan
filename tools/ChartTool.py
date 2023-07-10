# -*- coding: UTF-8 -*-
import numpy as np
import matplotlib.pyplot as plt

# 创建x取值范围
x1 = np.linspace(0, 51, 1000)
x2 = np.linspace(0, 102, 1000)
x3 = np.linspace(0, 153, 1000)
x4 = np.linspace(0, 204, 1000)
x5 = np.linspace(0, 255, 1000)

# 计算y的取值
y1 = np.power(255, x1 / 51)
y2 = np.power(255, x2 / 102)
y3 = np.power(255, x3 / 153)
y4 = np.power(255, x4 / 204)
y5 = np.power(255, x5 / 255)

# 设置图像的像素尺寸
fig = plt.figure(figsize=(20, 16), dpi=300)  # 宽度为8英寸，高度为6英寸
# 绘制图像
plt.plot(x1, y1, label='51')
plt.plot(x2, y2, label='102')
plt.plot(x3, y3, label='153')
plt.plot(x4, y4, label='204')
plt.plot(x5, y5, label='255')

# 设置图例和标题
legend = plt.legend(loc='lower right', fontsize='16')
legend.set_title('I_max')
legend.get_title().set_fontsize('16')  # 设置图例标题字号为14

plt.title('I_equ=[(255)^(I/I_max)]\n', fontsize=30)

plt.xlim([0, 255])
plt.ylim([0, 255])
# 添加纵向网格线
for i in range(51, 256, 51):
	plt.axvline(x=i, color='red', linestyle='--', linewidth=1)

# # 调整标题和图像的间距
# plt.subplots_adjust(top=0.5)  # 通过调整top参数的值来设置间距
# 显示图像
plt.savefig(r"D:\Fenwick\Pictures\Paper\Figure_1.png", bbox_inches='tight', pad_inches=0.5)

plt.show()
