# -*- coding: UTF-8 -*-
import cv2
import numpy as np
from sklearn.cluster import KMeans

# 读入图像
img = cv2.imread("barcode.jpg")

# 转换到灰度图
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 使用 Otsu 二值化
_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# 将二值图转换为数组
array = np.array(thresh)

# 对每一列像素的灰度值进行 K-Means 聚类
column_pixels = []
for i in range(array.shape[1]):
    column_pixels.append(array[:, i])

kmeans = KMeans(n_clusters=2)
kmeans.fit(column_pixels)

# 取得每一列像素的聚类标签
labels = kmeans.labels_

# 检测条码的起始和终止位置
start, end = -1, -1
for i in range(len(labels)):
    if labels[i] == 0 and start == -1:
        start = i
    if labels[i] == 1 and end == -1:
        end = i
        break

# 对条码位置进行标记
cv2.rectangle(img, (start, 0), (end, array.shape[0]), (0, 0, 255), 2)

# 显示标记后的图像
cv2.imshow("Barcode Detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
