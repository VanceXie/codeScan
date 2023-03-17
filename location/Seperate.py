# -*- coding: UTF-8 -*-
import cv2
import numpy as np

# 读入图像
image = cv2.imread(r'D:\Project\codeScan\test_images\16.png')

# 转换为灰度图
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 阈值分割
threshold, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# 对二值图像进行 K-Means 聚类
Z = thresh.reshape((-1, 1))
Z = np.float32(Z)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 2
ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# 确定背景和前景的灰度值
background_value = center[0][0]
foreground_value = center[1][0]

# 判断是否存在一维条码
barcode_present = False
for row in label:
    black_count = 0
    white_count = 0
    for p in row:
        if p == foreground_value:
            black_count += 1
        else:
            white_count += 1
    if black_count > 0 and white_count > 0:
        barcode_present = True
        break

if barcode_present:
    print("一维条码已检测到！")
else:
    print("未检测到一维条码！")
