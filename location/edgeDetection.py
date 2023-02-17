# -*- coding: UTF-8 -*-
import cv2
import numpy as np

# 读取图像
image = cv2.imread("./Defect_030.png")

# 灰度化
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 边缘检测
# edges = cv2.Canny(gray, 50, 150)
gray = cv2.medianBlur(gray, 3)


def nothing(x):
    # 阈值处理
    threshold = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 2 * x + 3, 0)
    # threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    # 分割行
    a = threshold.shape[0] // 300
    rows = np.array_split(threshold, threshold.shape[0] // 300)
    
    # 寻找条码位置
    for index, row in enumerate(rows):
        # 检测黑白交替的边缘
        if np.sum(row[0]) != 0:
            row_index = index
            break
    
    # 绘制条码位置
    cv2.rectangle(image, (0, (row_index - 1) * 300), (image.shape[1], row_index * 300), (0, 255, 0), 2)
    
    # 显示图像
    
    cv2.imshow("Barcode Detection", threshold)
    # cv2.imwrite('./Defect_030_gray2.png', threshold)


cv2.namedWindow("Barcode Detection", cv2.WINDOW_NORMAL)
cv2.createTrackbar('R', "Barcode Detection", 0, 500, nothing)
nothing(200)
if cv2.waitKey(0) == 27:
    cv2.destroyAllWindows()
