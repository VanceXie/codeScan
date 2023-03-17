# -*- coding: UTF-8 -*-
import cv2
import numpy as np

# 读取图像
image = cv2.imread(r'D:\Project\codeScan\test_images\16.png')

# 灰度化
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# 边缘检测
# edges = cv2.Canny(gray, 50, 150)


def nothing(x):
    global gray
    
    edge_horizon = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=int(2 * x + 1))
    edge_horizon_show = cv2.convertScaleAbs(edge_horizon)
    edge_vertical = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=int(2 * x + 1))
    edge_vertical_show = cv2.convertScaleAbs(edge_vertical)
    edge = edge_horizon - edge_vertical
    edge = np.where(edge >= 0, edge, 0)
    edge_show = cv2.normalize(edge, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
    # edge_gray = cv2.medianBlur(edge, 5)
    # # 阈值处理
    # threshold = cv2.adaptiveThreshold(edge_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 20 * x + 3, 0)
    # result = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=5)
    
    # threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    # # 分割行
    # a = threshold.shape[0] // 300
    # rows = np.array_split(threshold, threshold.shape[0] // 300)
    #
    # # 寻找条码位置
    # for index, row in enumerate(rows):
    #     # 检测黑白交替的边缘
    #     if np.sum(row[0]) != 0:
    #         row_index = index
    #         break
    #
    # # 绘制条码位置
    # cv2.rectangle(image, (0, (row_index - 1) * 300), (image.shape[1], row_index * 300), (0, 255, 0), 2)
    result = cv2.vconcat([edge_horizon_show, edge_vertical_show, edge_show])
    # 显示图像
    cv2.imshow("Barcode Detection", result)
    # cv2.imwrite('./Defect_030_gray2.png', threshold)


cv2.namedWindow("Barcode Detection", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
cv2.createTrackbar('R', "Barcode Detection", 1, 15, nothing)
if cv2.waitKey(0) == 27:
    cv2.destroyAllWindows()
