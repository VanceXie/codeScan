# -*- coding: UTF-8 -*-
from math import sqrt

import cv2
import numpy as np

from tools.PerformanceEval import calculate_time


def find_barcode_1(image):
    '''
    :param image: 'image as np.array'
    :return: np.array(dtype=np.uint8)
    '''
    # Perform edge detection
    edges = cv2.Canny(image, 50, 150)
    
    # Find lines in the image using HoughLinesP
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)
    
    left, top, right, bottom = lines[0][0]
    
    # 找到和一维条码类似的直线
    barcode_lines = []
    # Draw lines on the image
    for line in lines:
        x1, y1, x2, y2 = line[0]
        # 计算直线的夹角
        cos = (x2 - x1) / sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1))
        # 确定直线是否与一维条码相似
        if abs(cos) > 0.8:
            barcode_lines.append(line)
            cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            if x1 < left:
                left = x1
            if x2 < left:
                left = x2
            if x1 > right:
                right = x1
            if x2 > right:
                right = x2
            if y1 < top:
                top = y1
            if y2 < top:
                top = y2
            if y1 > bottom:
                bottom = y1
            if y2 > bottom:
                bottom = y2
    
    # Draw a rectangle around the barcode
    cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 2)
    return image


# '''适用于条码比较清晰、直线比较明显的情况下。对于条码比较模糊、扭曲的情况可能不太适用'''
def find_barcode_2(image):
    '''
    :param image: 'image as np.array'
    :return: np.array(dtype=np.uint8)
    '''
    # 将图像转为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 边缘检测
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    # 直线检测
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength=50, maxLineGap=10)
    
    # 计算直线的斜率
    slopes = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 - x1 != 0:
            slope = (y2 - y1) / (x2 - x1)
            slopes.append(slope)
    
    # 找到最常见的斜率
    mode_slope = max(set(slopes), key=slopes.count)
    
    # 计算直线的截距
    intercepts = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 - x1 != 0:
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - slope * x1
            intercepts.append(intercept)
    
    # 找到最常见的截距
    mode_intercept = max(set(intercepts), key=intercepts.count)
    
    # 根据斜率和截距计算条码的位置
    x1 = 0
    y1 = mode_intercept
    x2 = image.shape[1]
    y2 = mode_slope * x2 + mode_intercept
    
    # 在图像上绘制条码位置
    cv2.line(image, (x1, int(y1)), (x2, int(y2)), (0, 0, 255), 2)
    cv2.rectangle(image, (x1, int(y1) - 10), (x2, int(y1) + 10), (0, 0, 255), -1)
    
    return image


@calculate_time
def find_barcode_3(image):
    '''
    :param image: 'image as np.array'
    :return: np.array(dtype=np.uint8)
    '''
    # 转换为灰度图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # # 阈值处理
    # thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)[1]
    #
    # # 形态学开操作
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    # thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    # 边缘检测
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    # 直线检测
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=0, maxLineGap=3)
    
    # 筛选直线
    filtered_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
        length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        if length > 100 and abs(angle) < 10:
            filtered_lines.append(line)
    
    # 绘制直线和标记区域
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 1)
    
    filtered_lines_ndarray = np.asarray(filtered_lines, np.int32)
    x = filtered_lines_ndarray.take([0, 2], 2)
    y = filtered_lines_ndarray.take([1, 3], 2)
    min_x = x.min()
    max_x = x.max()
    min_y = y.min()
    max_y = y.max()
    
    cv2.rectangle(image, (min_x, min_y), (max_x, max_y), (0, 0, 255), 2)
    return image


# Load the image
image = cv2.imread(r'D:\Project\codeScan\test_images\16.png', 1)
result = find_barcode_3(image)
cv2.namedWindow("Barcode detection", cv2.WINDOW_NORMAL)
# Display the image
cv2.imshow("Barcode detection", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
