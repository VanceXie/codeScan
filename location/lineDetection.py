# -*- coding: UTF-8 -*-
import os.path
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


@calculate_time
def find_barcode_4(image):
	# 灰度化
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	# 读入原始图像
	# gray_thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
	# Scharr算子检测x和y方向梯度
	scharr_x = cv2.Scharr(gray, cv2.CV_64F, 1, 0, borderType=cv2.BORDER_ISOLATED)
	scharr_y = cv2.Scharr(gray, cv2.CV_64F, 0, 1, borderType=cv2.BORDER_ISOLATED)
	
	# edges = (cv2.Canny(gray, 50, 150)).astype(np.uint8)
	# 将梯度值做差
	diff = cv2.subtract(scharr_x, scharr_y)
	
	diff = cv2.convertScaleAbs(diff)
	# # 掩码索引的方法,更节省内存,稍微慢一些
	# threshold = np.percentile(arr, 80)  # 获取前20%的最大数
	# mask = arr > threshold
	# arr[~mask] = 0
	# np.where方法通常比使用掩码索引的方法更快,占用更多的内存
	# # threshold = np.percentile(diff, 93)  # 获取前20%的最大数
	# diff_sift = np.where(diff >= threshold, diff, 0)
	
	# 归一化到0至255作为像素值
	diff_norm = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
	gray_diff = cv2.subtract(gray, diff_norm)
	gray_diff = np.where(gray_diff > 0, gray_diff, 0)
	diff_norm = cv2.normalize(gray_diff, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
	# 均值滤波
	blur = cv2.medianBlur(diff_norm, 7)
	
	# 自适应阈值分割
	_, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
	
	# 形态学开运算去除杂点
	kernel = np.ones((3, 3), np.uint8)
	opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
	
	# 查找轮廓并筛选面积最大的轮廓
	contours, hierarchy = cv2.findContours(opening, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	# cv2.drawContours(image, contours, -1, (0, 0, 255), 1)
	max_area = 0
	best_cnt = None
	for cnt in contours:
		area = cv2.contourArea(cnt)
		if area > max_area:
			max_area = area
			best_cnt = cnt
	
	# 确定条码方向
	x, y, w, h = cv2.boundingRect(best_cnt)
	aspect_ratio = float(w) / h
	rect = cv2.minAreaRect(best_cnt)
	box = cv2.boxPoints(rect)
	box = np.int0(box)
	cv2.drawContours(image, [box], 0, (0, 0, 255), 2)
	if aspect_ratio > 1:
		# 横向条码
		print('横向条码')
	else:
		# 竖向条码
		print('竖向条码')
	
	# 显示结果
	return image


# Load the image
path = r'D:\fy.xie\fenx\fenx - General\Ubei\BarCode'
for index, item in enumerate(os.listdir(path)):
	file = os.path.join(path, item)
	if os.path.isfile(file):
		# image = cv2.imread(file, 1)  # cv2.imread(filename)方法都不支持中文路径的文件读入
		image = cv2.imdecode(np.fromfile(file, dtype=np.uint8), -1)
		result = find_barcode_4(image)
		filename = os.path.splitext(item)
		new_name = filename[0] + f'_{index}' + filename[-1]
		cv2.imwrite(os.path.join(path, new_name), result)

# cv2.namedWindow("Barcode detection", cv2.WINDOW_NORMAL)
# # Display the image
# cv2.imshow("Barcode detection", result)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
