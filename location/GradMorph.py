# -*- coding: UTF-8 -*-

import cv2
import numpy as np

from tools.DecoratorTools import calculate_time


@calculate_time
def find_barcode_by_diff(img):
	# 灰度化
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	# Scharr算子检测x和y方向梯度
	scharr_x = cv2.Scharr(gray, cv2.CV_64F, 1, 0, borderType=cv2.BORDER_ISOLATED)
	scharr_y = cv2.Scharr(gray, cv2.CV_64F, 0, 1, borderType=cv2.BORDER_ISOLATED)
	
	# 将梯度值做差
	diff_gradient = cv2.subtract(scharr_x, scharr_y)
	# diff = np.abs(diff)
	diff_gradient = cv2.convertScaleAbs(diff_gradient)
	# diff = np.where(diff > 0, diff, 0)
	
	# 归一化到0至255作为像素值
	norm_diff_gradient = cv2.normalize(diff_gradient, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
	
	cv2.imshow('norm_diff_gradient', norm_diff_gradient)
	
	# 形态学开运算去除白色杂点
	kernel = np.ones((3, 3), np.uint8)
	opening_norm_diff_gradient = cv2.morphologyEx(norm_diff_gradient, cv2.MORPH_OPEN, kernel, iterations=3)
	
	diff_gray = cv2.subtract(gray, opening_norm_diff_gradient)
	# diff_gray=np.abs(diff_gray)
	diff_gray = cv2.convertScaleAbs(diff_gray)
	# diff_gray = np.where(diff_gray > 0, diff_gray, 0)
	norm_diff_gray = cv2.normalize(diff_gray, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
	
	cv2.imshow('norm_diff_gray', norm_diff_gray)
	# 均值滤波
	blur_norm_diff_gray = cv2.medianBlur(norm_diff_gray, 5)
	
	# 自适应阈值分割
	_, thresh = cv2.threshold(blur_norm_diff_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
	# thresh = cv2.adaptiveThreshold(blur_norm_diff_gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,5,0)
	# # 形态学开运算去除白色杂点
	# kernel = np.ones((5, 5), np.uint8)
	# opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel,iterations=3)
	
	# 查找轮廓并筛选面积最大的轮廓
	contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	filtered_contours = [contours[i] for i in range(len(contours)) if hierarchy[0][i][3] == -1]
	
	# 先做形状筛选
	# # Filter out contours with an aspect ratio greater than 4
	aspect_ratio_filtered_contours = [contour for contour in filtered_contours if 6.5 > float(cv2.boundingRect(contour)[2]) / cv2.boundingRect(contour)[3] >= 3.5]  # 上述代码的列表推导式
	
	# 再做面积筛选
	# Get the maximum contour area
	max_area = max([cv2.contourArea(c) for c in aspect_ratio_filtered_contours])
	# Filter out contours with an area greater than 0.5 * max_area
	area_filtered_contours = [c for c in aspect_ratio_filtered_contours if cv2.contourArea(c) > 0.5 * max_area]
	# 画条码框
	try:
		# for best_cnt in area_filtered_contours:
		# 	rect = cv2.minAreaRect(best_cnt)
		# 	box = cv2.boxPoints(rect)
		# 	box = np.int0(box)
		# 	cv2.drawContours(img, [box], 0, (0, 0, 255), 2)
		[cv2.drawContours(img, [np.int0(cv2.boxPoints(cv2.minAreaRect(cnt)))], 0, (0, 0, 255), 2) for cnt in area_filtered_contours]
	except:
		pass
	
	# 显示结果
	return img
