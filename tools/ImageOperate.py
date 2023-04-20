# -*- coding: UTF-8 -*-
import cv2
import numpy as np


def img_equalize(img):
	lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
	# 将LAB色彩空间的L通道分离出来
	l, a, b = cv2.split(lab)
	# 创建CLAHE对象
	clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(15, 3))
	# 对L通道进行CLAHE均衡化
	l_clahe = clahe.apply(l)
	# 将CLAHE均衡化后的L通道合并回LAB色彩空间
	lab_clahe = cv2.merge((l_clahe, a, b))
	# 将LAB色彩空间转换回BGR色彩空间
	bgr_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
	return bgr_clahe


def sharpen(img):
	kernel = np.array([[-1, -1, -1],
					   [-1, 9, -1],
					   [-1, -1, -1]])
	
	# 应用锐化算子
	sharpened = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=1, borderType=cv2.BORDER_CONSTANT)
	sharpened_abs = cv2.convertScaleAbs(sharpened)
	# sharpened_norm = cv2.normalize(sharpened_abs, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
	return sharpened_abs


def pyr_down(image, pyr_levels=3):
	"""
	:param image:
	:param pyr_levels:
	:return:
	"""
	pyramid = [image]
	for i in range(pyr_levels):
		# 降采样
		image = cv2.pyrDown(image)
		pyramid.append(image)
	return pyramid


def get_location_original(location, pyr_levels):
	if pyr_levels == 0:
		return location
	else:
		location = location * 2 - 1
		return get_location_original(location, pyr_levels - 1)
