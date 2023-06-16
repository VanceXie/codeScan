# -*- coding: UTF-8 -*-

import cv2
import numpy as np
import concurrent.futures
from matplotlib import pyplot as plt
from scipy.stats import norm
from sympy import S, diff, solveset, symbols

from tools.DecoratorTools import calculate_time


def clahe_equalize(image_bgr: np.ndarray):
	"""
	:param image_bgr:ndarray of image
	:return: bgr_clahe, image applied by clahe
	"""
	lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
	# 将LAB色彩空间的L通道分离出来
	l, a, b = cv2.split(lab)
	# 创建CLAHE对象
	clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(3, 15))
	# 对L通道进行CLAHE均衡化
	l_clahe = clahe.apply(l)
	# 将CLAHE均衡化后的L通道合并回LAB色彩空间
	lab_clahe = cv2.merge((l_clahe, a, b))
	# 将LAB色彩空间转换回BGR色彩空间
	bgr_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
	return bgr_clahe


def filter_small_bright_spots(image, area_threshold):
	"""
	filter small bright spots
	:param image: image
	:param area_threshold: area threshold of abnormal bright spot
	:return:
	"""
	# 将图像转换为灰度图像
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	
	# 应用自适应阈值化，将图像转换为二值图像
	thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
	
	# 对二值图像执行连通组件分析
	connectivity = 8  # 连通性为8，考虑八个邻域像素
	output = cv2.connectedComponentsWithStats(thresh, connectivity, cv2.CV_32S)
	
	# 获取连通组件的标签和统计信息
	num_labels = output[0]
	labels = output[1]
	stats = output[2]
	
	# 遍历每个连通区域，根据面积阈值进行筛选
	for label in range(1, num_labels):
		area = stats[label, cv2.CC_STAT_AREA]
		if area < area_threshold:
			# 将小面积连通区域设为背景（黑色）
			labels[labels == label] = 0
	
	# 重新将图像转换为彩色
	filtered_image = cv2.cvtColor(labels.astype(np.uint8), cv2.COLOR_GRAY2BGR)
	
	return filtered_image


@calculate_time
def pyrdown_multithread(image_source: np.ndarray, pyr_levels: int = 2) -> list:
	if not isinstance(image_source, np.ndarray):
		raise TypeError("Input image must be a valid ndarray.")
	if not isinstance(pyr_levels, int):
		raise TypeError("Pyramid level must be an integer.")
	
	# Initialize thread pool executor
	executor = concurrent.futures.ThreadPoolExecutor()
	
	# Generate pyramid levels in parallel
	pyramid = [image_source]
	for level in range(0, pyr_levels):
		future = executor.submit(lambda img: (cv2.pyrDown(img)), pyramid[-1])
		image = future.result()
		pyramid.append(image)
	
	# Shutdown the executor
	executor.shutdown()
	
	return pyramid


@calculate_time
def pyrdown(image_source: np.ndarray, pyr_levels: int = 2) -> list:
	"""
	Downsample to get the list of graph pyramid
	:param image_source: ndarray of image
	:param pyr_levels: The order of the graph pyramid
	:return: list of graph pyramid
	"""
	if not isinstance(image_source, np.ndarray):
		raise TypeError("Input image must be a valid ndarray.")
	if not isinstance(pyr_levels, int):
		raise TypeError("Pyramid level must be an integer.")
	# Preallocate pyramid list
	pyramid = [None] * (pyr_levels + 1)
	pyramid[0] = image_source
	# Generate pyramid
	image = image_source
	for level in range(1, pyr_levels + 1):
		image = cv2.pyrDown(image)
		pyramid[level] = image.copy()
	
	return pyramid


def get_original_location(point_coordinates: np.ndarray, pyramid_order: int) -> np.ndarray:
	"""
	Get the position of the point on the original image from the downsampled image
	:param point_coordinates: coordinates of points on downsampled image, np.ndarray
	:param pyramid_order: The order of the graph pyramid, non-negative integer
	:return: coordinates of points on original image, np.ndarray
	"""
	if not isinstance(pyramid_order, int) or pyramid_order < 0:
		raise ValueError("Pyramid order must be a non-negative integer.")
	if pyramid_order != 0:
		point_coordinates = point_coordinates * 2 - 1
	return point_coordinates


@calculate_time
def block_threshold(image, block_size=500):
	# 计算图像大小
	height, width = image.shape
	
	# 计算行列
	rows = np.uint32(np.ceil(height / block_size))
	cols = np.uint32(np.ceil(width / block_size))
	final_image = np.zeros((height, width), dtype=np.uint8)
	# 循环处理每个块
	for r in range(rows):
		for c in range(cols):
			# 计算块的边界
			row_start = r * block_size
			col_start = c * block_size
			row_end = min(row_start + block_size, height)
			col_end = min(col_start + block_size, width)
			
			# 获取当前块的图像
			current_block = image[row_start:row_end, col_start:col_end]
			
			# 计算块的阈值
			thresh = cv2.threshold(current_block, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
			
			# 将二值化后的块拼合起来
			final_image[r * block_size:block_size * (r + 1), c * block_size:block_size * (c + 1)] = thresh
	return final_image


def get_threshold_by_convexity(hist, range_min, range_max, exponent):
	"""
	（效果不佳，弃用）根据直方图拟合曲线的凹凸性确定阈值
	:param hist: 根据图片计算出的灰度直方图
	:param exponent: 拟合曲线的阶数
	:return: 0-255 范围内最右侧第一个凹凸性变化点
	"""
	
	# 定义多项式函数
	def func(variable, coefficients, exponent_max):
		X = np.vander(variable, exponent_max + 1, increasing=True)  # 生成x的多个幂次的ndarray，从左到右，分别为[x^0,x^1,x^2...]
		
		f = np.dot(X, coefficients[::-1])  # 拟合所得原始参数是从高幂到低幂的系数，所以做一次翻转操作
		# # 获取求导后的系数
		# exponents = np.arange(exponent_max, 0, -1)
		# arr1 = exponents[:-2]
		# arr2 = exponents[1:-1]
		# arr3 = exponents[2:]
		# result = arr1 * arr2 * arr3
		# d3_f = np.dot(X[:, :exponent_max - 2], coefficients[::-1][3:] * result[::-1])
		return f
	
	def fit_func(variance, coeffs):
		f = np.polyval(coeffs, variance)
		return f
	
	# 生成样本数据
	xdata = np.arange(range_min, range_max + 1, 1)
	
	popt = np.polyfit(xdata, hist, exponent)
	
	x = symbols('x')
	d2_fit_func = diff(fit_func(x, popt), x, 2)
	solveset(d2_fit_func, x, domain=S.Reals)
	solution = np.array([solveset(d2_fit_func, x, domain=S.Reals)])
	print(solution)
	plt.bar(xdata, hist)
	plt.plot(xdata, fit_func(xdata, popt), color='r')
	plt.show()
	
	return int(solution[(solution >= 0) & (solution <= 255)][-1])


def hist_cut(img, mutation_quantity):
	"""
	:param img: 3-D ndarray
	:param mutation_quantity: 突变量
	:return: 3-D ndarray,image
	"""
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	# 计算灰度直方图
	hist, bins = np.histogram(gray.flatten(), 256, [0, 256])
	# hist = np.bincount(gray.flatten())
	# 去除灰度值最高且占比较少的的几个直方图
	# 计算直方图高度变化
	difference = np.diff(hist)
	index = np.where(np.abs(difference) > mutation_quantity)[0][-1]
	# 截断之前的大灰度值像素
	img[img > index] = 0
	return img


def hist_remap(img):
	# 统计灰度直方图
	hist, bins = np.histogram(img.ravel(), 256, [0, 256])
	# 计算直方图的均值和标准差
	mean = np.mean(img)
	std = np.std(img)
	# 映射灰度直方图到正态分布
	x = (bins - mean) / std
	remap_hist = norm.cdf(x)
	# 计算像素值映射表
	lut = np.uint8(remap_hist * 255)[:-1]
	# 应用像素值映射表，输出更改分布后的图像
	remap_img = cv2.LUT(img, lut)
	return remap_img
