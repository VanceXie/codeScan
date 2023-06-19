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


def filter_bright_spots(image):
	"""
	filter the bright spots in the image, the original image will be changed
	:param image:
	:return:
	"""
	image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	ret, image_threshold0 = cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
	contours, hierarchy = cv2.findContours(image_threshold0, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	lsd = cv2.createLineSegmentDetector()
	lines_list = []
	for contour in contours:
		x, y, w, h = cv2.boundingRect(contour)
		if contour.shape[0] > 50:
			image_gray_part = image_gray[y:y + h, x:x + w]
			# image_threshold = cv2.adaptiveThreshold(image_gray_part, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 0)
			ret1, image_threshold1 = cv2.threshold(image_gray_part, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
			# detect lines_list
			lines, width, prec, nfa = lsd.detect(image_threshold1)
			if lines.shape[0] < 70:
				cv2.drawContours(image, [contour], -1, 0, cv2.FILLED)
			lines_list.append(lines)
		else:
			# color = get_rect_corner_ave(image, x, y, w, h)
			cv2.drawContours(image, [contour], -1, 0, cv2.FILLED)
	return image


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


def get_box_corner_ave(image, box):
	# 确保 box 是一个 numpy.ndarray 类型的数组
	if isinstance(box, np.ndarray):
		# 使用 NumPy 的索引功能一次性提取所有顶点的像素值
		pixel_values = image[box[:, 1], box[:, 0]]
		# 计算每个通道的平均值
		avg_values = np.mean(pixel_values, axis=0)
		# 转为元素数据类型为int的list
		average_pixel_value = avg_values.astype(int).tolist()
		return average_pixel_value
	else:
		raise ValueError("Invalid input. 'box' must be a numpy.ndarray.")


def get_rect_corner_ave(image, x, y, w, h):
	# 提取矩形区域的像素值
	rect_pixels = image[y:y + h, x:x + w]
	# 计算像素值的平均值
	avg_pixel_value = np.mean(rect_pixels, axis=(0, 1))
	# 转为元素数据类型为int的list
	average_pixel_value = avg_pixel_value.astype(int).tolist()
	return average_pixel_value
