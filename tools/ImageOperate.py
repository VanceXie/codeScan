# -*- coding: UTF-8 -*-

import cv2
import numpy as np
import concurrent.futures
from matplotlib import pyplot as plt
from scipy.stats import norm
from sympy import S, diff, solveset, symbols

from tools.DecoratorTools import calculate_time


class ImageEqualize:
	def __init__(self, image):
		self.image = image
	
	@calculate_time
	def clahe_equalize(self):
		lab = cv2.cvtColor(self.image, cv2.COLOR_BGR2LAB)
		# 将LAB色彩空间的L通道分离出来
		l, a, b = cv2.split(lab)
		# 创建CLAHE对象
		clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(1, 4))
		# 对L通道进行CLAHE均衡化
		l_clahe = clahe.apply(l)
		# 将CLAHE均衡化后的L通道合并回LAB色彩空间
		lab_clahe = cv2.merge((l_clahe, a, b))
		# 将LAB色彩空间转换回BGR色彩空间
		self.image = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
		return self
	
	@calculate_time
	def adaptive_index_equalize(self):
		imax = np.max(self.image)
		self.image = (255 ** (self.image / imax)).astype(np.uint8)
		return self


class Sharpen:
	def __init__(self, image):
		self.image = image
	
	@calculate_time
	def mask_sharpen(self, kernel_size: tuple = (5, 5)):
		blurred = cv2.GaussianBlur(self.image, kernel_size, 0, cv2.CV_32F)
		res = self.image.astype(np.float32) - blurred
		
		# 增强后的图像
		image_add = self.image + res
		self.image = np.clip(image_add, 0, 255).astype(np.uint8)
		return self
	
	def frequency_enhancement(self):
		pass


@calculate_time
def filter_bright_spots(image_gray, lsd: cv2.LineSegmentDetector, lines_num: int = 70):
	"""
	filter the bright spots in the image, the original image will be changed
	:param image_gray:
	:param lsd:
	:param lines_num:
	:return:
	"""
	# 阈值分割
	r, t = cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
	# 开闭运算
	kernel = np.ones((3, 3), np.uint8)
	closing = cv2.morphologyEx(t, cv2.MORPH_CLOSE, kernel)  # 闭运算
	# opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)  # 开运算
	# 面积阈值
	area_threshold = int(image_gray.shape[0] * image_gray.shape[1] * 0.015)
	# 寻找轮廓
	contours, hierarchy = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	contour_list = []
	for contour in contours:
		x, y, w, h = cv2.boundingRect(contour)
		if cv2.contourArea(contour) > area_threshold:
			image_gray_part = image_gray[y:y + h, x:x + w]
			ret, image_threshold = cv2.threshold(image_gray_part, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
			# detect lines
			lines, width, prec, nfa = lsd.detect(image_threshold)
			if lines is not None:
				if lines.shape[0] < lines_num:
					cv2.drawContours(image_gray, [contour], -1, 0, cv2.FILLED)
				else:
					contour_list.append(contour)
		else:
			cv2.drawContours(image_gray, [contour], -1, 0, cv2.FILLED)
	return contour_list


@calculate_time
def pyrdown_multithread(image_source: np.ndarray, pyr_levels: int = 2) -> list:
	if not isinstance(image_source, np.ndarray):
		raise TypeError("Input image must be a valid ndarray.")
	if not isinstance(pyr_levels, int):
		raise TypeError("Pyramid level must be an integer.")
	
	# Initialize thread pool executor
	executor = concurrent.futures.ThreadPoolExecutor()
	
	# Generate gauss_pyramid levels in parallel
	pyramid = [image_source]
	for level in range(0, pyr_levels):
		future = executor.submit(lambda img: (cv2.pyrDown(img)), pyramid[-1])
		image = future.result()
		pyramid.append(image)
	
	# Shutdown the executor
	executor.shutdown()
	
	return pyramid


@calculate_time
def build_pyramid(image: np.ndarray, levels: int = 3, layer: int = 2):
	"""

	:param image: ndarray of image
	:param levels: Number of Pyramid Layers, down sampling times
	:param layer: layer index of the image pyramid to get
	:return: laplacian_pyramid including a low-resolution image of the original image
	"""
	if not isinstance(image, np.ndarray):
		raise TypeError("Input image must be a valid ndarray.")
	if not isinstance(levels, int):
		raise TypeError("Pyramid level must be an integer.")
	
	gaussian_pyramid = [None] * (levels + 1)
	gaussian_pyramid[0] = image
	
	laplacian_pyramid = [None] * (levels + 1)
	current_level = image
	
	for level in range(levels):
		# Gaussian pyramid formed by downsampling
		downsampled = cv2.pyrDown(current_level)
		gaussian_pyramid[level + 1] = downsampled
		# Gaussian pyramid formed by upsampling
		upsampled = cv2.pyrUp(downsampled, dstsize=current_level.shape[:2][::-1])
		# Laplace Pyramid
		laplacian_pyramid[level] = cv2.subtract(current_level, upsampled)
		
		# The low resolution pyramid layer to fetch, with edge enhance
		if level == layer:
			laplacian_ = laplacian_pyramid[level]
			enhanced = cv2.add(current_level, laplacian_)
		
		current_level = downsampled
	laplacian_pyramid[levels] = current_level
	
	return gaussian_pyramid, laplacian_pyramid, enhanced


def reconstruct_from_laplacian_pyramid(laplacian_pyramid, layer):
	"""
	:param laplacian_pyramid: laplacian_pyramid including a low-resolution image of the original image at the end of the list
	:param layer: layer index of the image pyramid to get
	:return:
	"""
	levels = len(laplacian_pyramid) - 1
	reconstructed = laplacian_pyramid[levels]
	for i in range(levels - 1, layer - 1, -1):
		upsampled = cv2.pyrUp(reconstructed, dstsize=laplacian_pyramid[i].shape[:2][::-1])
		reconstructed = cv2.add(upsampled, laplacian_pyramid[i])
	return reconstructed


@calculate_time
def downsample_with_edge_preservation(image: np.ndarray, scale_factor: int = 3):
	"""
	:param image:
	:param scale_factor: scaling ratio, 1/scale_factor
	:return: downsampled_image
	"""
	# 创建 gaussian 卷积核
	sigma = 0.1
	size = int(2 * np.ceil(3 * sigma) + 1)
	kernel_gaussian = cv2.getGaussianKernel(size, sigma)
	# 使用卷积核进行下采样
	smoothed_image = cv2.sepFilter2D(image, -1, kernel_gaussian.T, kernel_gaussian)
	return smoothed_image[::scale_factor, ::scale_factor]


def get_original_location(point_coordinates: np.ndarray, pyramid_order: int) -> np.ndarray:
	"""
	Get the position of the point on the original image from the downsampled image
	:param point_coordinates: coordinates of points on downsampled image, np.ndarray
	:param pyramid_order: The order of the graph gauss_pyramid, non-negative integer
	:return: coordinates of points on original image, np.ndarray
	"""
	if not isinstance(pyramid_order, int) or pyramid_order < 0:
		raise ValueError("Pyramid order must be a non-negative integer.")
	if pyramid_order != 0:
		point_coordinates = point_coordinates * 2 - 1
	return point_coordinates


@calculate_time
def block_threshold(image, block_size=101):
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


def hist_cut_by_mutation(img, mutation_quantity):
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


def hist_normalized(img):
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


def contour_filter(image_gray: np.ndarray, lsd: cv2.LineSegmentDetector, contours, area_threshold: int, lines_num_threshold: int):
	contour_list = []
	for contour in contours:
		x, y, w, h = cv2.boundingRect(contour)
		if cv2.contourArea(contour) > area_threshold:
			
			image_gray_part = image_gray[y:y + h, x:x + w]
			
			ret2, image_threshold = cv2.threshold(image_gray_part, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
			# detect lines_list
			lines, width, prec, nfa = lsd.detect(image_threshold)
			if lines is not None:
				if lines.shape[0] < lines_num_threshold:
					cv2.drawContours(image_gray, [contour], -1, 0, cv2.FILLED)
				else:
					contour_list.append(contour_list)
		else:
			cv2.drawContours(image_gray, [contour], -1, 0, cv2.FILLED)
	return contour_list
