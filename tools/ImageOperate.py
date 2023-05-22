# -*- coding: UTF-8 -*-
import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import norm
from sympy import S, diff, solveset, symbols

from tools.PerformanceEval import calculate_time


def img_equalize(image_rgb):
	"""
	:param image_rgb:
	:return:
	"""
	lab = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2LAB)
	# 将LAB色彩空间的L通道分离出来
	l, a, b = cv2.split(lab)
	# 创建CLAHE对象
	clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(3, 15))
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


@calculate_time
def get_threshold(hist):
	# 计算直方图高度变化
	# 定义四阶多项式函数
	def func(x, popt, exponent_max):
		X = np.vander(x, exponent_max + 1, increasing=True)  # 生成x的多个幂次的ndarray，从左到右，分别为[x^0,x^1,x^2...]
		f = np.dot(X, popt[::-1])  # 拟合所得原始参数是从高幂到低幂的系数，所以做一次翻转操作
		# # 获取求导后的系数
		# exponents = np.arange(exponent_max, 0, -1)
		# arr1 = exponents[:-2]
		# arr2 = exponents[1:-1]
		# arr3 = exponents[2:]
		# result = arr1 * arr2 * arr3
		# d3_f = np.dot(X[:, :exponent_max - 2], popt_reverse[3:] * result[::-1])
		return f
	
	def fit_func(x, coeffs):
		f = np.polyval(coeffs, x)
		return f
	
	# 生成样本数据
	xdata = np.arange(256.0)
	
	exponent = 6
	popt = np.polyfit(xdata, hist, exponent)
	
	x = symbols('x')
	d2_fit_func = diff(fit_func(x, popt), x, 2)
	solution = np.asarray(list(solveset(d2_fit_func, x, domain=S.Reals)))
	return int(solution[(solution >= 0) & (solution <= 255)][-1])


@calculate_time
def hist_cut(img):
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	# 计算灰度直方图
	hist, bins = np.histogram(gray.flatten(), 256, [0, 256])
	# hist = np.bincount(gray.flatten())
	# 去除灰度值最高且占比较少的的几个直方图
	image_cut = np.copy(img)
	threshold = get_threshold(hist)
	# 截断之前的大灰度值像素
	image_cut[image_cut > threshold] = 0
	# 绘制原始灰度直方图
	plt.subplot(1, 2, 1)
	plt.hist(gray.ravel(), 256, [0, 256], color='r')
	plt.xlim([0, 256])
	plt.xlabel('Gray Level')
	plt.ylabel('Number of Pixels')
	plt.title('Original Histogram')
	# 绘制去除部分直方图后的灰度直方图
	plt.subplot(1, 2, 2)
	plt.hist(image_cut.ravel(), 256, [0, 256], color='g')
	plt.xlim([0, 256])
	plt.xlabel('Gray Level')
	plt.ylabel('Normalized Number of Pixels')
	plt.title('cutted Histogram')
	plt.show()


# 显示去除部分直方图的图像
# cv2.imshow('Removed Image', image_cut)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# return img_cutted


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
	
	# norm_img = cv2.normalize(remap_img, None, 0, 255, cv2.NORM_MINMAX)
	# cv2.imshow("Remap Image", remap_img)
	#
	# # 绘制原始图像的灰度直方图和更改分布后的灰度直方图
	# plt.subplot(2, 2, 1)
	# plt.hist(img.ravel(), 256, [0, 256])
	# plt.title("Original Histogram")
	# plt.subplot(2, 2, 2)
	# plt.hist(remap_img.ravel(), 256, [0, 256])
	# plt.title("Remapped Histogram")
	#
	# plt.subplot(2, 2, 3)
	# plt.xlim()
	# plt.ylim()
	# plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
	# plt.subplot(2, 2, 4)
	# plt.xlim()
	# plt.ylim()
	# plt.imshow(cv2.cvtColor(remap_img, cv2.COLOR_BGR2RGB))
	# plt.show()
	#
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()
	return remap_img


file = r"D:\Fenkx\Fenkx - General\Ubei\Test_Label1\Over_004.png"
image_source = cv2.imdecode(np.fromfile(file, dtype=np.uint8), 1)
hist_cut(image_source)
# image_eq = img_equalize(img_cutted)
# cv2.namedWindow('result', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
# cv2.imshow('result', cv2.vconcat((img_cutted, image_eq)))
# if cv2.waitKey(0) == 27:
# 	cv2.destroyAllWindows()
