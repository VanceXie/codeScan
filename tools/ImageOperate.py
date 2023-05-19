# -*- coding: UTF-8 -*-
import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import norm

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


def hist_cut(img):
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	# 计算灰度直方图
	hist, bins = np.histogram(gray.flatten(), 256, [0, 256])
	# 去除灰度值最高且占比较少的的几个直方图
	
	img_cutted = np.copy(img)
	# 计算直方图高度变化
	diff = np.diff(hist)
	index = np.where(np.abs(diff) > 500)[0][-1]
	# 截断之前的大灰度值像素
	img_cutted[img_cutted > index] = 0
	image_eq = img_equalize(img_cutted)
	# 绘制原始灰度直方图
	plt.subplot(1, 2, 1)
	plt.hist(gray.ravel(), 256, [0, 256], color='r')
	plt.xlim([0, 256])
	plt.xlabel('Gray Level')
	plt.ylabel('Number of Pixels')
	plt.title('Original Histogram')
	# 绘制去除部分直方图后的灰度直方图
	plt.subplot(1, 2, 2)
	plt.hist(image_eq.ravel(), 256, [0, 256], color='g')
	plt.xlim([0, 256])
	plt.xlabel('Gray Level')
	plt.ylabel('Normalized Number of Pixels')
	plt.title('cutted Histogram')
	plt.show()
	# 显示去除部分直方图的图像
	cv2.imshow('Removed Image', image_eq)
	cv2.waitKey(0)
	cv2.destroyAllWindows()


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


file = r'D:\Fenkx\Fenkx - General\Ubei\Test_Label1\0211111238_NG_BarCode_Camera3_0211111238.jpg'
image_source = cv2.imdecode(np.fromfile(file, dtype=np.uint8), 1)
img_cutted = hist_cut(image_source)
# image_eq = img_equalize(img_cutted)
# cv2.namedWindow('result', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
# cv2.imshow('result', cv2.vconcat((img_cutted, image_eq)))
# if cv2.waitKey(0) == 27:
# 	cv2.destroyAllWindows()
