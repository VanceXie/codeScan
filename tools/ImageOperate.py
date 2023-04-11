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


# image = cv2.imread(r"D:\fy.xie\fenx\fenx - General\Ubei\Test_Label1\Defect_035.png", 1)
# iamge_equalized = img_equalize(image)
# gray = cv2.cvtColor(iamge_equalized, cv2.COLOR_BGR2GRAY)
# _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# cv2.namedWindow("Barcode detection", cv2.WINDOW_NORMAL)
# # Display the image
# cv2.imshow("Barcode detection", thresh)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

def sharpen(img):
	kernel = np.array([[-1, -1, -1],
					   [-1, 9, -1],
					   [-1, -1, -1]])
	
	# 应用锐化算子
	sharpened = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=1, borderType=cv2.BORDER_CONSTANT)
	sharpened_abs = cv2.convertScaleAbs(sharpened)
	# sharpened_norm = cv2.normalize(sharpened_abs, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
	return sharpened_abs
	
	image = cv2.imread(r"D:\Project\codeScan\location\pic\Defect_035.png", 1)
	sharpened_img = sharpen(image)
	blured_img = cv2.GaussianBlur(sharpened_img, (5, 5), 0)
	result = cv2.vconcat([image, sharpened_img, blured_img])
	result_n = cv2.normalize(sharpened_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
	cv2.namedWindow('Image', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
	cv2.imshow('Image', result_n)
	cv2.waitKey(0)
	cv2.destroyAllWindows()


# def CannyThreshold(lowThreshold):
#     detected_edges = cv2.GaussianBlur(gray, (5, 5), 0)
#     detected_edges = cv2.Canny(detected_edges, lowThreshold, lowThreshold * ratio, apertureSize=kernel_size)
#     dst = cv2.bitwise_and(img, img, mask=detected_edges)
#
#     dst = np.concatenate([img, dst], 0)
#     cv2.imshow('canny demo', dst)


# lowThreshold = 35
# max_lowThreshold = 200
# ratio = 3
# kernel_size = 3
#
# img = cv2.imread('.\\Images\\Input\\02.jpg')
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
# cv2.namedWindow('canny demo')
# cv2.createTrackbar('Min threshold', 'canny demo', lowThreshold, max_lowThreshold, CannyThreshold)
#
# CannyThreshold(lowThreshold)  # initialization
# if cv2.waitKey(0) == 27:
#     cv2.destroyAllWindows()
def pyr_down(image, levels=3):
	"""
	:param image:
	:param levels:
	:return:
	"""
	pyramid = [image]
	for i in range(levels):
		# 降采样
		image = cv2.pyrDown(image)
		pyramid.append(image)
	return pyramid
