# -*- coding: UTF-8 -*-
import math

import cv2
import numpy as np

from tools import *


def compute_overlap(x1, y1, w1, h1, x2, y2, w2, h2):
	# 计算矩形框的面积
	area1 = w1 * h1
	area2 = w2 * h2
	# 计算相交部分的坐标
	x_inter = max(x1, x2)
	y_inter = max(y1, y2)
	w_inter = min(x1 + w1, x2 + w2) - x_inter
	h_inter = min(y1 + h1, y2 + h2) - y_inter
	# 计算相交部分的面积
	if w_inter <= 0 or h_inter <= 0:
		return 0.0
	area_inter = w_inter * h_inter
	# 计算并集的面积
	area_union = area1 + area2 - area_inter
	# 计算重叠度
	overlap = area_inter / area_union
	return overlap


def rotate_and_scale(image, angle, scale):
	"""
	旋转和比例不变变换
	:param image: 输入图像
	:param angle: 旋转角度
	:param scale: 缩放比例
	:return:
	"""
	img_resized = cv2.resize(image, (int(scale * image.shape[1]), int(scale * image.shape[0])))
	rows, cols = img_resized.shape[:2]
	# 构造旋转变换矩阵
	M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
	cols_new = int(cols * math.cos(angle) + rows * math.sin(angle))
	rows_new = int(cols * math.sin(angle) + rows * math.cos(angle))
	# 进行旋转变换
	img_rotated = cv2.warpAffine(img_resized, M, (cols_new, rows_new))
	return img_rotated


@PerformanceEval.calculate_time
def template_match_multi(image, template, angle_step: int = 30, scale_start: float = 0.5, scale_stop: float = 1.0, scale_step: float = 0.2, similarity_threshold: float = 1.0, overlap_threshold=0.3):
	"""
	:param image:
	:param template:
	:param angle_step: 旋转步长
	:param scale_start: 放缩起始
	:param scale_stop: 放缩中止
	:param scale_step: 放缩步长
	:param similarity_threshold:相似度阈值
	:param overlap_threshold: 重叠阈值
	:return:
	"""
	# 获取模板图像的大小
	th, tw = template.shape[:2]
	# 初始化最大匹配度和对应的位置
	matches = []
	# 保存保留的检测框索引
	# 在一定范围内进行旋转和比例不变变换，并计算相似度
	for angle in range(0, 360, angle_step):
		for scale in np.arange(scale_start, scale_stop, scale_step):
			# 进行旋转和比例不变变换
			template_rotated = rotate_and_scale(template, angle, scale)
			# 计算相似度
			result = cv2.matchTemplate(image, template_rotated, cv2.TM_CCOEFF_NORMED)
			# 找到所有符合阈值的匹配位置
			locs = np.where(result <= similarity_threshold * result.min())
			# 将匹配结果保存到matches列表中
			for pt in zip(*locs):
				matches.append([pt, result[pt[0], pt[1]], angle, scale])
	
	# 对matches按照相似度从大到小排序
	matches = sorted(matches, key=lambda x: x[1], reverse=False)
	
	# 取出得分最高的检测框
	max_loc = matches[0][0]
	cv2.rectangle(image, (max_loc[1], max_loc[0]), (max_loc[1] + tw, max_loc[0] + th), (0, 0, 255), 2)
	
	# 遍历matches列表，画出符合重叠度要求的检测框
	for match in matches:
		pt = match[0]
		overlap = compute_overlap(max_loc[1], max_loc[0], tw, th, pt[1], pt[0], tw, th)
		if overlap <= overlap_threshold:
			cv2.rectangle(image, (pt[1], pt[0]), (pt[1] + tw, pt[0] + th), (0, 0, 255), 2)
	
	return image


def template_match_sift(image, template, threshold=0.75, draw_result=True):
	"""
	:param image:
	:param template:
	:param threshold:
	:param draw_result:
	:return:
	"""
	
	# Convert images to grayscale
	gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
	
	# Initialize SIFT detector and FLANN matcher
	sift = cv2.xfeatures2d.SIFT_create()
	matcher = cv2.FlannBasedMatcher_create()
	
	# Extract key_points and descriptors from template and image
	key_points_template, descriptors_template = sift.detectAndCompute(gray_template, None)
	key_points_image, descriptors_image = sift.detectAndCompute(gray_image, None)
	
	# Match descriptors using FLANN matcher
	matches = matcher.knnMatch(descriptors_template, descriptors_image, k=2)
	
	# Filter matches by Lowe's ratio test
	good_matches = []
	for m, n in matches:
		if m.distance < 0.4 * n.distance:
			good_matches.append(m)
	
	# Compute homography matrix using RANSAC algorithm
	src_pts = np.float32([key_points_template[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
	dst_pts = np.float32([key_points_image[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
	M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
	
	# Apply perspective transform to template
	h, w = template.shape[:2]
	template_warped = cv2.warpPerspective(template, M, (w, h))
	
	# Match warped template to image using normalized cross-correlation
	result = cv2.matchTemplate(gray_image, cv2.cvtColor(template_warped, cv2.COLOR_BGR2GRAY), cv2.TM_CCOEFF_NORMED)
	
	# Find locations of matched regions above threshold
	locations = np.where(result >= threshold)
	locations = list(zip(*locations[::-1]))
	
	# Draw bounding boxes around matched regions
	if draw_result:
		for pt in locations:
			cv2.rectangle(image, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)
	
	return image, locations


# 读取目标图像和模板图像
target_img = cv2.imread(r"D:\fy.xie\fenx\fenx - General\Ubei\Test_Label1\Defect_008.png")
template_img = cv2.imread(r"D:\fy.xie\fenx\fenx - General\Ubei\Test_Label1\template.png")
target_img = ImageOperate.img_equalize(target_img)
img = template_match_multi(target_img, template_img, 180, 0.5, 1.0, 0.2, 1.0, 0.1)
# img,_ = template_match_sift(target_img, template_img)
cv2.namedWindow("Barcode Detection", cv2.WINDOW_NORMAL)

cv2.imshow("Barcode Detection", img)
if cv2.waitKey(0) == 27:
	cv2.destroyAllWindows()
