# -*- coding: UTF-8 -*-
import math

import cv2
import numpy as np

from tools.DecoratorTools import calculate_time
from tools.ImageOperate import clahe_equalize


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


@calculate_time
def template_match_multi(source_img, image, template, angle_step: int = 180, scale_start: float = 0.7, scale_stop: float = 1.3, scale_step: float = 0.2, similarity_threshold: float = 1.0,
						 overlap_threshold=0.5):
	"""
	:param source_img:
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
			result = cv2.matchTemplate(image, template_rotated, cv2.TM_SQDIFF)
			# 找到所有符合阈值的匹配位置
			locs = np.where(result <= similarity_threshold * result.min())
			# 将匹配结果保存到matches列表中
			for pt in zip(*locs):
				matches.append([pt, result[pt[0], pt[1]], angle, scale])
	
	# 对matches按照相似度从大到小排序
	matches = sorted(matches, key=lambda x: x[1], reverse=False)
	
	# 取出得分最高的检测框
	max_loc = matches[0][0]
	cv2.rectangle(source_img, (max_loc[1], max_loc[0]), (max_loc[1] + tw, max_loc[0] + th), (0, 255, 0), 2)
	
	# 遍历matches列表，画出符合重叠度要求的检测框
	for match in matches:
		pt = match[0]
		overlap = compute_overlap(max_loc[1], max_loc[0], tw, th, pt[1], pt[0], tw, th)
		if overlap <= overlap_threshold:
			cv2.rectangle(source_img, (pt[1], pt[0]), (pt[1] + tw, pt[0] + th), (0, 0, 255), 2)
	
	return source_img


# 读取目标图像和模板图像
source_img = cv2.imread(r"D:\fy.xie\fenx\fenx - General\Ubei\Test_Label1\Defect_008.png")
# 生成和原图一样高度和宽度的矩形（全为0）
target_img = np.zeros(source_img.shape, np.uint8)
cv2.copyTo(source_img, mask=None, dst=target_img)

template_img = cv2.imread(r"C:\Users\fy.xie\Desktop\template.png", 0)
target_img = clahe_equalize(target_img)
img_gray = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)
# block_image = block_threshold(img_gray)
img = template_match_multi(source_img, img_gray, template_img, 180, 0.7, 1.3, 0.2, 1.0, 0.3)
# img,_ = template_match_sift(target_img, template_img)
cv2.namedWindow("Barcode Detection", cv2.WINDOW_NORMAL)

cv2.imshow("Barcode Detection", img)
if cv2.waitKey(0) == 27:
	cv2.destroyAllWindows()
