# -*- coding: UTF-8 -*-
import cv2
import numpy as np

from tools.ImageOperate import block_threshold, img_equalize
from tools.PerformanceEval import calculate_time


def custom_match_template(image, template, cost_func):
	h, w = image.shape[:2]
	th, tw = template.shape[:2]
	# 计算代价矩阵
	# cost = np.zeros((h - th + 1, w - tw + 1))
	# for i in range(h - th + 1):
	# 	for j in range(w - tw + 1):
	# 		patch = image[i:i + th, j:j + tw]
	# 		cost[i, j] = cost_func(patch, template)
	cost = [[cost_func(image[i:i + th, j:j + tw], template) for j in range(w - tw + 1)] for i in range(h - th + 1)]
	# 找到最佳匹配位置
	min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(cost)
	# if cost_func == cv2.TM_SQDIFF or cost_func == cv2.TM_SQDIFF_NORMED:
	# 	match_loc = min_loc
	# else:
	# 	match_loc = max_loc
	return min_loc


# 自定义代价函数
@calculate_time
def custom_cost_func(image_patch, template):
	template_shape = template.shape
	h, w = template_shape
	mask_one = np.ones(template_shape, dtype=np.uint8)
	mask_zero = np.zeros(template_shape, dtype=np.uint8)
	# 定义矩形区域的左上角和右下角坐标
	pt1 = (int(0.08 * w), int(0.1 * h))
	pt2 = (int(0.86 * w), int(0.92 * h))
	# 在灰度图像上绘制矩形区域
	cv2.rectangle(mask_one, pt1, pt2, (0), -1)
	cv2.rectangle(mask_zero, pt1, pt2, (1), -1)
	
	__diff = template - image_patch
	diff_bar = np.abs(np.sum(mask_zero * __diff))
	diff_space = np.sum(np.abs(mask_one * __diff))
	return diff_bar + diff_space


# 测试
image = cv2.imread(r"D:\fy.xie\fenx\fenx - General\Ubei\Test_Label1\Defect_008.png")
target_img = img_equalize(image)
img_gray = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)
block_image = block_threshold(img_gray)

template = cv2.imread(r"../location/template.png", 0)
match_loc = custom_match_template(block_image, template, custom_cost_func)
cv2.rectangle(image, match_loc, (match_loc[0] + template.shape[1], match_loc[1] + template.shape[0]), (0, 255, 0), 2)
cv2.imshow('Match Result', image)
cv2.waitKey(0)
