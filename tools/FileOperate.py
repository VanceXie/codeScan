# -*- coding: UTF-8 -*-
import os
from collections import defaultdict

import cv2
import numpy as np

from tools.DecoratorTools import calculate_time


@calculate_time
def get_files(path, include_subdir=False):
	"""
	获取当前路径下的文件（可选择获取所有子目录和文件）
	:param path: 文件夹路径
	:param include_subdir: 是否获取所有子目录和文件，默认为 False
	:return: 文件列表
	"""
	
	return [os.path.join(root, filename) for root, dirs, filenames in os.walk(path) for filename in filenames if (include_subdir or root == path)]


@calculate_time
def get_files_list(path, include_subdir=False):
	"""
	获取当前路径下文件（可选择获取所有子目录和文件）
	:param path: 文件夹路径
	:param include_subdir: 是否包含子文件夹，default False
	:return: 文件路径列表
	"""
	if not include_subdir:
		# files = [os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]  # os.listdir()
		return [f.path for f in os.scandir(path) if f.is_file()]  # os.scandir()
	else:
		# return glob.glob(os.path.join(path, "**", "*"), recursive=include_subdir)  # glob.glob()
		return [os.path.join(root, name) for root, dirs, files in os.walk(path) for name in files]


@calculate_time
def group_files_by_type(path, include_subdir: bool = False):
	"""
	根据文件类型分组，返回一个以文件类型为键，文件路径列表为值的字典
	:param path: 路径
	:param include_subdir: 是否包含子文件夹，Default，false
	:return: 一个以文件类型为键，文件路径为值的字典
	"""
	all_files = get_files_list(path, include_subdir)
	
	# 将所有文件按类型分组
	files_by_type = defaultdict(list)  # defaultdict,自动创建默认值，当访问一个不存在的键时，它会自动创建一个默认值作为这个键的值。这里的默认值传入的是一个空的列表
	for file in all_files:
		file_type = os.path.splitext(file)[-1].lower()  # 获取文件扩展名
		files_by_type[file_type].append(file)
	return files_by_type


@calculate_time
def get_template(height: int = None, width: int = 1200, bar_ratio: float = 5.0):
	if bar_ratio is None and (height is None or width is None):
		# ratio 为空且 height 或 width 中至少有一个为空时执行操作1
		pass
	elif bar_ratio is None:
		# ratio 为空且 height、width 均不为空时执行操作2
		h, w = height, width
	elif height is None or width is None:
		# ratio 不为空且 height 或 width 中至少有一个为空时执行操作3
		if height is None:
			# height 为空，width 不为空时执行操作4
			h, w = int(width / bar_ratio), width
		else:
			# height 不为空，width 为空时执行操作5
			h, w = height, height * bar_ratio
	else:
		# ratio 不为空且 height、width 均不为空时执行操作6
		h, w = height, width
	
	# 创建一个wxh的灰度图像
	img = np.ones((h, w), dtype=np.uint8) * 255
	# 定义矩形区域的左上角和右下角坐标
	pt1 = (int(0.08 * w), int(0.1 * h))
	pt2 = (int(0.86 * w), int(0.92 * h))
	# 在灰度图像上绘制矩形区域
	cv2.rectangle(img, pt1, pt2, 127, -1)
	# 显示生成的灰度图像
	cv2.imshow("gray image", img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
# get_template()
