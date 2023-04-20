# -*- coding: UTF-8 -*-
import os
from collections import defaultdict

from tools.PerformanceEval import calculate_time


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
