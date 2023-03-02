# -*- coding: UTF-8 -*-
import os


def get_all_files(path):
    # 遍历目录，获取所有文件
    # @parameter flag 表示是否优先遍历 top 目录
    all_files = []
    for root, dirs, files in os.walk(path):
        for file in files:
            all_files.append(os.path.join(root, file))
    
    return all_files


def group_files_by_type(path):
    # 获取指定目录下的所有文件
    # @parameter flag 表示是否优先遍历 top 目录
    all_files = get_all_files(path)
    
    # 将所有文件按类型分组
    files_by_type = {}
    for file in all_files:
        file_type = os.path.splitext(file)[-1].lower()  # 获取文件扩展名
        if file_type not in files_by_type:
            files_by_type[file_type] = []
        files_by_type[file_type].append(file)
    
    return files_by_type

