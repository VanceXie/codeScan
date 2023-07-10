# -*- coding: UTF-8 -*-
import os

import cv2
import numpy as np
import matplotlib.pyplot as plt
from tools.ImageOperate import downsample_with_edge_preservation, ImageEqualize, Sharpen, filter_bright_spots
from tools.DecoratorTools import Timer


def locate(image_source):
	# 降采样
	image_pydown_s = downsample_with_edge_preservation(image_source)
	# 边缘增强
	image_pydown_s = Sharpen(image_pydown_s).mask_sharpen().image
	# 阈值分割
	image_gray = cv2.cvtColor(image_pydown_s, cv2.COLOR_BGR2GRAY)
	
	lsd = cv2.createLineSegmentDetector()
	_ = filter_bright_spots(image_gray, lsd)
	
	# 重映射
	image_gray = cv2.medianBlur(image_gray, 3)
	image_gray = ImageEqualize(image_gray).adaptive_index_equalize().image
	
	contour_list = filter_bright_spots(image_gray, lsd)
	
	for contour in contour_list:
		rect = cv2.minAreaRect(contour)
		box = cv2.boxPoints(rect).astype(np.int0)
		cv2.drawContours(image_pydown_s, [box], -1, (0, 0, 255), 1)
	return image_pydown_s


# Load the image
path = r'D:\Fenkx\Fenkx - General\AI\Dataset\BarCode\My Datasets\Factory'
for index, item in enumerate(os.listdir(path)):
	file = os.path.join(path, item)
	if os.path.isfile(file):
		image_source = cv2.imdecode(np.fromfile(file, dtype=np.uint8), 1)
		try:
			image_pydown_s = locate(image_source)
		finally:
			filename = os.path.splitext(item)
			new_name = filename[0] + filename[-1]
			result_path = os.path.join(path, 'result_LineCluster')
			if not os.path.exists(result_path):
				os.makedirs(result_path)
			cv2.imwrite(os.path.join(result_path, new_name), image_pydown_s)
print('finished!')
