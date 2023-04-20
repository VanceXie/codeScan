from collections import defaultdict

import cv2
import numpy as np

import tools
from tools.PerformanceEval import calculate_time


@calculate_time
def find_barcode_2(img):
	"""
	:param img: 'image as np.array'
	:return: np.array(dtype=np.uint8)
	"""
	# Perform edge detection
	edges = cv2.Canny(img, 35, 155)
	
	# Find lines in the image using HoughLines
	lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength=100, maxLineGap=2)
	# Group lines by slope
	groups = defaultdict(list)
	for line in lines:
		x1, y1, x2, y2 = line[0]
		cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
		slope = (y2 - y1) / (x2 - x1) if x2 != x1 else float('inf')
		groups[slope].append(line)
	
	# Find the group with the most lines
	linemost = max(groups.values(), key=len)
	
	# Group lines by proximity
	groups = {}
	for line in linemost:
		x1, y1, x2, y2 = line[0]
		for key, value in groups.items():
			if abs((y2 - y1) / (x2 - x1) - key) < 0.1:
				value.append(line)
				break
		else:
			groups[(y2 - y1) / (x2 - x1)] = [line]
	
	# Find bounding box of lines in group
	left, top, right, bottom = np.inf, np.inf, 0, 0
	for line in groups.values():
		for l in line:
			x1, y1, x2, y2 = l[0]
			if x1 < left:
				left = x1
			if x2 < left:
				left = x2
			if x1 > right:
				right = x1
			if x2 > right:
				right = x2
			if y1 < top:
				top = y1
			if y2 < top:
				top = y2
			if y1 > bottom:
				bottom = y1
			if y2 > bottom:
				bottom = y2
		cv2.rectangle(img, (left, top), (right, bottom), (0, 0, 255), 2)
	
	return img


img = cv2.imread(r'D:\fy.xie\fenx\fenx - General\Ubei\Test_Label1\2.tif')
img = tools.ImageOperate.img_equalize(img)
detected_edges = find_barcode_2(img)

cv2.namedWindow('canny demo', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)

cv2.imshow('canny demo', detected_edges)
if cv2.waitKey(0) == 27:
	cv2.destroyAllWindows()
