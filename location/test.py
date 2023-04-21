from collections import defaultdict

import cv2
import numpy as np
from sklearn.cluster import DBSCAN

import tools
from tools.PerformanceEval import calculate_time


# 计算线段端点之间的距离矩阵
def compute_distance_matrix(lines):
	n = len(lines)
	dist = np.zeros((n, n))
	for i in range(n):
		for j in range(i + 1, n):
			p1 = lines[i][0]
			p2 = lines[j][0]
			d = np.linalg.norm(p1 - p2)
			dist[i][j] = d
			dist[j][i] = d
	return dist


# 对线段进行无监督聚类
def cluster_lines(lines, eps):
	# 计算距离矩阵
	dist = compute_distance_matrix(lines)
	# 使用DBSCAN进行聚类
	db = DBSCAN(eps=eps, min_samples=10, metric='precomputed')
	labels = db.fit_predict(dist)
	# 将不同簇的线段分组
	clusters = defaultdict(list)
	for i, label in enumerate(labels):
		clusters[label].append(lines[i])
	max_length = max(len(v) for v in clusters.values())
	clusters_new = {key: value for key, value in clusters.items() if len(value) >= 0.8 * max_length}
	return clusters_new


# 绘制分组区域
def draw_clusters(img, clusters):
	for label, lines in clusters.items():
		color = np.random.randint(0, 255, (3,))
		left = min([min(line[0][0], line[0][2]) for line in lines])
		top = min([min(line[0][1], line[0][3]) for line in lines])
		right = max([max(line[0][0], line[0][2]) for line in lines])
		bottom = max([max(line[0][1], line[0][3]) for line in lines])
		
		for line in lines:
			p1, p2 = line[0][:2], line[0][2:]
			cv2.line(img, (p1[0], p1[1]), (p2[0], p2[1]), color.tolist(), 2)
		
		cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)
	# left, top, right, bottom = np.inf, np.inf, 0, 0
	# for line in lines:
	# 	p1, p2 = line[0][:2], line[0][2:]
	# 	cv2.line(img, (p1[0], p1[1]), (p2[0], p2[1]), color.tolist(), 2)
	# 	x1, y1, x2, y2 = line[0]
	# 	if x1 < left:
	# 		left = x1
	# 	if x2 < left:
	# 		left = x2
	# 	if x1 > right:
	# 		right = x1
	# 	if x2 > right:
	# 		right = x2
	# 	if y1 < top:
	# 		top = y1
	# 	if y2 < top:
	# 		top = y2
	# 	if y1 > bottom:
	# 		bottom = y1
	# 	if y2 > bottom:
	# 		bottom = y2
	# 	cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)
	return img


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


@calculate_time
def find_barcode_by_cluster(img):
	"""
	:param img: 'image as np.array'
	:return: np.array(dtype=np.uint8)
	"""
	# Perform edge detection
	edges = cv2.Canny(img, 35, 155)
	
	# Find lines in the image using HoughLines
	lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 5, minLineLength=5, maxLineGap=2)
	# Group lines by slope
	groups = defaultdict(list)
	for line in lines:
		x1, y1, x2, y2 = line[0]
		cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 1)
		slope = (y2 - y1) / (x2 - x1) if x2 != x1 else float('inf')
		groups[slope].append(line)
	
	# Find the group with the most lines
	linemost = max(groups.values(), key=len)
	clusters = cluster_lines(linemost, eps=150)
	return clusters


image = cv2.imread(r'D:\fy.xie\fenx\fenx - General\Ubei\Test_Label1\Defect_034.png')
image = tools.ImageOperate.img_equalize(image)
img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
block_image = block_threshold(img_gray)
clusters = find_barcode_by_cluster(block_image)
image = draw_clusters(image, clusters)
cv2.namedWindow('canny demo', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
cv2.imshow('canny demo', image)
if cv2.waitKey(0) == 27:
	cv2.destroyAllWindows()
