import os
from collections import defaultdict

import cv2
import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import DBSCAN

from tools.ImageOperate import img_equalize
from tools.PerformanceEval import calculate_time


# 计算线段端点之间的距离矩阵
def compute_distance_matrix(lines):
	"""
	计算每条线之间的距离矩阵
	:param lines: 传入的线段list
	:return: 距离矩阵，对角线元素为0的对称矩阵
	"""
	lines = np.array(lines).reshape((-1, 4))  # 把含有m个元素（每个元素是(1×4)的ndarray）的线段list转化为m×4的ndarray，每一行代表每个线段的端点坐标(x0,y0,x1,y1)
	dist_triple = pdist(lines)  # 计算n维向量数组的成对距离的函数。输出结果会压缩成一个一维距离向量，只包含输入矩阵的下三角部分（省略了对角线上的元素）
	dist = squareform(dist_triple)  # 将pdist函数产生的压缩距离向量转换为距离矩阵。其输入是一个N*(N-1)/2维的压缩距离向量，输出是一个N*N维的距离矩阵
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
	clusters_new = {key: value for key, value in clusters.items() if len(value) >= 0.4 * max_length}
	return clusters_new


# 绘制分组区域
def draw_clusters(img, clusters):
	for label, lines in clusters.items():
		color = np.random.randint(0, 255, (3,))
		for line in lines:
			p1, p2 = line[0][:2], line[0][2:]
			cv2.line(img, (p1[0], p1[1]), (p2[0], p2[1]), color.tolist(), 2)
		
		contours = np.asarray(lines).reshape(-1, 2)
		cv2.drawContours(img, [np.int0(cv2.boxPoints(cv2.minAreaRect(contours)))], 0, (0, 0, 255), 2)
	return img


@calculate_time
def find_barcode_by_cluster(img):
	"""
	:param img: 'image as np.array'
	:return: np.array(dtype=np.uint8)
	"""
	# Perform edge detection
	edges = cv2.Canny(img, 200, 255)
	# Find lines in the image using HoughLines
	lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 5, minLineLength=10, maxLineGap=2)
	# Group lines by slope
	groups = defaultdict(list)
	for line in lines:
		x1, y1, x2, y2 = line[0]
		cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 1)
		slope = (y2 - y1) / (x2 - x1) if x2 != x1 else float('inf')
		groups[slope].append(line)
	
	# Find the group with the most lines
	linemost = max(groups.values(), key=len)
	clusters = cluster_lines(linemost, eps=100)
	return clusters


# Load the image
path = r'D:\fy.xie\fenx\fenx - General\Ubei\Test_Label1'
for index, item in enumerate(os.listdir(path)):
	file = os.path.join(path, item)
	if os.path.isfile(file):
		image = cv2.imdecode(np.fromfile(file, dtype=np.uint8), 1)
		image = img_equalize(image)
		clusters = find_barcode_by_cluster(image)
		image = draw_clusters(image, clusters)
		filename = os.path.splitext(item)
		new_name = filename[0] + f'_{index}' + filename[-1]
		result_path = os.path.join(path, 'result_LineCluster')
		if not os.path.exists(result_path):
			os.makedirs(result_path)
		cv2.imwrite(os.path.join(result_path, new_name), image)

# image = cv2.imread(r'D:\fy.xie\fenx\fenx - General\Ubei\Test_Label1\13.tif')
# image = img_equalize(image)
# clusters = find_barcode_by_cluster(image)
# image = draw_clusters(image, clusters)
# cv2.namedWindow('canny demo', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
# cv2.imshow('canny demo', image)
# if cv2.waitKey(0) == 27:
# 	cv2.destroyAllWindows()
