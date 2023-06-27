import cv2
import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import DBSCAN
from collections import defaultdict
from tools.DecoratorTools import calculate_time
from tools.ImageOperate import pyrdown_multithread


# 计算线段端点之间的距离矩阵
def compute_distance_matrix(lines: list) -> np.ndarray:
	"""
	Calculate the distance between two lines to form a distance matrix
	:param lines: list of lines
	:return: distance matrix, a symmetric matrix whose diagonal elements are 0
	"""
	lines = np.array(lines).reshape((-1, 4))  # 把含有m个元素（每个元素是(1×4)的ndarray）的线段list转化为m×4的ndarray，每一行代表每个线段的端点坐标(x0,y0,x1,y1)

	dist_triple = pdist(lines)  # 计算n维向量数组的成对距离的函数。输出结果会压缩成一个一维距离向量，只包含输入矩阵的下三角部分（省略了对角线上的元素）
	dist = squareform(dist_triple)  # 将pdist函数产生的压缩距离向量转换为距离矩阵。其输入是一个N*(N-1)/2维的压缩距离向量，输出是一个N*N维的距离矩阵
	return dist


# 对线段进行无监督聚类
def cluster_lines(lines: list, eps: int) -> dict:
	"""
	cluster lines by distance
	:param lines: list of lines
	:param eps: eps of cluster
	:return: Dictionary with label as key and list of lines as value
	"""
	# 计算距离矩阵
	dist = compute_distance_matrix(lines)
	# 使用DBSCAN进行聚类
	db = DBSCAN(eps=eps, min_samples=10, metric='precomputed')
	labels = db.fit_predict(dist)
	
	# 将不同簇的线段分组
	clusters = defaultdict(list)
	for i, label in enumerate(labels):
		clusters[label].append(lines[i:i + 1, :])
	for label, lines in clusters.items():
		clusters[label] = np.concatenate(lines, axis=0)
	
	max_length = max(len(v) for v in clusters.values())
	
	# 根据每一行首个元素进行排序
	# [lines.sort(axis=0) for lines in clusters.values()]
	
	# dist_dic = [np.diag(compute_distance_matrix(cluster[np.argsort(cluster[:, 0])]), k=1) for cluster in clusters.values()]
	# std_list = [np.std(dist_list) for dist_list in dist_dic]
	
	clusters_new = {key: value for key, value in clusters.items() if len(value) >= 0.3 * max_length}
	return clusters_new


# 绘制分组区域
def draw_clusters(img: np.ndarray, clusters: dict) -> np.ndarray:
	"""
	:param img: image to draw
	:param clusters: Dictionary with slope as key and list of lines as value
	:return: image
	"""
	for label, lines in clusters.items():
		color = np.random.randint(0, 255, (3,))
		for line in lines:
			p1, p2 = line[:2].astype(int), line[2:].astype(int)
			cv2.line(img, (p1[0], p1[1]), (p2[0], p2[1]), color.tolist(), 1)
		
		contours = np.asarray(lines).reshape(-1, 2)
		
		cv2.drawContours(img, [np.int0(cv2.boxPoints(cv2.minAreaRect(contours)))], 0, (0, 255, 0), 2)
	return img


def line_detect(lsd: cv2.LineSegmentDetector, edges: np.ndarray):
	"""
	detect lines and group the lines by slope
	:param lsd: LineSegmentDetector
	:param edges: Binary image or grayscale image
	:return: Dictionary with slope as key and list of lines as value
	"""
	# detect lines
	lines, width, prec, nfa = lsd.detect(edges)
	
	# Group lines by slope
	groups = defaultdict(list)
	for line in lines:
		x1, y1, x2, y2 = line[0].astype(int)
		slope = (y2 - y1) / (x2 - x1) if x2 != x1 else float('inf')
		
		# groups[slope] = np.vstack((groups[slope], line))
		'''
		使用 np.vstack 函数将每个线段添加到对应斜率的数组中，这会导致每次迭代都要进行内存分配和数据复制操作，造成性能损失。
		每次迭代中的动态分配空间会产生额外的开销，并且随着数组大小的增长，这种开销会逐渐累积。
		而且迭代前需要reshape操作更改lines数组的形状。
		'''
		groups[slope].append(line)
	
	for slope, lines_list in groups.items():
		# numpy.concatenate 函数一次性将多个数组合并为一个数组，不会产生额外的动态分配空间的开销
		groups[slope] = np.concatenate(lines_list, axis=0)
	return groups


@calculate_time
def find_barcode_by_cluster(img: np.ndarray, eps: int) -> dict:
	"""
	:param eps: max distance between two lines which could be divided into one group
	:param img: image as np.ndarry
	:return: np.ndarry(dtype=np.uint8)
	"""
	# Perform edge detection
	edges = cv2.Canny(img, 200, 255)
	
	# instant LineSegmentDetector
	lsd = cv2.createLineSegmentDetector()
	
	# Group lines by slope
	groups = line_detect(lsd, edges)
	
	# Find the group with the most lines
	group_with_most_lines = max(groups.values(), key=len)
	clusters = cluster_lines(group_with_most_lines, eps=eps)
	return clusters

# # Load the image
# path = r'D:\Fenkx\Fenkx - General\Ubei\Test_Label1'
# for index, item in enumerate(os.listdir(path)):
# 	file = os.path.join(path, item)
# 	if os.path.isfile(file):
# 		image_source = cv2.imdecode(np.fromfile(file, dtype=np.uint8), 1)
# 		# removed_img = hist_cut(image_source)
# 		# remapped_img = hist_remap(removed_img)
# 		image_equalized = clahe_equalize(image_source)
# 		try:
# 			clusters = find_barcode_by_cluster(image_equalized)
# 			image_drawed = draw_clusters(image_source, clusters)
# 		finally:
# 			filename = os.path.splitext(item)
# 			new_name = filename[0] + filename[-1]
# 			result_path = os.path.join(path, 'result_LineCluster')
# 			if not os.path.exists(result_path):
# 				os.makedirs(result_path)
# 			cv2.imwrite(os.path.join(result_path, new_name), image_drawed)
# print('finished!')
