import cv2
import numpy as np


def downsample_with_edge_preservation(image, scale_factor):
	# 定义卷积核
	# 创建高斯卷积核
	sigma = 0.4
	size = int(2 * np.ceil(3 * sigma) + 1)
	kernel_gaussian = cv2.getGaussianKernel(size, sigma)
	
	# 创建 LoG 卷积核
	kernel_log = cv2.sepFilter2D(kernel_gaussian, -1, kernel_gaussian.T, kernel_gaussian)
	kernel = np.array([[1, 4, 6, 4, 1],
					   [4, 16, 24, 16, 4],
					   [6, 24, 36, 24, 6],
					   [4, 16, 24, 16, 4],
					   [1, 4, 6, 4, 1]]) / 256.0
	
	# 使用卷积核进行下采样
	smoothed_image = cv2.filter2D(image, -1, kernel_log)
	downsampled_image = smoothed_image[::scale_factor, ::scale_factor]
	
	return downsampled_image


# 读取图像
image = cv2.imread(r"D:\Fenkx\Fenkx - General\AI\Dataset\BarCode\My Datasets\Factory\772FBFU0MPZZZS02H0_NG_BarCode_Camera3_1205171258.jpg")

# 设置下采样比例
scale_factor = 2

# 进行下采样并保留边缘信息
downsampled_image = downsample_with_edge_preservation(image, scale_factor)
print(downsampled_image.shape)
# 显示结果
cv2.namedWindow('Downsampled Image', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
cv2.imshow('Downsampled Image', downsampled_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
