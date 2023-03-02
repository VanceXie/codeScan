# -*- coding: UTF-8 -*-
import math
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from tools import *


def compute_overlap(x1, y1, w1, h1, x2, y2, w2, h2):
    # 计算矩形框的面积
    area1 = w1 * h1
    area2 = w2 * h2
    # 计算相交部分的坐标
    x_inter = max(x1, x2)
    y_inter = max(y1, y2)
    w_inter = min(x1 + w1, x2 + w2) - x_inter
    h_inter = min(y1 + h1, y2 + h2) - y_inter
    # 计算相交部分的面积
    if w_inter <= 0 or h_inter <= 0:
        return 0.0
    area_inter = w_inter * h_inter
    # 计算并集的面积
    area_union = area1 + area2 - area_inter
    # 计算重叠度
    overlap = area_inter / area_union
    return overlap


def rotate_and_scale(img, angle, scale):
    '''
    旋转和比例不变变换
    img: 输入图像
    angle: 旋转角度
    scale: 缩放比例
    '''
    img_resized = cv2.resize(img, (int(scale * img.shape[1]), int(scale * img.shape[0])))
    rows, cols = img_resized.shape[:2]
    # 构造旋转变换矩阵
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    cols_new = int(cols * math.cos(angle) + rows * math.sin(angle))
    rows_new = int(cols * math.sin(angle) + rows * math.cos(angle))
    # 进行旋转变换
    img_rotated = cv2.warpAffine(img_resized, M, (cols_new, rows_new))
    return img_rotated


@PerformanceEval.calculate_time
def template_match_multi(img, template, angle_step=30, scale_step=0.2, similarity_threshold=1, overlap_threshold=0.3):
    '''
    旋转和比例不变模板匹配多目标检测
    img: 输入图像
    template: 模板图像
    angle_step: 旋转角度步长
    scale_step: 缩放比例步长
    threshold: 相似度阈值
    '''
    # 获取模板图像的大小
    th, tw = template.shape[:2]
    # 初始化最大匹配度和对应的位置
    matches = None
    # 保存保留的检测框索引
    # 在一定范围内进行旋转和比例不变变换，并计算相似度
    for angle in range(0, 360, angle_step):
        for scale in np.arange(0.5, 1.2, scale_step):
            # 进行旋转和比例不变变换
            template_rotated = rotate_and_scale(template, angle, scale)
            # 计算相似度
            result = cv2.matchTemplate(img, template_rotated, cv2.TM_CCORR)
            # 找到所有符合阈值的匹配位置
            locs = np.where(result >= similarity_threshold * result.max())
            
            for pt in zip(*locs):
                # 记录匹配的位置、相似度、角度和尺度
                # matches.append([pt, result[pt[0], pt[1]], angle, scale])
                if matches is None:
                    matches = np.asarray([pt, result[pt[0], pt[1]], angle, scale])
                else:
                    matches = np.vstack((matches, np.asarray([pt, result[pt[0], pt[1]], angle, scale])))
    # 取出得分最高的检测框
    matches = matches[np.argsort(matches[:, 1])]
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    cv2.rectangle(img, (max_loc[0], max_loc[1]), (max_loc[0] + tw, max_loc[1] + th), (0, 0, 255), 2)
    for pt in zip(*locs[::-1]):
        # 计算当前检测框与得分最高的检测框的重叠度
        overlap = compute_overlap(max_loc[0], max_loc[1], tw, th, pt[1], pt[0], tw, th)
        if overlap <= overlap_threshold:
            cv2.rectangle(img, (pt[1], pt[0]), (pt[1] + tw, pt[0] + th), (0, 0, 255), 2)
    return img


# 读取目标图像和模板图像
target_img = cv2.imread(r"pic/Over_008.png")
template_img = cv2.imread("pic/template.png")
target_img = ImageOperate.img_equalize(target_img)
img = template_match_multi(target_img, template_img, 180, 0.1, 0.99, 0.1)
cv2.namedWindow("Barcode Detection", cv2.WINDOW_NORMAL)
cv2.imshow("Barcode Detection", img)
if cv2.waitKey(0) == 27:
    cv2.destroyAllWindows()


def detect_template(image, template, threshold=0.75, draw_result=True):  # '''基于特征提取的多目标旋转和尺度不变匹配'''
    
    # Convert images to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    
    # Initialize SIFT detector and FLANN matcher
    sift = cv2.xfeatures2d.SIFT_create()
    matcher = cv2.FlannBasedMatcher_create()
    
    # Extract keypoints and descriptors from template and image
    keypoints_template, descriptors_template = sift.detectAndCompute(gray_template, None)
    keypoints_image, descriptors_image = sift.detectAndCompute(gray_image, None)
    
    # Match descriptors using FLANN matcher
    matches = matcher.knnMatch(descriptors_template, descriptors_image, k=2)
    
    # Filter matches by Lowe's ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)
    
    # Compute homography matrix using RANSAC algorithm
    src_pts = np.float32([keypoints_template[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints_image[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    
    # Apply perspective transform to template
    h, w = template.shape[:2]
    template_warped = cv2.warpPerspective(template, M, (w, h))
    
    # Match warped template to image using normalized cross-correlation
    result = cv2.matchTemplate(gray_image, cv2.cvtColor(template_warped, cv2.COLOR_BGR2GRAY), cv2.TM_CCOEFF_NORMED)
    
    # Find locations of matched regions above threshold
    locations = np.where(result >= threshold)
    locations = list(zip(*locations[::-1]))
    
    # Draw bounding boxes around matched regions
    if draw_result:
        for pt in locations:
            cv2.rectangle(image, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)
    
    return locations


@PerformanceEval.calculate_time
def template_match(template_image):
    h, w = template_image.shape[:2]
    methods = ['cv2.TM_SQDIFF_NORMED']
    results = []
    for meth in methods:
        # img = target_img.copy()
        
        method = eval(meth)
        for root, dirs, files in os.walk(r"D:\Project\test"):
            for file in files:
                img = cv2.imread(os.path.join(root, file))
                
                img_equalized = ImageOperate.img_equalize(img)
                img_blur = cv2.medianBlur(img_equalized, 3)
                res = cv2.matchTemplate(img_blur, template_image, method)
                #
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
                #
                if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
                    #     threshold = 0.15
                    #     loc = np.where(res <= 0.15)
                    top_left = min_loc
                else:
                    #     threshold = 0.85
                    #     loc = np.where(res >= 0.85)
                    top_left = max_loc
                bottom_right = (top_left[0] + w, top_left[1] + h)
                cv2.rectangle(img, top_left, bottom_right, (0, 255, 0), 2)
                results.append(img)
    return results
    # # for i in zip(*loc[::-1]):
    # #     cv2.rectangle(img, i, (i[0] + w, i[1] + h), (0, 255, 0), 2)
    # plt.subplot(121), plt.imshow(res, cmap='gray')
    # plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
    #
    # plt.subplot(122), plt.imshow(img, cmap='gray')
    # plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
    # plt.suptitle(meth)
    # plt.savefig('D:\\Project\\result\\2\\' + file.title() + meth + '.jpg',
    #             format='jpg', dpi=500, bbox_inches='tight')
    # # cv2.namedWindow("Barcode Detection", cv2.WINDOW_NORMAL)
    # # cv2.imshow("Barcode Detection", img_blur)
    # # if cv2.waitKey(0) == 27:
    # #     cv2.destroyAllWindows()
    # # plt.show()
