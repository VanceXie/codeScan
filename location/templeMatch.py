# -*- coding: UTF-8 -*-
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from tools import *


def rotate_and_scale(img, angle, scale):
    '''
    旋转和比例不变变换
    img: 输入图像
    angle: 旋转角度
    scale: 缩放比例
    '''
    rows, cols = img.shape[:2]
    # 构造旋转变换矩阵
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, scale)
    # 进行旋转变换
    img_rotated = cv2.warpAffine(img, M, (cols, rows))
    return img_rotated


def template_match_rot_scale(img, template, angle_step=1, scale_step=0.1):
    '''
    旋转和比例不变模板匹配
    img: 输入图像
    template: 模板图像
    angle_step: 旋转角度步长
    scale_step: 缩放比例步长
    '''
    # 获取模板图像的大小
    th, tw = template.shape[:2]
    # 初始化最大匹配度和对应的位置
    max_similarity = 0
    max_location = None
    # 在一定范围内进行旋转和比例不变变换，并计算相似度
    for angle in range(0, 180, angle_step):
        for scale in np.arange(0.5, 2.0, scale_step):
            # 进行旋转和比例不变变换
            template_rotated = rotate_and_scale(img, angle, scale)
            # 计算相似度
            result = cv2.matchTemplate(img, template_rotated, cv2.TM_CCORR)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            # 更新最大匹配度和对应的位置
            if max_val > max_similarity:
                max_similarity = max_val
                max_location = max_loc
                max_angle = angle
                max_scale = scale
    return max_location, max_similarity, max_angle, max_scale


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


# 读取目标图像和模板图像
target_img = cv2.imread(r"./Defect_030.png")
template_img = cv2.imread("./template.png")
# 获取模板图像的尺寸
h, w = template_img.shape[:2]


@PerformanceEval.calculate_time
def template_match():
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
                res = cv2.matchTemplate(img_blur, template_img, method)
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
