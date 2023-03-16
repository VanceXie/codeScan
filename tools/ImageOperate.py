# -*- coding: UTF-8 -*-
import cv2
import numpy as np


def img_equalize(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    # 将LAB色彩空间的L通道分离出来
    l, a, b = cv2.split(lab)
    # 创建CLAHE对象
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(12, 12))
    # 对L通道进行CLAHE均衡化
    l_clahe = clahe.apply(l)
    # 将CLAHE均衡化后的L通道合并回LAB色彩空间
    lab_clahe = cv2.merge((l_clahe, a, b))
    # 将LAB色彩空间转换回BGR色彩空间
    bgr_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
    return bgr_clahe


def sharpen(img):
    kernel = np.array([[-1, -1, -1],
                       [-1, 9, -1],
                       [-1, -1, -1]])
    
    # 应用锐化算子
    sharpened = cv2.filter2D(img, -1, kernel)
    return sharpened


image = cv2.imread(r"D:\Project\codeScan\location\pic\template.png", 1)
sharpened_img = sharpen(image)
result = cv2.medianBlur(sharpened_img, 3)
cv2.imshow('Original Image', image)
cv2.imshow('Sharpened Image', sharpened_img)
cv2.imshow('Filtered Image', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
