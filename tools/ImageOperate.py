# -*- coding: UTF-8 -*-
import cv2


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
