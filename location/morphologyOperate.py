# -*- coding: UTF-8 -*-
import cv2
import numpy as np

# 读取图像
image = cv2.imread("barcode_with_noise.jpg")

# 灰度化
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 阈值处理
threshold = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)[1]

# 腐蚀操作
kernel = np.ones((5,5), np.uint8)
eroded = cv2.erode(threshold, kernel, iterations=2)

# 膨胀操作
dilated = cv2.dilate(eroded, kernel, iterations=2)

# 查找轮廓
contours, hierarchy = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# 绘制条码位置
for contour in contours:
    (x,y,w,h) = cv2.boundingRect(contour)
    aspect_ratio = float(w) / h
    if w >= 150 and h <= 50 and aspect_ratio >= 2:
        cv2.rectangle(image, (x,y), (x+w, y+h), (0,255,0), 2)
    
    # # 计算轮廓的周长
    # perimeter = cv2.arcLength(cnt, True)
    # # 计算轮廓的面积
    # area = cv2.contourArea(cnt)
    # # 判断轮廓是否是一维条码
    # if perimeter > 100 and perimeter < 200 and area > 2000 and area < 5000:
    #     # 绘制轮廓
    #     cv2.drawContours(img, [cnt], 0, (0, 255, 0), 3)
    
# 显示图像
cv2.imshow("Barcode Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
