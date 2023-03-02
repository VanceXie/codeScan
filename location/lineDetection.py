# -*- coding: UTF-8 -*-
from math import sqrt

import cv2
import numpy as np

# Load the image
image = cv2.imread("pic/4.tif")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Perform edge detection
edges = cv2.Canny(gray, 50, 150)

# Find lines in the image using HoughLinesP
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)

left, top, right, bottom = lines[0][0]

# 找到和一维条码类似的直线
barcode_lines = []
# Draw lines on the image
for line in lines:
    x1, y1, x2, y2 = line[0]
    # 计算直线的夹角
    cos = (x2 - x1) / sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1))
    # 确定直线是否与一维条码相似
    if abs(cos) > 0.8:
        barcode_lines.append(line)
        cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
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

# Draw a rectangle around the barcode
cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 2)

cv2.namedWindow("Barcode detection", cv2.WINDOW_NORMAL)
# Display the image
cv2.imshow("Barcode detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
