# -*-coding:utf-8 -*-

import cv2 as cv
import numpy as np

# 加载图片
src_image = cv.imread(r"/home/ubei/Pictures/01.png")
# 实例化
qrcoder = cv.QRCodeDetector()
# qr检测并解码
codeinfo, points, straight_qrcode = qrcoder.detectAndDecode(src_image)
# 绘制qr的检测结果
cv.drawContours(src_image, [np.int32(points)], 0, (0, 0, 255), 2)
print(points)
# 打印解码结果
print("qrcode :", codeinfo)
cv.imshow("result", src_image)
cv.waitKey(0)