# -- coding: utf-8 --

import cv2
import numpy as np


detector = cv2.wechat_qrcode_WeChatQRCode("./wechat_qrcode_model/detect.prototxt",
                                          "./wechat_qrcode_model/detect.caffemodel",
                                          "./wechat_qrcode_model/sr.prototxt",
                                          "./wechat_qrcode_model/sr.caffemodel")
img = cv2.imread(r'D:\Test_Label1\Defect_001.png')
res, points = detector.detectAndDecode(img)

print(res, points)
cv2.drawContours(img, [np.int32(points)], -1, (0, 0, 255), 2)
cv2.namedWindow('result', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
cv2.imshow('result', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
