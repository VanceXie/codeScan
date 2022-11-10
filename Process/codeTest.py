# -*-coding:utf-8 -*-

import cv2 as cv
import numpy as np
from pyzxing import BarCodeReader

filename = "D:\\Test_Label1\\12.tif"
filenameList = filename.split('.')
result_name = filenameList[0] + '_detected' + '.' + filenameList[1]
# 加载图片
src_image = cv.imread(filename)
# 实例化
qrcoder = cv.barcode_BarcodeDetector()
# qr检测并解码
retval, decoded_info, decoded_type, points = qrcoder.detectAndDecode(src_image)

reader = BarCodeReader()
results = reader.decode(filename)
# 打印解码结果
for result in results:
    if result['parsed'] is not None:
        print(result)
        cv.putText(src_image, str(result['parsed']), (40, 50), cv.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 255), 2)
        # 绘制qr的检测结果
        cv.drawContours(src_image, [np.int32(result['points'])], -1, (0, 0, 255), 2)

# cv.namedWindow('result',cv.WINDOW_AUTOSIZE)
cv.imwrite(result_name, src_image)
