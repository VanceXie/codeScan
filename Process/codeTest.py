# -*-coding:utf-8 -*-

import cv2 as cv
import numpy as np
from pyzxing import BarCodeReader

filename = r'D:\Test_Label1\Defect_017.png'
filenameList = filename.split('.')
result_name = filenameList[0] + '_detected' + '.' + filenameList[1]

# 加载图片
src_image = cv.imread(filename)
# '''WeChat'''
# cv.wechat_qrcode_WeChatQRCode()
'''opencv读码'''
# 实例化
barcodeDetector = cv.barcode_BarcodeDetector()
# qr检测并解码
retval, decoded_info, decoded_type, points = barcodeDetector.detectAndDecode(src_image)

'''pyzxing解码'''
reader = BarCodeReader()
results = reader.decode(filename)
# 打印解码结果
for index, result in enumerate(results):
    if len(result) > 1:
        print(result)
        cv.putText(src_image, str(result['parsed']), (40, 100 + 100 * index), cv.FONT_HERSHEY_PLAIN, 5.0, (0, 255, 0),
                   3)
        # 绘制qr的检测结果
        cv.drawContours(src_image, [np.int32(result['points'])], -1, (0, 0, 255), 2)
    else:
        print(result)
        cv.putText(src_image, 'code not found', (40, 100 + 100 * index), cv.FONT_HERSHEY_PLAIN, 5.0, (0, 0, 255), 3)

cv.namedWindow('result', cv.WINDOW_NORMAL)
cv.imshow('result', src_image)
cv.waitKey(0)
cv.destroyAllWindows()
