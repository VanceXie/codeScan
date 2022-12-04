import cv2
import numpy as np
from pyzxing import *

from detection.codeDetectionWithZxing import ZXQRcode


class DeCode():
    def __int__(self):
        pass

    '''pyzxing解码'''

    def zXingCode(self, img, url):
        reader = BarCodeReader()
        results = reader.decode(url)
        # 打印解码结果
        for index, result in enumerate(results):
            if len(result) > 1:
                print(result)
                img = cv2.putText(img, str(result['parsed']), (40, 100 + 100 * index), cv2.FONT_HERSHEY_PLAIN, 3.0,
                                  (0, 255, 0), 3)
                # 绘制qr的检测结果
                img_detected = cv2.drawContours(img, [np.int32(result['points'])], -1, (0, 0, 255), 2)
            else:
                print(result)
                img_detected = cv2.putText(img, 'code not found', (40, 100 + 100 * index), cv2.FONT_HERSHEY_PLAIN, 3.0,
                                           (0, 0, 255),
                                           3)
        return img_detected

    def weChatCode(self, img):
        detector = cv2.wechat_qrcode_WeChatQRCode("./wechat_qrcode_model/detect.prototxt",
                                                  "./wechat_qrcode_model/detect.caffemodel",
                                                  "./wechat_qrcode_model/sr.prototxt",
                                                  "./wechat_qrcode_model/sr.caffemodel")
        res, points = detector.detectAndDecode(img)
        print(res, points)
        img_detected = cv2.drawContours(img, [np.int32(points)], -1, (0, 0, 255), 2)
        return img_detected

    def zXing_java(self, image):
        zx = ZXQRcode()
        if zx.analysis_QR(image):
            code_result, matrix_map, corner_points = zx.analysis_QR(image)
            print(code_result)
            print(str(matrix_map))
            print(corner_points)
        else:
            print('未检出！\n' * 3)
        # zx.dels()


cap = cv2.VideoCapture(0)  # 调整参数实现读取视频或调用摄像头
deCode = DeCode()
while True:
    # 采集图片
    ret, frame = cap.read()
    src_image = cv2.flip(frame, 1)
    # 保存图片
    path = r"D:\Project\codeScan\img\01.jpg"
    cv2.imwrite(path, src_image)

    deCode.zXing_java(src_image)
    cv2.imshow("result", src_image)
    if cv2.waitKey(100) & 0xff == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
