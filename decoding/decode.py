import cv2
import numpy as np
from pyzxing import *

from .zxingDecoder import ZXQRcode


class Decode:
    def __init__(self):
        self.zx = ZXQRcode()
    
    def decode_pyzxing(self, img, url):
        reader = BarCodeReader()
        results = reader.decode(url)
        # 打印解码结果
        for index, result in enumerate(results):
            if len(result) > 1:
                print(result)
                img = cv2.putText(img, str(result['parsed']), (40, 100 + 100 * index), cv2.FONT_HERSHEY_PLAIN, 3.0,
                                  (0, 255, 0), 3)
                # 绘制qr的检测结果
                image_detected = cv2.drawContours(img, [np.int32(result['points'])], -1, (0, 0, 255), 2)
            else:
                print(result)
                image_detected = cv2.putText(img, 'code not found', (40, 100 + 100 * index), cv2.FONT_HERSHEY_PLAIN,
                                             3.0,
                                             (0, 0, 255), 3)
        return image_detected
    
    def decode_wechat(self, img):
        detector = cv2.wechat_qrcode_WeChatQRCode(r"D:\Project\codeScan\detection\wechat_qrcode_model\detect.prototxt",
                                                  r"D:\Project\codeScan\detection\wechat_qrcode_model\detect.caffemodel",
                                                  r"D:\Project\codeScan\detection\wechat_qrcode_model\sr.prototxt",
                                                  r"D:\Project\codeScan\detection\wechat_qrcode_model\sr.caffemodel")
        res, points = detector.detectAndDecode(img)
        print(res, points)
        if points is not None:
            image_detected_contours = cv2.drawContours(img, [np.int32(points)], -1, (0, 0, 255), 2)
            image_detected_all = cv2.putText(image_detected_contours, str(res), (5, 50), cv2.FONT_HERSHEY_PLAIN,
                                             3.0, (0, 255, 0), 3)
        else:
            image_detected_all = cv2.putText(img, 'not Found!', (5, 50), cv2.FONT_HERSHEY_PLAIN, 3.0,
                                             (0, 0, 255), 3)
        return image_detected_all
    
    def decode_zxing(self, image_captured):
        if self.zx.analysis_QR(image_captured):
            code_result, matrix_map, corner_points = self.zx.analysis_QR(image_captured)
            # print("解码信息" + code_result)
            # print(str(matrix_map))
            # print(corner_points)
            corner_points = np.asarray(corner_points)
            image_detected_contours = cv2.drawContours(image_captured, [np.int32(corner_points)], -1, (0, 0, 255), 2)
            image_detected_all = cv2.putText(image_detected_contours, str(code_result), (5, 50), cv2.FONT_HERSHEY_PLAIN,
                                             3.0, (0, 255, 0), 3)
        else:
            image_detected_all = cv2.putText(image_captured, 'not Found!', (5, 50), cv2.FONT_HERSHEY_PLAIN, 3.0,
                                             (0, 0, 255), 3)
        return image_detected_all
        # zx.dels()
    
    # cap = cv2.VideoCapture(0)  # 调整参数实现读取视频或调用摄像头
    # cap.set(3, 1280)
    # cap.set(4, 720)
    # while True:
    #     # 采集图片
    #     ret, frame = cap.read()
    #     img_captured = cv2.flip(frame, 1)
    #
    #     img_detected = zXing_java(img_captured)
    #     if ret:
    #         cv2.imshow("result", img_detected)
    #
    #     else:
    #         cv2.imshow("result", img_captured)
    #     if cv2.waitKey(1) & 0xff == ord('q'):
    #         break
    # cap.release()
    # cv2.destroyAllWindows()
