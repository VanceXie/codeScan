import cv2
import numpy as np
from pyzxing import *

from detection.codeDetectionWithZxing import ZXQRcode


def zXingCode(img, url):
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
            image_detected = cv2.putText(img, 'code not found', (40, 100 + 100 * index), cv2.FONT_HERSHEY_PLAIN, 3.0,
                                         (0, 0, 255), 3)
    return image_detected


def weChatCode(img):
    detector = cv2.wechat_qrcode_WeChatQRCode("./wechat_qrcode_model/detect.prototxt",
                                              "./wechat_qrcode_model/detect.caffemodel",
                                              "./wechat_qrcode_model/sr.prototxt",
                                              "./wechat_qrcode_model/sr.caffemodel")
    res, points = detector.detectAndDecode(img)
    
    print(res, points)
    image_detected = cv2.drawContours(img, [np.int32(points)], -1, (0, 0, 255), 2)
    return image_detected


def zXing_java(image_captureed, image_bytes):
    zx = ZXQRcode()
    
    if zx.analysis_QR(image_bytes):
        code_result, matrix_map, corner_points = zx.analysis_QR(image_bytes)
        print(code_result)
        print(str(matrix_map))
        print(corner_points)
        corner_points = np.asarray(corner_points)
        image_detected = cv2.drawContours(image_captureed, [np.int32(corner_points)], -1, (0, 0, 255), 2)
        return image_detected
    else:
        print('---未检出！---' * 3)
    
    # zx.dels()


cap = cv2.VideoCapture(0)  # 调整参数实现读取视频或调用摄像头

while True:
    # 采集图片
    ret, frame = cap.read()
    img_captured = cv2.flip(frame, 1)
    success, encoded_image = cv2.imencode(".jpg", img_captured)
    img_byte = encoded_image.tobytes()
    
    img_detected = zXing_java(img_captured, img_byte)
    if img_detected is not None:
        cv2.imshow("result", img_detected)
    else:
        cv2.imshow("result", img_captured)
    if cv2.waitKey(500) & 0xff == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
