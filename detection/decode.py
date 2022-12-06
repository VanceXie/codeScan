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
    detector = cv2.wechat_qrcode_WeChatQRCode(r"D:\Project\codeScan\detection\wechat_qrcode_model\detect.prototxt",
                                              r"D:\Project\codeScan\detection\wechat_qrcode_model\detect.caffemodel",
                                              r"D:\Project\codeScan\detection\wechat_qrcode_model\sr.prototxt",
                                              r"D:\Project\codeScan\detection\wechat_qrcode_model\sr.caffemodel")
    res, points = detector.detectAndDecode(img)

    print(res, points)
    if points is not None:
        image_detected = cv2.drawContours(img, [np.int32(points)], -1, (0, 0, 255), 2)
    return image_detected


def zXing_java(image_captured):
    success, encoded_image = cv2.imencode(".jpg", image_captured)  # 将获取的图片编码，以便后续转换为字节流
    image_bytes = encoded_image.tobytes()  # 将编码后的图片编码为字节流，不能直接将ndarray对象直接编码为字节，Java接口不能识别
    zx = ZXQRcode()  # 创建解码对象

    if zx.analysis_QR(image_bytes):
        code_result, matrix_map, corner_points = zx.analysis_QR(image_bytes)
        print(code_result)
        print(str(matrix_map))
        print(corner_points)
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
