import cv2
import numpy as np
from PIL import Image

result = None


def capture():
    global result
    cap = cv2.VideoCapture(0)  # 调整参数实现读取视频或调用摄像头
    cap.set(3, 1280)
    cap.set(4, 720)
    fps = cap.get(cv2.CAP_PROP_FPS)  # 计算视频的帧率
    print("fps:", fps)
    while True:
        # 加载图片
        ret, frame = cap.read()
        # 实例化
        qrcoder = cv2.QRCodeDetector()
        # qr检测并解码
        codeinfo, points, straight_qrcode = qrcoder.detectAndDecode(frame)
        # 绘制qr的检测结果
        if codeinfo and points is not None:
            cv2.drawContours(frame, [np.int32(points)], 0, (0, 0, 255), 2)
            print(points)
            # 打印解码结果
            print("qrcode :", codeinfo)
        else:
            print("QR code not detected")
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cv2.imshow("result", frame)
        if cv2.waitKey(50) & 0xff == ord('q'):
            break


capture()
