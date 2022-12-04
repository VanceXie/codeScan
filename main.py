import sys
import threading
import time

from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5 import uic
from ui.scan_display_main import Ui_MainWindow
import cv2
import numpy as np
from PIL import Image


# 动态载入
class mainWindow(QMainWindow):
    def __init__(self, img):
        super().__init__()
        # PyQt5
        # self.ui = uic.loadUi('./ui/scan_display_main.ui')
        self.ui = Ui_MainWindow(img)
        self.ui.show()


result = None


def capture():
    global result
    cap = cv2.VideoCapture(0)  # 调整参数实现读取视频或调用摄像头
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
        if points is not None:
            cv2.drawContours(frame, [np.int32(points)], 0, (0, 0, 255), 2)
            print(points)
            # 打印解码结果
            print("qrcode :", codeinfo)
        else:
            print("QR code not detected")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
        result = frame.toqpixmap()
        if result is not None:
            break
    # cv2.imshow("result", frame)
    # if cv2.waitKey(50) & 0xff == ord('q'):
    #     break


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = mainWindow(QPixmap(r"ui/icons/01.jpg"))
    sys.exit(app.exec_())
