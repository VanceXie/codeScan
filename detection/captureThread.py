# -*- coding: UTF-8 -*-
import cv2
from PyQt5.QtCore import QThread
from PyQt5.QtGui import QPixmap
from detection.decode import weChatCode, zXing_java


class CaptureThread(QThread):

    def __init__(self, window):
        super().__init__()
        self.ui = window

    def run(self):
        cap = cv2.VideoCapture(0)  # 读取视频或调用摄像头
        cap.set(3, 1280)
        cap.set(4, 720)
        while True:
            # 采集图片
            ret, frame = cap.read()
            img_captured = cv2.flip(frame, 1)
            img_detected = zXing_java(img_captured)
            cv2.imwrite(r"D:\Project\codeScan\images\01.jpg", img_detected)
            self.ui.imageScene_img.addPixmap(QPixmap(r"D:\Project\codeScan\images\01.jpg"))
