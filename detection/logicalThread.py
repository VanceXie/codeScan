# -*- coding: UTF-8 -*-
import time

import cv2
from PIL import Image
from PyQt5 import QtWidgets
from PyQt5.QtCore import QThread, QWaitCondition, QMutex, pyqtSignal
from PyQt5.QtGui import QPixmap
from detection.decode import decode_wechat, decode_zxing


class ShowThread(QThread):
    showSignal = pyqtSignal(object)

    def __init__(self, window, image):
        super().__init__()
        self.ui = window
        img_pil = Image.fromarray(image)
        img_pix = img_pil.toqpixmap()  # QPixmap
        self.image = img_pix

    def run(self):
        self.imageScene_img.addPixmap(QPixmap(self.image))


class CaptureThread(QThread):
    capture_signal = pyqtSignal(object)

    def __init__(self):
        super().__init__()
        self.cap = cv2.VideoCapture(0)  # 读取视频或调用摄像头
        self.cap.set(3, 1280)
        self.cap.set(4, 720)

    def run(self):
        while True:
            # 采集图片
            ret, frame = self.cap.read()
            img_captured = cv2.flip(frame, 1)
            image = decode_zxing(img_captured)
            self.capture_signal.emit(image)
            time.sleep(1)
            # QtWidgets.QApplication.processEvents()
