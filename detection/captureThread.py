# -*- coding: UTF-8 -*-
import cv2
from PIL import Image
from PyQt5 import QtWidgets
from PyQt5.QtCore import QThread, QWaitCondition, QMutex, pyqtSignal
from PyQt5.QtGui import QPixmap
from detection.decode import weChatCode, zXing_java


class ShowThread(QThread):
    def __init__(self, window, image):
        super().__init__()
        self.ui = window
        img_pil = Image.fromarray(image)
        img_pix = img_pil.toqpixmap()  # QPixmap
        self.image = img_pix

    def run(self):
        self.ui.imageScene_img.addPixmap(QPixmap(self.image))


class CaptureThread(QThread):

    def __init__(self):
        super().__init__()
        self._isPause = False
        self.cond = QWaitCondition()
        self.mutex = QMutex()
        self.cap = cv2.VideoCapture(0)  # 读取视频或调用摄像头
        self.cap.set(3, 1280)
        self.cap.set(4, 720)
        self.image = None

    def pause(self):
        self._isPause = True
        self.cap.release()

    def resume(self):
        self._isPause = False
        self.cap = cv2.VideoCapture(0)  # 读取视频或调用摄像头
        self.cap.set(3, 1280)
        self.cap.set(4, 720)
        self.cond.wakeAll()

    def run(self):

        # self.thread1.start()
        while True:
            self.mutex.lock()
            if self._isPause:
                self.cond.wait(self.mutex)
                self.image = None

            else:
                # 采集图片
                ret, frame = self.cap.read()
                img_captured = cv2.flip(frame, 1)
                self.image = zXing_java(img_captured)
            self.mutex.unlock()
            # QtWidgets.QApplication.processEvents()

