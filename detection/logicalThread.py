# -*- coding: UTF-8 -*-
import time

import cv2
from PIL import Image
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage

from detection.decode import *


class CaptureThread(QThread):
    capture_signal = pyqtSignal(object)
    
    def __init__(self, width: int, height: int):
        super().__init__()
        self.cap = cv2.VideoCapture(0)  # 读取视频或调用摄像头
        self.cap.set(3, width)
        self.cap.set(4, height)
        self.flag = False
    
    def setImageSize(self, width, height):
        self.cap.set(3, width)
        self.cap.set(4, height)
    
    def run(self):
        while True:
            # 采集图片
            ret, frame = self.cap.read()
            if ret:
                img_captured = cv2.flip(frame, 1)
                image = decode_zxing(img_captured)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = QImage(image.data, image.shape[1], image.shape[0], QImage.Format_RGB888)
                pix_image = QPixmap.fromImage(image)
                self.capture_signal.emit(pix_image)
                if self.flag:
                    break
            # time.sleep(1000)


class DecodeThread():
    pass
