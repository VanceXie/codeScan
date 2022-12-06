# -*- coding: UTF-8 -*-
import cv2
from PIL import Image
from PyQt5.QtCore import QThread
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
        while True:
            self.ui.imageScene_img.addPixmap(QPixmap(self.image))


class CaptureThread(QThread):
    
    def __init__(self):
        super().__init__()
        self.image = None
        # self.thread1 = ShowThread(window, self.image)
        # self.ui = window
    
    def run(self):
        cap = cv2.VideoCapture(0)  # 读取视频或调用摄像头
        cap.set(3, 1280)
        cap.set(4, 720)
        # self.thread1.start()
        while True:
            # 采集图片
            ret, frame = cap.read()
            img_captured = cv2.flip(frame, 1)
            self.image = zXing_java(img_captured)
            
            # img_img = img_pil.toqimage()  # QImage
            # self.ui.imageScene_img.addPixmap(img_pix)
    

