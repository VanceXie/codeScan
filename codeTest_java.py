import os
import sys

import jpype
from jpype import *


class ZXQRcode(object):
    def __init__(self):
        # jar包的路径
        self.jar_path = r"D:\Project\zxing\out\artifacts\core_jar\core.jar"
        # 启动JVM
        try:
            startJVM(r"D:\Program Files\Java\jdk-17.0.4.1\bin\server\jvm.dll", "-ea",
                     "-Djava.class.path=%s;%s" % (self.jar_path))
        except:
            pass
        # 加载需要加载的类
        self.File = JClass("java.io.File")
        self.ImageIO = JClass("javax.imageio.ImageIO")
        self.BufferedImageLuminanceSource = JClass("com.google.zxing.client.j2se.BufferedImageLuminanceSource")
        self.Hashtable = JClass("java.util.Hashtable")
        self.MultiFormatReader = JClass("com.google.zxing.MultiFormatReader")
        self.HybridBinarizer = JClass("com.google.zxing.common.HybridBinarizer")
        self.DecodeHintType = JClass("com.google.zxing.DecodeHintType")
        self.BinaryBitmap = JClass("com.google.zxing.BinaryBitmap")
        self.BitMatrix = JClass("com.google.zxing.common.BitMatrix")
        self.Detector = JClass("com.google.zxing.qrcode.detector.Detector")
        self.DetectorResult = JClass("com.google.zxing.common.DetectorResult")
    
    # 释放JVM
    def dels(self):
        try:
            shutdownJVM()
        except Exception as e:
            pass
    
    # 解析二维码
    def analysis_QR(self, image_path):
        # 读入图片
        try:
            imageFile = self.File(image_path)
            image = self.ImageIO.read(imageFile)
            source = self.BufferedImageLuminanceSource(image)
            hybridBinarizer = self.HybridBinarizer(source)
            matrix = hybridBinarizer.getBlackMatrix()
            binaryBitmap = self.BinaryBitmap(hybridBinarizer)
            hints = self.Hashtable()
            hints.put(self.DecodeHintType.CHARACTER_SET, "UTF-8")
            detectorResult = self.Detector(matrix).detect(hints)
            resultPoints = self.MultiFormatReader().decodeWithState(binaryBitmap).getResultPoints()
            coordinateList = [str(resultPoints[0]), str(resultPoints[1]), str(resultPoints[2])]
            matrix1 = detectorResult.getBits()
            result = self.MultiFormatReader().decode(binaryBitmap, hints)
            return result.getText(), matrix1, coordinateList
        except Exception as e:
            return False


image_path = r"D:\Test_Label1\2.tif"
zx = ZXQRcode()
print(zx.analysis_QR(image_path))
zx.dels()
