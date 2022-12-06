import jpype
from jpype import *


class ZXQRcode(object):
    def __init__(self):
        # jar包的路径
        self.jar_path1 = r"D:\Project\codeScan\jar_lib\core.jar"
        self.jar_path2 = r"D:\Project\codeScan\jar_lib\javase.jar"
        # 启动JVM
        try:
            jvm_path = jpype.getDefaultJVMPath()
            jpype.startJVM(jvmpath=jvm_path, classpath=[self.jar_path1, self.jar_path2])
        except:
            pass
        # 加载需要加载的类
        # self.File = JClass("java.io.File")
        self.byteArrayInputStream = JClass("java.io.ByteArrayInputStream")
        self.memoryCacheImageInputStream = JClass("javax.imageio.stream.MemoryCacheImageInputStream")
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
            jpype.shutdownJVM()
        except Exception as e:
            pass

    # 解析二维码
    def analysis_QR(self, image_byte):
        # 读入图片
        try:
            # imageFile = self.File(image_path)

            image_stream = self.byteArrayInputStream(image_byte)  # 根据字节数组创建图片的字节数组输入流
            image_cache = self.memoryCacheImageInputStream(image_stream)  # 根据输入流创建一个MemoryCacheImageInputStream对象
            bufferedImage = self.ImageIO.read(image_cache)  # 根据输入流创建一个BufferedImage对象
            source = self.BufferedImageLuminanceSource(
                bufferedImage)  # 根据BufferedImage对象创建一个BufferedImageLuminanceSource对象
            hybridBinarizer = self.HybridBinarizer(source)
            matrix = hybridBinarizer.getBlackMatrix()
            binaryBitmap = self.BinaryBitmap(hybridBinarizer)
            hints = self.Hashtable()
            hints.put(self.DecodeHintType.CHARACTER_SET, "UTF-8")
            detectorResult = self.Detector(matrix).detect(hints)
            resultPoints = self.MultiFormatReader().decodeWithState(binaryBitmap).getResultPoints()
            # coordinateList = [str(resultPoints[0]), str(resultPoints[1]), str(resultPoints[2]), str(resultPoints[3])]
            coordinateList = []
            for i in resultPoints:
                coordinateList.append([i.getX(), i.getY()])
            matrix1 = detectorResult.getBits()
            result = self.MultiFormatReader().decode(binaryBitmap, hints)
            return result.getText(), matrix1, coordinateList
        except Exception as e:
            print(e)
            return False
