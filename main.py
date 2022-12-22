import os
import sys

from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox

from gui.scan_display_main import *


# 动态载入
class mainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
    
    def closeEvent(self, event):
        # 创建一个消息盒子（提示框）
        quitMsgBox = QMessageBox()
        # 设置提示框的标题
        quitMsgBox.setWindowTitle('确认提示')
        # 设置提示框的内容
        quitMsgBox.setText('你确认退出吗？')
        # 设置按钮标准，一个yes一个no
        quitMsgBox.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        # 获取两个按钮并且修改显示文本
        buttonY = quitMsgBox.button(QMessageBox.Yes)
        buttonY.setText('确定')
        buttonN = quitMsgBox.button(QMessageBox.No)
        buttonN.setText('取消')
        quitMsgBox.exec_()
        # 判断返回值，如果点击的是Yes按钮，我们就关闭组件和应用，否则就忽略关闭事件
        if quitMsgBox.clickedButton() == buttonY:
            # self.captureThread.serialRobot.close()
            # self.captureThread.cap.release()
            # self.captureThread.quit()
            self.img_capture_thread.flag = True
            self.img_capture_thread.cap.release()
            self.img_capture_thread.quit()
            event.accept()
            os._exit(0)
            # self.closeSignal.emit()
        
        else:
            event.ignore()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = mainWindow()
    window.show()
    sys.exit(app.exec_())
