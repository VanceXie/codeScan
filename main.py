import sys

from PyQt5.QtWidgets import QApplication, QMainWindow

from gui.scan_display_main import *


# 动态载入
class mainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
    
    def closeEvent(self, a0: QtGui.QCloseEvent) -> None:
        self.img_capture_thread.flag = True
        self.img_capture_thread.cap.release()
        self.img_capture_thread.quit()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = mainWindow()
    window.show()
    sys.exit(app.exec_())
