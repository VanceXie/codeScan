import sys

from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow

from gui.scan_display_main import Ui_MainWindow
from detection.captureThread import CaptureThread


# 动态载入
class mainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        # PyQt5
        # self.gui = uic.loadUi('./gui/scan_display_main.gui')
        self.ui = Ui_MainWindow()
        self.ui.show()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = mainWindow()

    # capThread = CaptureThread(window)
    # capThread.start()
    if not app.exec_():
        window.ui.img_capture_thread.terminate()
        # window.ui.img_capture_thread.wait()

        sys.exit(app.exec_())
