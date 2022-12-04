import sys
import cv2
import numpy as np
from PIL import Image
from PyQt5.QtWidgets import QApplication, QMainWindow

from gui.scan_display_main import Ui_MainWindow


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
    sys.exit(app.exec_())
