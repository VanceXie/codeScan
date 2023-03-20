# -*- coding: UTF-8 -*-
import sys
import time

from PyQt5.QtCore import QDateTime, QThread, pyqtSignal
from PyQt5.QtWidgets import QApplication, QDialog, QLineEdit


class BThread(QThread):
    update_signal = pyqtSignal(str)
    
    def run(self) -> None:
        while True:
            date = QDateTime.currentDateTime()
            self.update_signal.emit(str(date.toString("yyyy-MM-dd hh:mm:ss")))
            time.sleep(1)


class Window(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Clock')
        self.resize(400, 100)
        self.input = QLineEdit(self)
        self.input.resize(400, 100)
        self.initUI()

    def initUI(self):
        self.bThread = BThread()
        self.bThread.update_signal.connect(self.handleDisplay)
        self.bThread.start()

    def handleDisplay(self, date):
        self.input.setText(date)
app = QApplication(sys.argv)
window = Window()
window.show()
sys.exit(app.exec_())
