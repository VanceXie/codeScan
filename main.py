# import sys
# from PyQt5 import QtWidgets
# import ui.scan_display_main
#
#
# if __name__ == "__main__":
#     mypro = QtWidgets.QApplication(sys.argv)
#     mywin = ui.scan_display_main.Ui_MainWindow()
#     mywin.show()
#     sys.exit(mypro.exec_())
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5 import uic


# 动态载入
class mainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        # PyQt5
        self.ui = uic.loadUi('./ui/scan_display_main.ui')
        # 这里与静态载入不同，使用 self.ui.show()
        # 如果使用 self.show(),会产生一个空白的 MainWindow
        self.ui.show()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = mainWindow()
    sys.exit(app.exec_())
