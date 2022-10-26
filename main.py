from PyQt5 import QtWidgets
import sys
app = QtWidgets.QApplication(sys.argv)
test_window = QtWidgets.QWidget()
test_window.resize(500,500)
test_window.setWindowTitle("测试窗口")
test_window.show()
sys.exit(app.exec())

import sys
from PyQt5 import QtWidgets, QtGui
if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_Form()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec())
