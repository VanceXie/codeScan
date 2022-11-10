import PyQt5.QtWidgets

import sys
app = PyQt5.QtWidgets.QApplication(sys.argv)
test_window = PyQt5.QtWidgets.QWidget()
test_window.resize(500,500)
test_window.setWindowTitle("测试窗口")
test_window.show()
sys.exit(app.exec())