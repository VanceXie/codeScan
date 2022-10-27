import sys
import calendar
from PyQt5 import QtWidgets
class RunTest(calendar.Ui_Dialog):
    def __init__(self, myinherit):
        calendar.Ui_Dialog.setupUi(self, myinherit)


if __name__ == "__main__":
    mypro = QtWidgets.QApplication(sys.argv)
    mywin = QtWidgets.QDialog()
    RunTest(mywin)
    mywin.show()
    sys.exit(mypro.exec_())

