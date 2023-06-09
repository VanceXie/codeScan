# -*- coding: UTF-8 -*-
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import sys


class TableWidget(QWidget):
    def __init__(self, rows, cloumns):
        super().__init__()
        self.tableWidget = QTableWidget(rows, cloumns)
        self.table_setting()
        
        hhbox = QHBoxLayout()  # 横向布局
        hhbox.addWidget(self.tableWidget)  # 把表格加入布局
        self.setLayout(hhbox)  # 设定布局
        
        self.resize(920, 240)
        self.show()
    
    def table_setting(self):
        # 设置表头
        self.tableWidget.setHorizontalHeaderLabels(["第一列", "第二列", "第三列", "第四列", "第五列"])
        self.tableWidget.setVerticalHeaderLabels(["第一行", "第二行"])
        # 添加表格的文字内容.
        self.tableWidget.setItem(0, 0, QTableWidgetItem("你的名字"))
        self.tableWidget.setItem(0, 1, QTableWidgetItem("性别"))
        self.tableWidget.setItem(0, 2, QTableWidgetItem("出生日期"))
        self.tableWidget.setItem(0, 3, QTableWidgetItem("职业"))
        self.tableWidget.setItem(0, 4, QTableWidgetItem("收入"))
        
        lbp = QLabel()
        lbp.setPixmap(QPixmap("Male.png"))
        self.tableWidget.setCellWidget(1, 1, lbp)
        # 在表中添加一张图片
        
        twi = QTableWidgetItem("新海诚")
        twi.setFont(QFont("Times", 10, ))
        self.tableWidget.setItem(1, 0, twi)
        
        # 添加一个自己设置了大小和类型的文字。
        dte = QDateTimeEdit()
        dte.setDateTime(QDateTime.currentDateTime())
        dte.setDisplayFormat("yyyy/MM/dd")
        dte.setCalendarPopup(True)
        self.tableWidget.setCellWidget(1, 2, dte)
        
        # 添加一个弹出的日期选择，设置默认值为当前日期,显示格式为年月日。
        cbw = QComboBox()
        cbw.addItem("医生")
        cbw.addItem("老师")
        cbw.addItem("律师")
        self.tableWidget.setCellWidget(1, 3, cbw)
        
        # 添加了一个下拉选择框
        sb = QSpinBox()
        sb.setRange(1000, 10000)
        sb.setValue(5000)  # 设置最开始显示的数字
        sb.setDisplayIntegerBase(10)  # 这个是显示数字的进制，默认是十进制。
        sb.setSuffix("元")  # 设置后辍
        sb.setPrefix("RMB: ")  # 设置前辍
        sb.setSingleStep(100)
        self.tableWidget.setCellWidget(1, 4, sb)
