# -*- coding: utf-8 -*-
from PIL import Image
# Form implementation generated from reading ui file 'scan_display_main.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QApplication

from detection.logicalThread import *


class Ui_MainWindow(object):
    def __init__(self):
        self.showThread = ShowThread()

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1280, 960)
        font = QtGui.QFont()
        font.setFamily("Adobe Heiti Std")
        font.setPointSize(10)
        font.setBold(False)
        font.setWeight(50)
        MainWindow.setFont(font)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/Rose/camera.svg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        MainWindow.setWindowIcon(icon)
        MainWindow.setWindowOpacity(1.0)
        MainWindow.setIconSize(QtCore.QSize(20, 20))
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        font = QtGui.QFont()
        font.setFamily("Adobe 黑体 Std R")
        font.setPointSize(12)
        font.setBold(False)
        font.setWeight(50)
        self.centralwidget.setFont(font)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setContentsMargins(5, 0, 5, 0)
        self.gridLayout.setSpacing(0)
        self.gridLayout.setObjectName("gridLayout")
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.tabWidget.sizePolicy().hasHeightForWidth())
        self.tabWidget.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("Adobe 黑体 Std R")
        font.setPointSize(11)
        font.setBold(False)
        font.setWeight(50)
        self.tabWidget.setFont(font)
        self.tabWidget.setTabPosition(QtWidgets.QTabWidget.West)
        self.tabWidget.setTabShape(QtWidgets.QTabWidget.Rounded)
        self.tabWidget.setIconSize(QtCore.QSize(20, 20))
        self.tabWidget.setDocumentMode(False)
        self.tabWidget.setTabsClosable(False)
        self.tabWidget.setMovable(True)
        self.tabWidget.setTabBarAutoHide(True)
        self.tabWidget.setObjectName("tabWidget")
        self.imageShow_table = QtWidgets.QWidget()
        font = QtGui.QFont()
        font.setFamily("Adobe 黑体 Std R")
        font.setPointSize(11)
        font.setBold(False)
        font.setWeight(50)
        self.imageShow_table.setFont(font)
        self.imageShow_table.setObjectName("imageShow_table")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.imageShow_table)
        self.gridLayout_2.setContentsMargins(5, 5, 5, 5)
        self.gridLayout_2.setSpacing(5)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setSpacing(5)
        self.verticalLayout.setObjectName("verticalLayout")

        self.imageScene_img = QtWidgets.QGraphicsScene(self.imageShow_table)
        self.imageScene_img.addPixmap(QPixmap(r"./gui/images/welcome.jpg"))

        self.imageView_img = QtWidgets.QGraphicsView(self.imageScene_img)
        self.imageView_img.setObjectName("imageInput_img")
        self.imageView_img.setScene(self.imageScene_img)

        self.verticalLayout.addWidget(self.imageView_img)

        self.tableWidget = QtWidgets.QTableWidget(self.imageShow_table)
        self.tableWidget.setShowGrid(True)
        self.tableWidget.setObjectName("tableWidget")
        self.tableWidget.setColumnCount(8)
        self.tableWidget.setRowCount(2)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setVerticalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setVerticalHeaderItem(1, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(1, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(2, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(3, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(4, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(5, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(6, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(7, item)
        self.tableWidget.horizontalHeader().setVisible(True)
        self.tableWidget.horizontalHeader().setCascadingSectionResizes(False)
        self.tableWidget.horizontalHeader().setDefaultSectionSize(105)
        self.tableWidget.horizontalHeader().setMinimumSectionSize(100)
        self.tableWidget.horizontalHeader().setSortIndicatorShown(False)
        self.tableWidget.horizontalHeader().setStretchLastSection(True)
        self.tableWidget.verticalHeader().setDefaultSectionSize(40)
        self.tableWidget.verticalHeader().setMinimumSectionSize(30)
        self.tableWidget.verticalHeader().setSortIndicatorShown(True)
        self.tableWidget.verticalHeader().setStretchLastSection(False)
        self.verticalLayout.addWidget(self.tableWidget)
        self.frame = QtWidgets.QFrame(self.imageShow_table)
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.frame)
        self.horizontalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_3.setSpacing(0)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        spacerItem = QtWidgets.QSpacerItem(394, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_3.addItem(spacerItem)

        self.imageCapture = QtWidgets.QPushButton(self.frame)
        self.imageCapture.clicked.connect(self.capture_signal_slot)

        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.imageCapture.sizePolicy().hasHeightForWidth())
        self.imageCapture.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("Adobe 黑体 Std R")
        font.setPointSize(12)
        font.setBold(False)
        font.setWeight(50)
        self.imageCapture.setFont(font)
        self.imageCapture.setMouseTracking(False)
        self.imageCapture.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.imageCapture.setAutoFillBackground(False)
        self.imageCapture.setStyleSheet("")
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap(":/Rose/start.svg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.imageCapture.setIcon(icon1)
        self.imageCapture.setCheckable(True)
        self.imageCapture.setAutoDefault(False)
        self.imageCapture.setFlat(False)
        self.imageCapture.setObjectName("imageCapture")
        self.horizontalLayout_3.addWidget(self.imageCapture)
        spacerItem1 = QtWidgets.QSpacerItem(309, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_3.addItem(spacerItem1)
        self.verticalLayout.addWidget(self.frame)
        self.verticalLayout.setStretch(0, 5)
        self.verticalLayout.setStretch(1, 2)
        self.gridLayout_2.addLayout(self.verticalLayout, 0, 0, 1, 1)
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setSpacing(7)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.treeWidget = QtWidgets.QTreeWidget(self.imageShow_table)
        self.treeWidget.setSelectionMode(QtWidgets.QAbstractItemView.ContiguousSelection)
        self.treeWidget.setAutoExpandDelay(-1)
        self.treeWidget.setRootIsDecorated(True)
        self.treeWidget.setUniformRowHeights(False)
        self.treeWidget.setItemsExpandable(True)
        self.treeWidget.setAnimated(True)
        self.treeWidget.setAllColumnsShowFocus(False)
        self.treeWidget.setHeaderHidden(True)
        self.treeWidget.setObjectName("treeWidget")
        item_0 = QtWidgets.QTreeWidgetItem(self.treeWidget)
        item_0.setCheckState(0, QtCore.Qt.Checked)
        item_1 = QtWidgets.QTreeWidgetItem(item_0)
        item_1.setCheckState(0, QtCore.Qt.Checked)
        item_2 = QtWidgets.QTreeWidgetItem(item_1)
        item_2.setCheckState(0, QtCore.Qt.Checked)
        item_3 = QtWidgets.QTreeWidgetItem(item_2)
        item_3.setCheckState(0, QtCore.Qt.Checked)
        item_3 = QtWidgets.QTreeWidgetItem(item_2)
        item_3.setCheckState(0, QtCore.Qt.Checked)
        item_3 = QtWidgets.QTreeWidgetItem(item_2)
        item_3.setCheckState(0, QtCore.Qt.Checked)
        item_3 = QtWidgets.QTreeWidgetItem(item_2)
        item_3.setCheckState(0, QtCore.Qt.Checked)
        item_3 = QtWidgets.QTreeWidgetItem(item_2)
        item_3.setCheckState(0, QtCore.Qt.Checked)
        item_2 = QtWidgets.QTreeWidgetItem(item_1)
        item_2.setCheckState(0, QtCore.Qt.Checked)
        item_3 = QtWidgets.QTreeWidgetItem(item_2)
        item_3.setCheckState(0, QtCore.Qt.Checked)
        item_3 = QtWidgets.QTreeWidgetItem(item_2)
        item_3.setCheckState(0, QtCore.Qt.Checked)
        item_3 = QtWidgets.QTreeWidgetItem(item_2)
        item_3.setCheckState(0, QtCore.Qt.Checked)
        item_3 = QtWidgets.QTreeWidgetItem(item_2)
        item_3.setCheckState(0, QtCore.Qt.Checked)
        item_3 = QtWidgets.QTreeWidgetItem(item_2)
        item_3.setCheckState(0, QtCore.Qt.Checked)
        item_1 = QtWidgets.QTreeWidgetItem(item_0)
        item_1.setCheckState(0, QtCore.Qt.Checked)
        item_2 = QtWidgets.QTreeWidgetItem(item_1)
        item_2.setCheckState(0, QtCore.Qt.Checked)
        item_2 = QtWidgets.QTreeWidgetItem(item_1)
        item_2.setCheckState(0, QtCore.Qt.Checked)
        item_2 = QtWidgets.QTreeWidgetItem(item_1)
        item_2.setCheckState(0, QtCore.Qt.Checked)
        item_2 = QtWidgets.QTreeWidgetItem(item_1)
        item_2.setCheckState(0, QtCore.Qt.Checked)
        item_2 = QtWidgets.QTreeWidgetItem(item_1)
        item_2.setCheckState(0, QtCore.Qt.Checked)
        item_2 = QtWidgets.QTreeWidgetItem(item_1)
        item_2.setCheckState(0, QtCore.Qt.Checked)
        item_2 = QtWidgets.QTreeWidgetItem(item_1)
        item_2.setCheckState(0, QtCore.Qt.Checked)
        item_0 = QtWidgets.QTreeWidgetItem(self.treeWidget)
        item_0.setCheckState(0, QtCore.Qt.Checked)
        item_1 = QtWidgets.QTreeWidgetItem(item_0)
        item_1.setCheckState(0, QtCore.Qt.Checked)
        item_2 = QtWidgets.QTreeWidgetItem(item_1)
        item_2.setCheckState(0, QtCore.Qt.Checked)
        item_2 = QtWidgets.QTreeWidgetItem(item_1)
        item_2.setCheckState(0, QtCore.Qt.Checked)
        item_2 = QtWidgets.QTreeWidgetItem(item_1)
        item_2.setCheckState(0, QtCore.Qt.Checked)
        item_2 = QtWidgets.QTreeWidgetItem(item_1)
        item_2.setCheckState(0, QtCore.Qt.Checked)
        item_1 = QtWidgets.QTreeWidgetItem(item_0)
        item_1.setCheckState(0, QtCore.Qt.Checked)
        item_2 = QtWidgets.QTreeWidgetItem(item_1)
        item_2.setCheckState(0, QtCore.Qt.Checked)
        item_2 = QtWidgets.QTreeWidgetItem(item_1)
        item_2.setCheckState(0, QtCore.Qt.Checked)
        item_2 = QtWidgets.QTreeWidgetItem(item_1)
        item_2.setCheckState(0, QtCore.Qt.Checked)
        self.treeWidget.header().setVisible(False)
        self.treeWidget.header().setCascadingSectionResizes(False)
        self.treeWidget.header().setDefaultSectionSize(33)
        self.treeWidget.header().setHighlightSections(True)
        self.treeWidget.header().setStretchLastSection(True)
        self.verticalLayout_2.addWidget(self.treeWidget)
        self.frame_2 = QtWidgets.QFrame(self.imageShow_table)
        self.frame_2.setObjectName("frame_2")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout(self.frame_2)
        self.horizontalLayout_4.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_4.setSpacing(0)
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        spacerItem2 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_4.addItem(spacerItem2)
        self.pushButton = QtWidgets.QPushButton(self.frame_2)
        font = QtGui.QFont()
        font.setFamily("Adobe 黑体 Std R")
        font.setPointSize(12)
        font.setBold(False)
        font.setWeight(50)
        self.pushButton.setFont(font)
        self.pushButton.setCheckable(True)
        self.pushButton.setChecked(True)
        self.pushButton.setObjectName("pushButton")
        self.horizontalLayout_4.addWidget(self.pushButton)
        spacerItem3 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_4.addItem(spacerItem3)
        self.verticalLayout_2.addWidget(self.frame_2)
        self.gridLayout_2.addLayout(self.verticalLayout_2, 0, 1, 1, 1)
        self.gridLayout_2.setColumnStretch(0, 3)
        self.gridLayout_2.setColumnStretch(1, 1)
        self.tabWidget.addTab(self.imageShow_table, icon, "")
        self.parameter_table = QtWidgets.QWidget()
        font = QtGui.QFont()
        font.setFamily("Adobe 黑体 Std R")
        font.setPointSize(12)
        font.setBold(False)
        font.setWeight(50)
        self.parameter_table.setFont(font)
        self.parameter_table.setObjectName("parameter_table")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.parameter_table)
        self.gridLayout_3.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_3.setSpacing(0)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.scrollArea = QtWidgets.QScrollArea(self.parameter_table)
        font = QtGui.QFont()
        font.setFamily("Adobe 黑体 Std R")
        font.setPointSize(12)
        font.setBold(False)
        font.setWeight(50)
        self.scrollArea.setFont(font)
        self.scrollArea.setAutoFillBackground(False)
        self.scrollArea.setStyleSheet("background-color:transparent;")
        self.scrollArea.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.scrollArea.setFrameShadow(QtWidgets.QFrame.Plain)
        self.scrollArea.setLineWidth(0)
        self.scrollArea.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
        self.scrollArea.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setObjectName("scrollArea")
        self.scrollAreaWidgetContents = QtWidgets.QWidget()
        self.scrollAreaWidgetContents.setGeometry(QtCore.QRect(0, 0, 1224, 904))
        font = QtGui.QFont()
        font.setFamily("Adobe 黑体 Std R")
        font.setPointSize(11)
        font.setBold(False)
        font.setWeight(50)
        self.scrollAreaWidgetContents.setFont(font)
        self.scrollAreaWidgetContents.setObjectName("scrollAreaWidgetContents")
        self.gridLayout_4 = QtWidgets.QGridLayout(self.scrollAreaWidgetContents)
        self.gridLayout_4.setContentsMargins(5, 5, 0, 0)
        self.gridLayout_4.setSpacing(0)
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.toolBox = QtWidgets.QToolBox(self.scrollAreaWidgetContents)
        font = QtGui.QFont()
        font.setFamily("Adobe 黑体 Std R")
        font.setPointSize(12)
        font.setBold(False)
        font.setWeight(50)
        self.toolBox.setFont(font)
        self.toolBox.setObjectName("toolBox")
        self.codesacn_para = QtWidgets.QWidget()
        self.codesacn_para.setGeometry(QtCore.QRect(0, 0, 1219, 841))
        font = QtGui.QFont()
        font.setFamily("Adobe 黑体 Std R")
        font.setPointSize(12)
        font.setBold(False)
        font.setWeight(50)
        self.codesacn_para.setFont(font)
        self.codesacn_para.setObjectName("codesacn_para")
        self.gridLayout_5 = QtWidgets.QGridLayout(self.codesacn_para)
        self.gridLayout_5.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_5.setSpacing(0)
        self.gridLayout_5.setObjectName("gridLayout_5")
        self.scrollArea_2 = QtWidgets.QScrollArea(self.codesacn_para)
        font = QtGui.QFont()
        font.setFamily("Adobe 黑体 Std R")
        font.setPointSize(12)
        font.setBold(False)
        font.setWeight(50)
        self.scrollArea_2.setFont(font)
        self.scrollArea_2.setFrameShape(QtWidgets.QFrame.Box)
        self.scrollArea_2.setFrameShadow(QtWidgets.QFrame.Plain)
        self.scrollArea_2.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
        self.scrollArea_2.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
        self.scrollArea_2.setWidgetResizable(True)
        self.scrollArea_2.setObjectName("scrollArea_2")
        self.scrollAreaWidgetContents_2 = QtWidgets.QWidget()
        self.scrollAreaWidgetContents_2.setGeometry(QtCore.QRect(0, 0, 1204, 826))
        font = QtGui.QFont()
        font.setFamily("Adobe 黑体 Std R")
        font.setPointSize(12)
        font.setBold(False)
        font.setWeight(50)
        self.scrollAreaWidgetContents_2.setFont(font)
        self.scrollAreaWidgetContents_2.setObjectName("scrollAreaWidgetContents_2")
        self.scrollArea_2.setWidget(self.scrollAreaWidgetContents_2)
        self.gridLayout_5.addWidget(self.scrollArea_2, 0, 0, 1, 1)
        self.toolBox.addItem(self.codesacn_para, "")
        self.detect_para = QtWidgets.QWidget()
        self.detect_para.setGeometry(QtCore.QRect(0, 0, 87, 59))
        font = QtGui.QFont()
        font.setFamily("Adobe 黑体 Std R")
        font.setPointSize(12)
        font.setBold(False)
        font.setWeight(50)
        self.detect_para.setFont(font)
        self.detect_para.setObjectName("detect_para")
        self.gridLayout_6 = QtWidgets.QGridLayout(self.detect_para)
        self.gridLayout_6.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_6.setSpacing(0)
        self.gridLayout_6.setObjectName("gridLayout_6")
        self.scrollArea_3 = QtWidgets.QScrollArea(self.detect_para)
        font = QtGui.QFont()
        font.setFamily("Adobe 黑体 Std R")
        font.setPointSize(12)
        font.setBold(False)
        font.setWeight(50)
        self.scrollArea_3.setFont(font)
        self.scrollArea_3.setFrameShape(QtWidgets.QFrame.Box)
        self.scrollArea_3.setFrameShadow(QtWidgets.QFrame.Plain)
        self.scrollArea_3.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
        self.scrollArea_3.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
        self.scrollArea_3.setWidgetResizable(True)
        self.scrollArea_3.setObjectName("scrollArea_3")
        self.scrollAreaWidgetContents_3 = QtWidgets.QWidget()
        self.scrollAreaWidgetContents_3.setGeometry(QtCore.QRect(0, 0, 72, 44))
        font = QtGui.QFont()
        font.setFamily("Adobe 黑体 Std R")
        font.setPointSize(12)
        font.setBold(False)
        font.setWeight(50)
        self.scrollAreaWidgetContents_3.setFont(font)
        self.scrollAreaWidgetContents_3.setObjectName("scrollAreaWidgetContents_3")
        self.scrollArea_3.setWidget(self.scrollAreaWidgetContents_3)
        self.gridLayout_6.addWidget(self.scrollArea_3, 0, 0, 1, 1)
        self.toolBox.addItem(self.detect_para, "")
        self.gridLayout_4.addWidget(self.toolBox, 0, 0, 1, 1)
        self.scrollArea.setWidget(self.scrollAreaWidgetContents)
        self.gridLayout_3.addWidget(self.scrollArea, 0, 0, 1, 1)
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap(":/Rose/save.svg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.tabWidget.addTab(self.parameter_table, icon2, "")
        self.gridLayout.addWidget(self.tabWidget, 0, 0, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1280, 19))
        font = QtGui.QFont()
        font.setFamily("Adobe 黑体 Std R")
        font.setPointSize(10)
        font.setBold(False)
        font.setWeight(50)
        self.menubar.setFont(font)
        self.menubar.setObjectName("menubar")
        self.file = QtWidgets.QMenu(self.menubar)
        self.file.setGeometry(QtCore.QRect(265, 111, 119, 91))
        self.file.setObjectName("file")
        self.log = QtWidgets.QMenu(self.menubar)
        self.log.setObjectName("log")
        self.help = QtWidgets.QMenu(self.menubar)
        self.help.setObjectName("help")
        self.view = QtWidgets.QMenu(self.menubar)
        self.view.setObjectName("view")
        self.setting = QtWidgets.QMenu(self.menubar)
        self.setting.setObjectName("setting")
        self.window = QtWidgets.QMenu(self.menubar)
        self.window.setObjectName("window")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        font = QtGui.QFont()
        font.setFamily("Adobe 黑体 Std R")
        font.setPointSize(10)
        font.setBold(False)
        font.setWeight(50)
        self.statusbar.setFont(font)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.tutorials = QtWidgets.QAction(MainWindow)
        self.tutorials.setObjectName("tutorials")
        self.about = QtWidgets.QAction(MainWindow)
        self.about.setObjectName("about")
        self.showLog = QtWidgets.QAction(MainWindow)
        self.showLog.setObjectName("showLog")
        self.open = QtWidgets.QAction(MainWindow)
        self.open.setObjectName("open")
        self.save = QtWidgets.QAction(MainWindow)
        self.save.setObjectName("save")
        self.saveAs = QtWidgets.QAction(MainWindow)
        self.saveAs.setObjectName("saveAs")
        self.logLocation = QtWidgets.QAction(MainWindow)
        self.logLocation.setObjectName("logLocation")
        self.minimize = QtWidgets.QAction(MainWindow)
        self.minimize.setObjectName("minimize")
        self.function = QtWidgets.QAction(MainWindow)
        self.function.setCheckable(True)
        self.function.setChecked(True)
        self.function.setObjectName("function")
        self.actionzhuto = QtWidgets.QAction(MainWindow)
        self.actionzhuto.setObjectName("actionzhuto")
        self.actionziti = QtWidgets.QAction(MainWindow)
        self.actionziti.setObjectName("actionziti")
        self.file.addAction(self.open)
        self.file.addAction(self.save)
        self.file.addAction(self.saveAs)
        self.log.addAction(self.showLog)
        self.log.addAction(self.logLocation)
        self.help.addAction(self.tutorials)
        self.help.addSeparator()
        self.help.addAction(self.about)
        self.view.addAction(self.function)
        self.setting.addAction(self.actionzhuto)
        self.setting.addAction(self.actionziti)
        self.window.addAction(self.minimize)
        self.menubar.addAction(self.file.menuAction())
        self.menubar.addAction(self.view.menuAction())
        self.menubar.addAction(self.setting.menuAction())
        self.menubar.addAction(self.window.menuAction())
        self.menubar.addAction(self.log.menuAction())
        self.menubar.addAction(self.help.menuAction())

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(0)
        self.toolBox.setCurrentIndex(0)
        self.toolBox.layout().setSpacing(5)
        self.imageCapture.clicked.connect(self.imageView_img.show)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        MainWindow.setTabOrder(self.tabWidget, self.imageCapture)
        MainWindow.setTabOrder(self.imageCapture, self.imageView_img)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "智能相机"))
        self.tableWidget.setSortingEnabled(True)
        item = self.tableWidget.verticalHeaderItem(0)
        item.setText(_translate("MainWindow", "1"))
        item = self.tableWidget.verticalHeaderItem(1)
        item.setText(_translate("MainWindow", "2"))
        item = self.tableWidget.horizontalHeaderItem(0)
        item.setText(_translate("MainWindow", "检测结果"))
        item = self.tableWidget.horizontalHeaderItem(1)
        item.setText(_translate("MainWindow", "条码缺陷"))
        item = self.tableWidget.horizontalHeaderItem(2)
        item.setText(_translate("MainWindow", "条码结构特性"))
        item = self.tableWidget.horizontalHeaderItem(3)
        item.setText(_translate("MainWindow", "条码光学特性"))
        item = self.tableWidget.horizontalHeaderItem(4)
        item.setText(_translate("MainWindow", "字符缺陷"))
        item = self.tableWidget.horizontalHeaderItem(5)
        item.setText(_translate("MainWindow", "字符结构特性"))
        item = self.tableWidget.horizontalHeaderItem(6)
        item.setText(_translate("MainWindow", "字码一致性"))
        item = self.tableWidget.horizontalHeaderItem(7)
        item.setText(_translate("MainWindow", "标签位置姿态"))
        self.imageCapture.setToolTip(
            _translate("MainWindow", "<html><head/><body><p>开始采集图像 Ctrl+P</p></body></html>"))
        self.imageCapture.setStatusTip(_translate("MainWindow", "图像采集中"))
        self.imageCapture.setText(_translate("MainWindow", "开始采图"))
        self.imageCapture.setShortcut(_translate("MainWindow", "Ctrl+P"))
        self.treeWidget.headerItem().setText(0, _translate("MainWindow", "功能选项"))
        __sortingEnabled = self.treeWidget.isSortingEnabled()
        self.treeWidget.setSortingEnabled(False)
        self.treeWidget.topLevelItem(0).setText(0, _translate("MainWindow", "扫码功能"))
        self.treeWidget.topLevelItem(0).child(0).setText(0, _translate("MainWindow", "一维码"))
        self.treeWidget.topLevelItem(0).child(0).child(0).setText(0, _translate("MainWindow", "商品码"))
        self.treeWidget.topLevelItem(0).child(0).child(0).child(0).setText(0, _translate("MainWindow", "UPC-A"))
        self.treeWidget.topLevelItem(0).child(0).child(0).child(1).setText(0, _translate("MainWindow", "UPC-E"))
        self.treeWidget.topLevelItem(0).child(0).child(0).child(2).setText(0, _translate("MainWindow", "EAN-8"))
        self.treeWidget.topLevelItem(0).child(0).child(0).child(3).setText(0, _translate("MainWindow", "EAN-13"))
        self.treeWidget.topLevelItem(0).child(0).child(0).child(4).setText(0, _translate("MainWindow",
                                                                                         "UPC/EAN Extension 2/5"))
        self.treeWidget.topLevelItem(0).child(0).child(1).setText(0, _translate("MainWindow", "工业码"))
        self.treeWidget.topLevelItem(0).child(0).child(1).child(0).setText(0, _translate("MainWindow", "Code 39"))
        self.treeWidget.topLevelItem(0).child(0).child(1).child(1).setText(0, _translate("MainWindow", "Code 93"))
        self.treeWidget.topLevelItem(0).child(0).child(1).child(2).setText(0, _translate("MainWindow", "Code 128"))
        self.treeWidget.topLevelItem(0).child(0).child(1).child(3).setText(0, _translate("MainWindow", "Codebar"))
        self.treeWidget.topLevelItem(0).child(0).child(1).child(4).setText(0, _translate("MainWindow", "ITF"))
        self.treeWidget.topLevelItem(0).child(1).setText(0, _translate("MainWindow", "二维码"))
        self.treeWidget.topLevelItem(0).child(1).child(0).setText(0, _translate("MainWindow", "QR Code"))
        self.treeWidget.topLevelItem(0).child(1).child(1).setText(0, _translate("MainWindow", "Data Matrix"))
        self.treeWidget.topLevelItem(0).child(1).child(2).setText(0, _translate("MainWindow", "Aztec"))
        self.treeWidget.topLevelItem(0).child(1).child(3).setText(0, _translate("MainWindow", "PDF 417"))
        self.treeWidget.topLevelItem(0).child(1).child(4).setText(0, _translate("MainWindow", "MaxiCode"))
        self.treeWidget.topLevelItem(0).child(1).child(5).setText(0, _translate("MainWindow", "RSS-14"))
        self.treeWidget.topLevelItem(0).child(1).child(6).setText(0, _translate("MainWindow", "RSS-Expanded"))
        self.treeWidget.topLevelItem(1).setText(0, _translate("MainWindow", "检测功能"))
        self.treeWidget.topLevelItem(1).child(0).setText(0, _translate("MainWindow", "条码检测"))
        self.treeWidget.topLevelItem(1).child(0).child(0).setText(0, _translate("MainWindow", "条码缺陷"))
        self.treeWidget.topLevelItem(1).child(0).child(1).setText(0, _translate("MainWindow", "条码结构特性"))
        self.treeWidget.topLevelItem(1).child(0).child(2).setText(0, _translate("MainWindow", "条码光学特性"))
        self.treeWidget.topLevelItem(1).child(0).child(3).setText(0, _translate("MainWindow", "标签位置姿态"))
        self.treeWidget.topLevelItem(1).child(1).setText(0, _translate("MainWindow", "字符检测"))
        self.treeWidget.topLevelItem(1).child(1).child(0).setText(0, _translate("MainWindow", "字符缺陷"))
        self.treeWidget.topLevelItem(1).child(1).child(1).setText(0, _translate("MainWindow", "字码一致性"))
        self.treeWidget.topLevelItem(1).child(1).child(2).setText(0, _translate("MainWindow", "字符结构特性"))
        self.treeWidget.setSortingEnabled(__sortingEnabled)
        self.pushButton.setText(_translate("MainWindow", "功能确认"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.imageShow_table), _translate("MainWindow", "显示界面"))
        self.toolBox.setItemText(self.toolBox.indexOf(self.codesacn_para), _translate("MainWindow", "扫码参数"))
        self.toolBox.setItemText(self.toolBox.indexOf(self.detect_para), _translate("MainWindow", "检测参数"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.parameter_table), _translate("MainWindow", "参数设置"))
        self.file.setTitle(_translate("MainWindow", "文件&F"))
        self.log.setTitle(_translate("MainWindow", "日志&L"))
        self.help.setTitle(_translate("MainWindow", "帮助&H"))
        self.view.setTitle(_translate("MainWindow", "视图&V"))
        self.setting.setTitle(_translate("MainWindow", "设置&S"))
        self.window.setTitle(_translate("MainWindow", "窗口&W"))
        self.tutorials.setText(_translate("MainWindow", "教程&T"))
        self.about.setText(_translate("MainWindow", "关于&A"))
        self.showLog.setText(_translate("MainWindow", "查看当前日志&E"))
        self.open.setText(_translate("MainWindow", "打开&O"))
        self.open.setShortcut(_translate("MainWindow", "Ctrl+O"))
        self.save.setText(_translate("MainWindow", "保存&S"))
        self.save.setShortcut(_translate("MainWindow", "Ctrl+S"))
        self.saveAs.setText(_translate("MainWindow", "另存为&S&A"))
        self.saveAs.setShortcut(_translate("MainWindow", "Ctrl+S, Ctrl+A"))
        self.logLocation.setText(_translate("MainWindow", "打开日志所在位置&O"))
        self.minimize.setText(_translate("MainWindow", "最小化&M"))
        self.minimize.setShortcut(_translate("MainWindow", "Ctrl+M"))
        self.function.setText(_translate("MainWindow", "功能&F"))
        self.actionzhuto.setText(_translate("MainWindow", "主题&T"))
        self.actionziti.setText(_translate("MainWindow", "字体&F"))

    def capture_signal_slot(self, status):
        if status:
            self.img_capture_thread = CaptureThread()
            self.img_capture_thread.capture_signal.connect(self.show_slot)
            self.img_capture_thread.start()
        else:
            self.img_capture_thread.cap.release()
            self.img_capture_thread.quit()

    def show_slot(self, image):
        img_pil = Image.fromarray(image)
        img_pix = img_pil.toqpixmap()  # QPixmap
        # self.imageScene_img.clear()
        self.imageScene_img.addPixmap(img_pix)
        self.imageView_img.viewport().update()
        QApplication.processEvents()
