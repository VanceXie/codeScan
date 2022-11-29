# -*- coding: utf-8 -*-
import sys
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *


class TreeWidgetWithState(QWidget):
    def __init__(self):
        super(TreeWidgetWithState, self).__init__()
        
        self.tree = QTreeWidget()  # 实例化一个TreeWidget对象
        self.tree.setColumnCount(1)  # 设置部件的列数为1
        self.tree.setDropIndicatorShown(True)
        
        self.tree.setSelectionMode(QAbstractItemView.ExtendedSelection)  # 设置item可以多选
        self.tree.setHeaderLabels(['功能选项'])  # 设置头部信息对应列的标识符
        
        # 设置root为self.tree的子树，故root是根节点
        code = QTreeWidgetItem(self.tree)
        code.setText(0, '扫码功能')  # 设置根节点的名称
        code.setCheckState(0, Qt.Unchecked)
        # 为root节点设置子结点
        child1 = QTreeWidgetItem(code)
        child1.setText(0, '一维码')
        child1.setCheckState(0, Qt.Unchecked);
        
        child2 = QTreeWidgetItem(code)
        child2.setText(0, '二维码')
        child2.setCheckState(0, Qt.Unchecked);
        
        defect = QTreeWidgetItem(self.tree)
        defect.setText(0, '检测功能')  # 设置根节点的名称
        defect.setCheckState(0, Qt.Unchecked)
        
        lay = QVBoxLayout()
        lay.addWidget(self.tree)
        
        self.tree.itemChanged.connect(self.handleChanged)
        
        self.tree.addTopLevelItem(code)
        self.setLayout(lay)  # 将tree部件设置为该窗口的核心框架
    
    def handleChanged(self, item, column):
        # 当check状态改变时得到他的状态。
        if item.checkState(column) == Qt.Checked:
            print
            "checked", item, item.text(column)
        if item.checkState(column) == Qt.Unchecked:
            print
            "unchecked", item, item.text(column)
    
    def getText(self):
        # 当item多选时获取选择的某几项，text(0),代表第0列，text(1),代表第1列
        Item_list = self.tree.selectedItems()
        for ii in Item_list:
            print(ii.text(1))


app = QApplication(sys.argv)
app.aboutToQuit.connect(app.deleteLater)
tp = TreeWidgetWithState()
tp.show()
app.exec_()
