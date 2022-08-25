import sys
from PyQt5 import QtCore, QtWidgets
from Frame import Ui_MainWindow
from predict2 import *
from PyQt5.QtWidgets import QApplication,QMainWindow
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import pandas as pd
import numpy as np
from model import Net
from Thread import New_Thread


class MyMainWindow(QMainWindow,Ui_MainWindow):
    def __init__(self):
        super(MyMainWindow,self).__init__()
        self.setupUi(self)
        self.initUI()

    def initUI(self):
        self.select_file_button.clicked.connect(lambda:self.get_file())  # 获取文件路径
        self.show_data_button.clicked.connect(lambda:self.creat_table_show())  # 展示文件内容
        self.predict_data_button.clicked.connect(lambda:self.predict_data())  # 预测数据
        self.close_button.clicked.connect(lambda:self.close_window())  # 退出程序
        pass

    def get_file(self):
        m = QtWidgets.QFileDialog.getOpenFileName(None, 'open')  # 起始路径
        m = str(m).split(',')[0][1:]
        self.file_path_label.setText(m[1:-1])

    def creat_table_show(self):
        path_openfile_name =self.file_path_label.text()
        ###===========读取表格，转换表格，===========================================
        if len(path_openfile_name) > 0:
            input_table = pd.read_csv(path_openfile_name)
            input_table = input_table.iloc[:, 0:10]
            #print(input_table)
            input_table_rows = input_table.shape[0]
            input_table_colunms = input_table.shape[1]
            #print(input_table_rows)
            #print(input_table_colunms)
            input_table_header = input_table.columns.values.tolist()
            #print(input_table_header)

            ###===========读取表格，转换表格，============================================
            ###======================给tablewidget设置行列表头============================

            self.tableWidget.setColumnCount(input_table_colunms)
            self.tableWidget.setRowCount(input_table_rows)
            self.tableWidget.setHorizontalHeaderLabels(input_table_header)

            ###======================给tablewidget设置行列表头============================

            ###================遍历表格每个元素，同时添加到tablewidget中========================
            for i in range(input_table_rows):
                input_table_rows_values = input_table.iloc[[i]]
                # print(input_table_rows_values)
                input_table_rows_values_array = np.array(input_table_rows_values)
                input_table_rows_values_list = input_table_rows_values_array.tolist()[0]
                # print(input_table_rows_values_list)
                for j in range(input_table_colunms):
                    input_table_items_list = input_table_rows_values_list[j]
                    # print(input_table_items_list)
                    # print(type(input_table_items_list))

                    ###==============将遍历的元素添加到tablewidget中并显示=======================

                    input_table_items = str(input_table_items_list)
                    newItem = QTableWidgetItem(input_table_items)
                    newItem.setTextAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
                    newItem.setFlags(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled) # 设置内容不能修改
                    self.tableWidget.setItem(i, j, newItem)

                    ###================遍历表格每个元素，同时添加到tablewidget中========================
        else:
            self.centralWidget.show()

    def predict_data(self):

        path = self.file_path_label.text()
        pre, real = predict(path)
        self.realscore.setText(str(real))
        self.score.setText(str(pre))
        if 0 < pre <= 0.4:
            self.level.setText('差')
        elif 0.4 < pre <= 0.6:
            self.level.setText('中')
        elif 0.6 < pre <= 0.8:
            self.level.setText('良')
        else:
            self.level.setText('优')

        if 0 < real <= 0.4:
            self.reallevel.setText('差')
        elif 0.4 < real <= 0.6:
            self.reallevel.setText('中')
        elif 0.6 < real <= 0.8:
            self.reallevel.setText('良')
        else:
            self.reallevel.setText('优')

    def close_window(self):
        sys.exit()






if __name__ =='__main__':
    app = QApplication(sys.argv)
    myWin = MyMainWindow()
    myWin.show()
    sys.exit(app.exec_())