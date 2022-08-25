# -*- coding: utf-8 -*-
import time
from PyQt5.QtCore import QThread, pyqtSignal
from data_deal import *
import torch
from model import Net
#定义一个线程类
class New_Thread(QThread):
    #自定义信号声明
    # 使用自定义信号和UI主线程通讯，参数是发送信号时附带参数的数据类型，可以是str、int、list等
    finishSignal = pyqtSignal(float,float)

    # 带一个参数t
    def __init__(self, path,parent=None):
        super(New_Thread, self).__init__(parent)
        self.path= path
        test_features, test_labels = data_to_tensor(self.path)
        # 加载网络模型
        model = torch.load('data/model2.pth')
        net = Net(5, 1)
        net.load_state_dict(model)

        # 测试集进行测试
        y_ = net(test_features)
        y_ = y_.cpu().detach().numpy()
        real_value = test_labels.cpu().detach().numpy()
        # print(abs(y_.mean()-real_value.mean()))
        self.pre = np.float(y_.mean())
        self.real = np.float(real_value.mean())
    #show函数是子线程中的操作，线程启动后开始执
    def show(self):
            self.finishSignal.emit(self.pre,self.real)  # 注意这里与_signal = pyqtSignal(str)中的类型相同
