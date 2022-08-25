# 对单个文件进行预测，评价
from cpu_data_deal import *
import torch
from cpu_model import Net,Net2
import os
#import matplotlib.pyplot as plt

# 读取数据
def predict(file_path):
    test_features, test_labels = data_to_tensor(file_path)
    # 加载网络模型
    model = torch.load('data/model_cpu.pth')
    net = Net(5, 1)
    net.load_state_dict(model)
    #print(net)

    # 测试集进行测试
    y_ = net(test_features)
    y_ = y_.cpu().detach().numpy()
    print(y_)
    real_value = test_labels.cpu().detach().numpy()
    #print(abs(y_.mean()-real_value.mean()))
    return np.float(y_.mean()), np.float(real_value.mean())


if __name__ == '__main__':

    file_name = 'data/vsss2.csv'

    pre, real = predict(file_name)
    print(pre,real)