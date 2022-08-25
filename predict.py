# 对一个文件夹中的所有文件进行预测评价
from data_deal import *
import torch
from model import Net
import os
import matplotlib.pyplot as plt

# 读取数据
def predict(file_path):
    test_features, test_labels = data_to_tensor(file_path)
    # 加载网络模型
    model = torch.load('data/model2.pth')
    net = Net(5, 1)
    net.load_state_dict(model)
    #print(net)

    # 测试集进行测试
    y_ = net(test_features)
    y_ = y_.cpu().detach().numpy()
    real_value = test_labels.cpu().detach().numpy()
    return np.float(y_.mean()), np.float(real_value.mean())


if __name__ == '__main__':
    file_name = os.listdir('data/test')
    real_value = []
    pre_value = []
    cross = []
    for i in range(len(file_name)):
        file_path = os.path.join('data', 'test', file_name[i])
        pre,real = predict(file_path)
        cross.append(pre-real)
        real_value.append(real)
        pre_value.append(pre)
    acc_test(pre_value,real_value)
    # 绘制误差分析图
    plt.scatter(range(1,len(cross)+1),cross)
    plt.show()