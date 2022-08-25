import pandas as pd
import numpy as np
from data_deal import *
import torch
from torch.utils.data import DataLoader,TensorDataset
import matplotlib.pyplot as plt
import time

# strat = time.clock()

#读取原始数据
train = pd.read_csv('data/train1.csv',header=0)
test = pd.read_csv('data/test1.csv',header=0)
print(train.shape) #(1500, 16)
print(test.shape)  #(600,16)


# 处理训练集和测试集

train_feature,train_label = deal_data3(train)
test_feature,test_label = deal_data3(test)

print(train_feature.shape,test_feature.shape)


# #自己的数据集，需要对原始数据进行处理
# #原数据 第一列是序号， 从第二列到倒数第二列都是 维度，最后一列是综合评分
# #对各维度的预处理(标准化)方式：数值型的转为[-1,1]之间 z-score 标准化，新数据=（原数据-均值）/标准差
# #非数值型中的  无序型进行独热编码(one-hot encoding)，有序型 自己定义其数值 转换为数值型  本数据集默认全部为无序型
# #空值：每一个特征的全局平均值来代替无效值
#
#
# #将训练集与测试集的特征数据合并在一起 统一进行处理
# #loc：通过行标签索引数据 iloc：通过行号索引行数据 ix：通过行标签或行号索引数据（基于loc和iloc的混合）
all_features = pd.concat((train_feature.loc[:,'feature1':'feature5'],test_feature.loc[:,'feature1':'feature5']))
all_labels = pd.concat((train_label.loc[:,'label'],test_label.loc[:,'label']))
#
print(all_features.shape)
#
# # #对特征值进行数据预处理
# # # 取出所有的数值型特征名称
numeric_feats = all_features.dtypes[all_features.dtypes != "object"].index
object_feats = all_features.dtypes[all_features.dtypes == "object"].index
#
# # # 将数值型特征进行 z-score 标准化
#all_features[numeric_feats] = all_features[numeric_feats].apply(lambda x: (x - x.mean()) / (x.std()))
#
#
# # #对无序型进行one-hot encoding
# # all_features = pd.get_dummies(all_features,prefix=object_feats, dummy_na=True)#
#
# # #空值：每一个特征的全局平均值来代替无效值 NA就是指空值
all_features = all_features.fillna(all_features.mean())
#
#
# # #对标签进行数据预处理
# # #对标签进行 z-score 标准化
# mean = all_labels.mean()
# std = all_labels.std()
# all_labels = (all_labels - mean)/std
#
num_train = train.shape[0]
train_features = all_features[:num_train].values.astype(np.float32)#(1600, 5)
test_features = all_features[num_train:].values.astype(np.float32)#(600, 5)
train_labels = all_labels[:num_train].values.astype(np.float32)
test_labels = all_labels[num_train:].values.astype(np.float32)

#
train_features = torch.from_numpy(train_features).cuda()
train_labels = torch.from_numpy(train_labels).unsqueeze(1).cuda()
test_features = torch.from_numpy(test_features).cuda()
test_labels = torch.from_numpy(test_labels).unsqueeze(1).cuda()
train_set = TensorDataset(train_features,train_labels)
test_set = TensorDataset(test_features,test_labels)


#定义迭代器
train_data = DataLoader(dataset=train_set,batch_size=64,shuffle=True)
test_data = DataLoader(dataset=test_set,batch_size=64,shuffle=False)

# def plot_png(data1,data2):
#     plt.plot(range(1,len(data1)+1),data1,data2)
#     plt.savefig('chazi.png')
#     plt.show

# #构建网络结构
class Net(torch.nn.Module):# 继承 torch 的 Module
    def __init__(self, n_feature, n_output):
        super(Net, self).__init__()     # 继承 __init__ 功能
        # 定义每层用什么样的形式
        self.layer1 = torch.nn.Linear(n_feature,64).cuda()   #
        self.layer2 = torch.nn.Linear(64, 128).cuda()   #
        self.layer3 = torch.nn.Linear(128, n_output).cuda()
        #self.dropout = torch.nn.Dropout(p=0.8)

    def forward(self, x):   # 这同时也是 Module 中的 forward 功能
        x = self.layer1(x)
        #x = self.dropout(x)
        x = torch.relu(x)      #
        x = self.layer2(x)
        #x = self.dropout(x)
        x = torch.relu(x)      #
        x = self.layer3(x)
        return x


net = Net(5,1)
#
#反向传播算法 SGD Adam等
optimizer = torch.optim.SGD(net.parameters(), lr=1e-4)
#均方损失函数
criterion = torch.nn.MSELoss()

#记录用于绘图
losses = [] #记录每次迭代后训练的loss
eval_losses = [] #验证的

for i in range(1000):
    train_loss = 0
    # train_acc = 0
    net.train() #网络设置为训练模式 暂时可加可不加
    for tdata,tlabel in train_data:
        #前向传播
        y_ = net(tdata)
        #记录单批次一次batch的loss
        loss = criterion(y_, tlabel)
        #反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #累计单批次误差
        train_loss = train_loss + loss.item()

    losses.append(train_loss / len(train_data))

    # 测试集进行测试
    eval_loss = 0
    net.eval()  # 可加可不加
    for edata, elabel in test_data:
        # 前向传播
        y_ = net(edata)
        # 记录单批次一次batch的loss，测试集就不需要反向传播更新网络了
        loss = criterion(y_, elabel)
        # 累计单批次误差
        eval_loss = eval_loss + loss.item()
    eval_losses.append(eval_loss / len(test_data))
    print('epoch: {}, trainloss: {}, evalloss: {}'.format(i, train_loss / len(train_data), eval_loss / len(test_data)))


torch.save(net.state_dict(),'data/model2.pth')
plt.plot(range(1,1001),losses,eval_losses)
plt.savefig('loss2.png')
#plt.plot(range(1,1001),eval_losses)
plt.show()





# #测试最终模型的精准度 算一下测试集的平均误差

#测试最终模型的精准度 算一下测试集的平均误差
y_ = net(test_features)
y_ = y_.cpu().detach().numpy()
real_value = test_labels.cpu().detach().numpy()
print(len(y_), len(real_value))
plt.plot(range(1, len(y_)+1), real_value)
plt.show()
print(y_.mean())

print(type(y_))
print(abs(y_ - real_value).mean().item())
