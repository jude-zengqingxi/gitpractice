
# 数据集不进行标准化处理

from data_deal import *
import torch
import matplotlib.pyplot as plt
from model import Net,Net2


#读取原始数据
train_path = 'data/train.csv'
validate_path = 'data/validation.csv'
train_data, validate_data = data_to_guiyi2(train_path,validate_path)


# #构建网络结构
net = Net(5, 1)

#反向传播算法 SGD Adam等
optimizer = torch.optim.SGD(net.parameters(), lr=1e-4)
#均方损失函数
criterion = torch.nn.MSELoss()

#记录用于绘图
losses = []  # 记录每次迭代后训练的loss
eval_losses = []  # 验证的

for i in range(2000):
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

    # 验证集进行测试
    eval_loss = 0
    net.eval()  # 可加可不加
    for edata, elabel in validate_data:
        # 前向传播
        y_ = net(edata)
        # 记录单批次一次batch的loss，测试集就不需要反向传播更新网络了
        loss = criterion(y_, elabel)
        # 累计单批次误差
        eval_loss = eval_loss + loss.item()
    eval_losses.append(eval_loss / len(validate_data))
    print('epoch: {}, trainloss: {}, evalloss: {}'.format(i, train_loss / len(train_data), eval_loss / len(validate_data)))

torch.save(net.state_dict(),'data/model3.pth')
plt.plot(range(1,2001),losses,eval_losses)
plt.savefig('loss3.png')
plt.show()





# # #测试最终模型的精准度 算一下测试集的平均误差
#
# #测试最终模型的精准度 算一下测试集的平均误差
# y_ = net(test_features)
# y_ = y_.cpu().detach().numpy()
# real_value = test_labels.cpu().detach().numpy()
# print(len(y_), len(real_value))
# plt.plot(range(1, len(y_)+1), real_value)
# plt.show()
# print(y_.mean())

# print(type(y_))
# print(abs(y_ - real_value).mean().item())
