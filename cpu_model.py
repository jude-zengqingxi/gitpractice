
import torch

class Net(torch.nn.Module):# 继承 torch 的 Module
    def __init__(self, n_feature, n_output):
        super(Net, self).__init__()     # 继承 __init__ 功能
        # 定义每层用什么样的形式
        self.layer1 = torch.nn.Linear(n_feature,128)  #
        self.layer2 = torch.nn.Linear(128, 256)   #
        self.layer3 = torch.nn.Linear(256, n_output)
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


class Net2(torch.nn.Module):# 继承 torch 的 Module
    def __init__(self, n_feature, n_output):
        super(Net2, self).__init__()     # 继承 __init__ 功能
        # 定义每层用什么样的形式
        self.layer1 = torch.nn.Linear(n_feature,128)   #
        self.layer2 = torch.nn.Linear(128, 256)   #
        self.layer3 = torch.nn.Linear(256,512)
        self.layer4 = torch.nn.Linear(512,1024)
        self.layer5 = torch.nn.Linear(1024, n_output)
        #self.dropout = torch.nn.Dropout(p=0.8)

    def forward(self, x):   # 这同时也是 Module 中的 forward 功能
        x = self.layer1(x)
        #x = self.dropout(x)
        x = torch.relu(x)      #
        x = self.layer2(x)
        #x = self.dropout(x)
        x = torch.relu(x)      #
        x = self.layer3(x)
        x = torch.relu(x)      #
        x = self.layer4(x)
        x = torch.relu(x)      #
        x = self.layer5(x)
        return x
