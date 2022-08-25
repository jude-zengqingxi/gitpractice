import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
def deal_data1(train):
    # 训练集
    # # 提取特征属性
    feature1 = train.iloc[:, 0][1]/(train.iloc[:, 2].values-train.iloc[:, 1].values)
    feature2 = train.iloc[:, 3].values/(train.iloc[:, 5].values - train.iloc[:, 3].values)
    #feature3 = abs(train.iloc[:, 6].values-train.iloc[:,7].values)/max(train.iloc[:,6][0],train.iloc[:,7][0]) if train.iloc[:,6][0]!=train.iloc[:,7][0] else 1
    feature3 = (train.iloc[:, 6].values-train.iloc[:,7].values)
    feature4 = train.iloc[:, 8].values
    feature5 = train.iloc[:, 9].values

    for i in range(len(feature1)):
        if feature1[i] > 1:
            feature1[i] = 1
        if feature2[i] > 1:
            feature2[i] = 1
        if feature3[i] >= 0:
            feature3[i] = 1
        else:
            feature3[i] = 1 - ((train.iloc[:, 7][0]-train.iloc[:,6][0])/train.iloc[:, 7][0])
    label = train.iloc[:, 15].values

    # 合成新的训练数据
    datalist = []
    datalist.append(feature1)
    datalist.append(feature2)
    datalist.append(feature3)
    datalist.append(feature4)
    datalist.append(feature5)
    datalist = list(np.array(datalist).transpose(1, 0))
    #datalist = list(datalist_temp.reshape(datalist_temp.shape[1], datalist_temp.shape[0]))
    #label = label.reshape(label.shape[0],-1)
    #print(label.shape)
    label = list(np.array(label).reshape(label.shape[0], -1))

    all_features = pd.DataFrame(datalist,columns=['feature1','feature2','feature3','feature4','feature5'])
    all_labels = pd.DataFrame(label,columns=['label'])
    return all_features,all_labels

def deal_data2(train):
    # 训练集
    # # 提取特征属性
    feature1 = train.iloc[:, 0].values
    feature2 = train.iloc[:,1].values
    feature3 = train.iloc[:,2].values
    feature4 = train.iloc[:, 3].values
    feature5 = train.iloc[:,4].values
    feature6 = train.iloc[:,5].values
    feature7= train.iloc[:, 6].values
    feature8 = train.iloc[:,7].values
    feature9 = train.iloc[:,8].values
    feature10 = train.iloc[:,9].values
    feature11 = train.iloc[:,10].values
    label = train.iloc[:,15].values

    # 合成新的训练数据
    datalist = []
    datalist.append(feature1)
    datalist.append(feature2)
    datalist.append(feature3)
    datalist.append(feature4)
    datalist.append(feature5)
    datalist.append(feature6)
    datalist.append(feature7)
    datalist.append(feature8)
    datalist.append(feature9)
    datalist.append(feature10)


    datalist=list(np.array(datalist).transpose(1,0))
    label = list(np.array(label).reshape(label.shape[0],-1))
    all_features = pd.DataFrame(datalist,columns=['feature1','feature2','feature3','feature4','feature5','feature6','feature7','feature8','feature9','feature10'])
    all_labels = pd.DataFrame(label,columns=['label'])
    return all_features,all_labels

def data_to_tensor(file_name):
    data = pd.read_csv(file_name, header=0)
    #print(data.shape)

    #print(data.shape)  # (600,16)

    # 处理训练集和测试集

    data_feature, data_label = deal_data1(data)

    # #自己的数据集，需要对原始数据进行处理
    # #原数据 第一列是序号， 从第二列到倒数第二列都是 维度，最后一列是综合评分
    # #对各维度的预处理(标准化)方式：数值型的转为[-1,1]之间 z-score 标准化，新数据=（原数据-均值）/标准差
    # #非数值型中的  无序型进行独热编码(one-hot encoding)，有序型 自己定义其数值 转换为数值型  本数据集默认全部为无序型
    # #空值：每一个特征的全局平均值来代替无效值
    #
    #
    # #将训练集与测试集的特征数据合并在一起 统一进行处理
    # #loc：通过行标签索引数据 iloc：通过行号索引行数据 ix：通过行标签或行号索引数据（基于loc和iloc的混合）
    all_features = data_feature
    all_labels = data_label
    #
    #print(all_features.shape)
    #
    # # #对特征值进行数据预处理
    # # # 取出所有的数值型特征名称
    numeric_feats = all_features.dtypes[all_features.dtypes != "object"].index
    object_feats = all_features.dtypes[all_features.dtypes == "object"].index
    #
    # # # 将数值型特征进行 z-score 标准化
    # all_features[numeric_feats] = all_features[numeric_feats].apply(lambda x: (x - x.mean()) / (x.std()))
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
    # #

    data_features = all_features[0:].values.astype(np.float32)  # (600, 5)

    data_labels = all_labels[0:].values.astype(np.float32)

    #

    test_features = torch.from_numpy(data_features)
    test_labels = torch.from_numpy(data_labels)

    return test_features, test_labels

def data_to_tensor2(train_path,test_path):
    # 读取原始数据
    train = pd.read_csv(train_path, header=0)
    test = pd.read_csv(test_path, header=0)
    print(train.shape)  # (1500, 16)
    print(test.shape)  # (600,16)

    # 处理训练集和测试集

    train_feature, train_label = deal_data1(train)
    test_feature, test_label = deal_data1(test)

    print(train_feature.shape, test_feature.shape)

    # #自己的数据集，需要对原始数据进行处理
    # #原数据 第一列是序号， 从第二列到倒数第二列都是 维度，最后一列是综合评分
    # #对各维度的预处理(标准化)方式：数值型的转为[-1,1]之间 z-score 标准化，新数据=（原数据-均值）/标准差
    # #非数值型中的  无序型进行独热编码(one-hot encoding)，有序型 自己定义其数值 转换为数值型  本数据集默认全部为无序型
    # #空值：每一个特征的全局平均值来代替无效值
    #
    #
    # #将训练集与测试集的特征数据合并在一起 统一进行处理
    # #loc：通过行标签索引数据 iloc：通过行号索引行数据 ix：通过行标签或行号索引数据（基于loc和iloc的混合）
    all_features = pd.concat((train_feature.loc[:, 'feature1':'feature5'], test_feature.loc[:, 'feature1':'feature5']))
    all_labels = pd.concat((train_label.loc[:, 'label'], test_label.loc[:, 'label']))
    #
    print(all_features.shape)
    #
    # # #对特征值进行数据预处理
    # # # 取出所有的数值型特征名称
    numeric_feats = all_features.dtypes[all_features.dtypes != "object"].index
    object_feats = all_features.dtypes[all_features.dtypes == "object"].index
    #
    # # # 将数值型特征进行 z-score 标准化
    # all_features[numeric_feats] = all_features[numeric_feats].apply(lambda x: (x - x.mean()) / (x.std()))
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
    train_features = all_features[:num_train].values.astype(np.float32)  # (1600, 5)
    test_features = all_features[num_train:].values.astype(np.float32)  # (600, 5)
    train_labels = all_labels[:num_train].values.astype(np.float32)
    test_labels = all_labels[num_train:].values.astype(np.float32)

    #
    train_features = torch.from_numpy(train_features)
    train_labels = torch.from_numpy(train_labels).unsqueeze(1)
    test_features = torch.from_numpy(test_features)
    test_labels = torch.from_numpy(test_labels).unsqueeze(1)
    train_set = TensorDataset(train_features, train_labels)
    test_set = TensorDataset(test_features, test_labels)

    # 定义迭代器
    train_data = DataLoader(dataset=train_set, batch_size=64, shuffle=True)
    test_data = DataLoader(dataset=test_set, batch_size=64, shuffle=False)

    return train_data,test_data

def data_to_guiyi1(file_name):
    data = pd.read_csv(file_name, header=0)

    #print(data.shape)  # (600,16)

    # 处理训练集和测试集

    data_feature, data_label = deal_data1(data)

    # #自己的数据集，需要对原始数据进行处理
    # #原数据 第一列是序号， 从第二列到倒数第二列都是 维度，最后一列是综合评分
    # #对各维度的预处理(标准化)方式：数值型的转为[-1,1]之间 z-score 标准化，新数据=（原数据-均值）/标准差
    # #非数值型中的  无序型进行独热编码(one-hot encoding)，有序型 自己定义其数值 转换为数值型  本数据集默认全部为无序型
    # #空值：每一个特征的全局平均值来代替无效值
    #
    #
    # #将训练集与测试集的特征数据合并在一起 统一进行处理
    # #loc：通过行标签索引数据 iloc：通过行号索引行数据 ix：通过行标签或行号索引数据（基于loc和iloc的混合）
    all_features = data_feature
    all_labels = data_label
    #
    #print(all_features.shape)
    #
    # # #对特征值进行数据预处理
    # # # 取出所有的数值型特征名称
    numeric_feats = all_features.dtypes[all_features.dtypes != "object"].index
    object_feats = all_features.dtypes[all_features.dtypes == "object"].index
    #
    # # # 将数值型特征进行 z-score 标准化
    # all_features[numeric_feats] = all_features[numeric_feats].apply(lambda x: (x - x.mean()) / (x.std()))
    all_features[numeric_feats] = all_features[numeric_feats].apply(lambda x: (x - min(x)) / (max(x) - min(x)))
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
    all_labels = (all_labels - min(all_labels)) / (max(all_labels) - min(all_labels))
    # #

    data_features = all_features[0:].values.astype(np.float32)  # (600, 5)

    data_labels = all_labels[0:].values.astype(np.float32)

    #

    test_features = torch.from_numpy(data_features).cuda()
    test_labels = torch.from_numpy(data_labels).cuda()

    return test_features, test_labels

def data_to_guiyi2(train_path,test_path):
    # 读取原始数据
    train = pd.read_csv(train_path, header=0)
    test = pd.read_csv(test_path, header=0)
    print(train.shape)  # (1500, 16)
    print(test.shape)  # (600,16)

    # 处理训练集和测试集

    train_feature, train_label = deal_data1(train)
    test_feature, test_label = deal_data1(test)

    print(train_feature.shape, test_feature.shape)

    # #自己的数据集，需要对原始数据进行处理
    # #原数据 第一列是序号， 从第二列到倒数第二列都是 维度，最后一列是综合评分
    # #对各维度的预处理(标准化)方式：数值型的转为[-1,1]之间 z-score 标准化，新数据=（原数据-均值）/标准差
    # #非数值型中的  无序型进行独热编码(one-hot encoding)，有序型 自己定义其数值 转换为数值型  本数据集默认全部为无序型
    # #空值：每一个特征的全局平均值来代替无效值
    #
    #
    # #将训练集与测试集的特征数据合并在一起 统一进行处理
    # #loc：通过行标签索引数据 iloc：通过行号索引行数据 ix：通过行标签或行号索引数据（基于loc和iloc的混合）
    all_features = pd.concat((train_feature.loc[:, 'feature1':'feature5'], test_feature.loc[:, 'feature1':'feature5']))
    all_labels = pd.concat((train_label.loc[:, 'label'], test_label.loc[:, 'label']))
    #
    #print(all_features.shape)
    #
    # # #对特征值进行数据预处理
    # # # 取出所有的数值型特征名称
    numeric_feats = all_features.dtypes[all_features.dtypes != "object"].index
    object_feats = all_features.dtypes[all_features.dtypes == "object"].index
    #
    # # # 将数值型特征进行 z-score 标准化
    all_features[numeric_feats] = all_features[numeric_feats].apply(lambda x: (x - min(x)) / (max(x)-min(x)))
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
    #all_labels = (all_labels - mean)/std
    all_labels = (all_labels - min(all_labels))/(max(all_labels)-min(all_labels))
    #
    num_train = train.shape[0]
    train_features = all_features[:num_train].values.astype(np.float32)  # (1600, 5)
    test_features = all_features[num_train:].values.astype(np.float32)  # (600, 5)
    train_labels = all_labels[:num_train].values.astype(np.float32)
    test_labels = all_labels[num_train:].values.astype(np.float32)

    #
    train_features = torch.from_numpy(train_features).cuda()
    train_labels = torch.from_numpy(train_labels).unsqueeze(1).cuda()
    test_features = torch.from_numpy(test_features).cuda()
    test_labels = torch.from_numpy(test_labels).unsqueeze(1).cuda()
    train_set = TensorDataset(train_features, train_labels)
    test_set = TensorDataset(test_features, test_labels)

    # 定义迭代器
    train_data = DataLoader(dataset=train_set, batch_size=64, shuffle=True)
    test_data = DataLoader(dataset=test_set, batch_size=64, shuffle=False)

    return train_data,test_data


def acc_test(pre_value,real_value):
    for i in range(len(pre_value)):
        if 0 < pre_value[i] <= 0.4:
            pre_value[i] = 1
        elif 0.4 < pre_value[i] <= 0.6:
            pre_value[i] = 2
        elif 0.6 < pre_value[i] <= 0.8:
            pre_value[i] = 3
        else:
            pre_value[i]=4

    for i in range(len(real_value)):
        if 0 < real_value[i] <= 0.4:
            real_value[i] = 1
        elif 0.4 < real_value[i] <= 0.6:
            real_value[i] = 2
        elif 0.6 < real_value[i] <= 0.8:
            real_value[i] = 3
        else:
            real_value[i] = 4
    count = 0
    print(pre_value,real_value)
    for i in range(len(real_value)):
        if pre_value[i] == real_value[i]:
            count += 1
    print(count/len(real_value))


#
# if __name__ == '__main__':
#     train = pd.read_csv('data/train1.csv', header=0)
#     test = pd.read_csv('data/test1.csv', header=0)
#     features,labels = deal_data3(train)
#     print(features)