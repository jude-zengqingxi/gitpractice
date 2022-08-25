
import numpy as np
import random
import pandas as pd

from random import choice

def make_data(number,reorganization_time,fusion_time,multi_source,complexity):
    gap1 = 0
    gap2 = 0
    if multi_source == 0.4:
        gap1 = 360
        gap2 = 420
    elif multi_source == 0.6:
        gap1 = 240
        gap2 = 240
    elif multi_source == 0.8:
        gap1 = 120
        gap2 = 120
    else:
        gap1 = 60
        gap2 = 60
    reorganization = []  # 标准整编时间
    fusion = []  # 标准融合时间
    start_time = (np.random.randint(1, 120, number))
    start1_time = []  # 情报出现时间
    end1_time = []  # 情报出现时间
    start2_time = []  # 开始融合时间
    end2_time = []  # 融合结束时间
    put1_time = []  # 发布标准时间
    put2_time = []  # 情报发布时间
    multi_value = []  # 情报多源性
    complex_value = []  # 战场复杂度

    score1 = []  # 整编得分
    score2 = []  # 融合得分
    score3 = []  # 发布得分
    score4 = []  # 客观得分
    score5 = []  # 主观得分
    score = []  # 综合得分

    for i in range(number):
        end1 = int(np.random.randint(start_time[i]+1,start_time[i]+gap1,1))
        start2 = int(np.random.randint(end1,end1+30,1))
        end2 = int(np.random.randint(start2+1,start2+gap2,1))
        start1_time.append(start_time[i])
        end1_time.append(end1)
        start2_time.append(start2)
        end2_time.append(end2)

    put1 = max(end2_time)+180
    put2 = int(np.random.randint(max(end2_time),max(end2_time)+gap1))

    for i in range(number):

        reorganization.append(reorganization_time)
        fusion.append(fusion_time)
        put1_time.append(put1)
        put2_time.append(put2)
        multi_value.append(multi_source)
        complex_value.append(complexity)
    for i in range(number):
        temp1 = 0.3 if (reorganization[i]/(end1_time[i]-start1_time[i]))*0.3 > 0.3 else (reorganization[i]/(end1_time[i]-start1_time[i]))*0.3
        temp2 = 0.3 if (fusion[i]/(end2_time[i]-start2_time[i]))*0.3 > 0.3 else (fusion[i]/(end2_time[i]-start2_time[i]))*0.3
        if put1_time[i] > put2_time[i]:
            temp3 = 0.1
        elif put1_time[i] == put2_time[i]:
            temp3 = 0.1
        else:
            temp3 = 0.1 - ((put2_time[i]-put1_time[i])/put2_time[i])*0.1
        temp4 = temp1+temp2+temp3+(multi_value[i]*0.15+complex_value[i]*0.15)

        score1.append(temp1)
        score2.append(temp2)
        score3.append(temp3)
        score4.append((temp1+temp2+temp3))
        score5.append((multi_value[i]*0.15+complex_value[i]*0.15))
        score.append(temp4)

    data = [reorganization,start1_time,end1_time,fusion,start2_time,end2_time,put1_time,put2_time,multi_value,complex_value,score1,score2,score3,score4,score5,score]
    data = (np.array(data).transpose(1, 0)).tolist()

    column = ['标准整编时间', '情报出现时间', '情报上报时间','标准融合时间','开始融合时间', '融合结束时间', '发布标准时间','情报发布时间','情报多源性', '战场复杂性', '整编得分','融合得分',
              '发布得分', '客观得分', '主观得分','综合得分']  # 列表对应每列的列名

    test = pd.DataFrame(data=data,columns=column)

    return test

def make_test_data(num1,standard_time):
    for i in range(num1):
        value = choice([0.4,0.6,0.8,1])  # 随机生成情报复杂性数值
        #print(value)
        num2 = int(random.randint(10,30)) #随机生成的情报条数
        #print(num2)
        file_path = 'data/test/test'+str(i+1)+'.csv'
        temp = make_data(num2, standard_time,standard_time,value,value)
        temp.to_csv(file_path, encoding='utf-8-sig',index=False)
        print(file_path)

def make_train_data(num1,standard_time):
    train_path = 'data/train.csv'
    validation_path = 'data/validation.csv'
    for i in range(num1):
        number = random.randrange(1000, 4000)
        value = choice([0.4,0.6,0.8,1])  # 随机生成情报复杂性数值
        temp = make_data(number, standard_time,standard_time,value,value)
        if i < 1:
            temp.to_csv(train_path, encoding='utf-8-sig',index=False)
        else:
            temp.to_csv(train_path, encoding='utf-8-sig', mode='a',header=False, index=False)

    for i in range(num1):
        number = random.randrange(100, 400)
        value = choice([0.4,0.6,0.8,1])  # 随机生成情报复杂性数值
        temp = make_data(number, standard_time, standard_time, value, value)
        if i < 1:
            temp.to_csv(validation_path, encoding='utf-8-sig', index=False)
        else:
            temp.to_csv(validation_path, encoding='utf-8-sig', mode='a', header=False, index=False)

if __name__=='__main__':

    print('data_maker is on road!')
    make_test_data(100,30)
    #make_train_data(6,30)
