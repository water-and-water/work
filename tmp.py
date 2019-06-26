#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd
from pandas import Series,DataFrame

#------------------------------------------------------------------
#读取csv数据，保存到data里
#data是numpy.array格式的数组
csvFile = open('sample08.csv', 'r')
reader = csv.reader(csvFile)
data = []        #此时data是空的
for item in reader:
    data.append(item)
csvFile.close()
data = np.array(data)
data = data[1: ]    #从数据的第2行取值到最后

#------------------------------------------------------------------
#将头和尾两个不完整的周期去掉
#index0是头部不完整周期最后一个点的位置
#index2是尾部不完整周期第一个点的位置
elapse = np.array(list(map(float, data[:, 0])))
abp = np.array(list(map(float, data[:, 4])))
#for index, ele in enumerate(data[:,6]):
  #  try:
   #     float(ele)
   # except:
   #     print(ele)
   #     print(index)
#pass
pleth = np.array(list(map(float, data[:, 6])))

tmp0 = pleth[0: 300]
min0 = np.min(tmp0)
index0 = np.argmin(tmp0)  # argmin()返回最小值在数组中的位置

elapse = elapse[index0: ]
abp = abp[index0: ]
pleth = pleth[index0: ]

tmp1 = pleth[299700: ]
min1 = np.min(tmp1)
index1 = np.argmin(tmp1)
index2 = len(tmp1) - index1

#elapse，abp，pleth为最终原始数据，且均是numpy.array格式的数组
elapse = elapse[0: -index2 + 1]
abp = abp[0: -index2 + 1]
pleth = pleth[0: -index2 + 1]

#------------------------------------------------------------------
#定义两个函数分别搜寻pleth极大值和极小值的位置index和对应的数值ans
#搜寻极大值的函数窗长为40，搜寻极小值的函数窗长为24
#考虑到原始数据没有经过滤波等操作，故并不是完全意义上的平滑函数，故在写极值搜索函数时，直接试出最合适的窗长 -。-
#一般情况下都是将函数作平滑处理，然后自适应搜索 -。-
#index和ans都是numpy.array格式的数组
#这样搜寻出的极小值数目应该比极大值数目少一，并且不包含一头一尾两个极小值，后面会添加上
def localmin(x):
    ans = []
    index = []
    for i in range(20, len(x) - 20):
        if x[i] < x[i - 1] and x[i] < x[i - 2] and x[i] < x[i - 3] and x[i] < x[i - 4] and x[i] < x[i - 5]:
            if x[i] < x[i - 6] and x[i] < x[i - 7] and x[i] < x[i - 8] and x[i] < x[i - 9] and x[i] < x[i - 10]:
                if x[i] < x[i - 11] and x[i] < x[i - 12] and x[i] < x[i - 13] and x[i] < x[i - 14] and x[i] < x[i - 15]:
                    if x[i] < x[i - 16] and x[i] < x[i - 17] and x[i] < x[i - 18] and x[i] < x[i - 19] and x[i] < x[i - 20]:
                        if x[i] <= x[i + 1] and x[i] <= x[i + 2] and x[i] <= x[i + 3] and x[i] <= x[i + 4] and x[i] <= x[i + 5]:
                            if x[i] <= x[i + 6] and x[i] <= x[i + 7] and x[i] <= x[i + 8] and x[i] <= x[i + 9] and x[i] <= x[i + 10]:
                                if x[i] <= x[i + 11] and x[i] <= x[i + 12] and x[i] <= x[i + 13] and x[i] <= x[i + 14] and x[i] <= x[i + 15]:
                                    if x[i] <= x[i + 16] and x[i] <= x[i + 17] and x[i] <= x[i + 18] and x[i] <= x[i + 19] and x[i] <= x[i + 20]:
                                        ans.append(x[i])
                                        index.append(i)
    return ans, index

def localmax(x):
    ans = []
    index = []
    for i in range(12, len(x) - 12):
        if x[i] > x[i - 1] and x[i] > x[i - 2] and x[i] > x[i - 3] and x[i] > x[i - 4] and x[i] > x[i - 5]:
            if x[i] > x[i - 6] and x[i] > x[i - 7] and x[i] > x[i - 8] and x[i] > x[i - 9] and x[i] > x[i - 10]:
                if x[i] > x[i - 11] and x[i] > x[i - 12]:
                    if x[i] >= x[i + 1] and x[i] >= x[i + 2] and x[i] >= x[i + 3] and x[i] >= x[i + 4] and x[i] >= x[i + 5]:
                        if x[i] >= x[i + 6] and x[i] >= x[i + 7] and x[i] >= x[i + 8] and x[i] >= x[i + 9] and x[i] >= x[i + 10]:
                            if x[i] >= x[i + 11] and x[i] >= x[i + 12]:
                                ans.append(x[i])
                                index.append(i)
    return ans, index

localmin_value, localmin_index = localmin(pleth)    #调用极小值函数localmin
localmax_value, localmax_index = localmax(pleth)    #调用极大值函数localmax

#把第一个极小值和最后一个极小值添加进去
localmin_index = np.append(np.array(0), localmin_index)
localmin_index = np.append(localmin_index, np.array(len(pleth) - 1))
localmin_value = np.append(pleth[0], localmin_value)
localmin_value = np.append(localmin_value, pleth[-1])

#------------------------------------------------------------------
tmpindex = np.hstack((localmin_index, localmax_index))   #hstack函数把数组堆起来，返回numpy的数组
tmpindex = sorted(tmpindex)  #sorted()对可迭代的对象进行升序（默认）排列，返回重新排列的列表

tmparray = np.array([])
for i in range(len(tmpindex)):
    tmparray = np.append(tmparray, pleth[tmpindex[i]])

halfAB_value = np.array([])
halfAB_elapse_value = np.array([])
features = np.zeros(shape=[int((len(tmpindex) - 1) / 4), 1, 14])   #14=9个特征参数+5个血压值
#print(features)   #全是0

###  一次性把所有的极值点都找出来了，然后利用sorted函数排序

for i in range(int((len(tmpindex) - 1) / 4)):
    #h1,h2,h3,h4,t1,t2,t3,t4,t均是定义的
    h1 = pleth[[tmpindex[i * 4 + 1]]] - pleth[[tmpindex[i * 4]]]
    h2 = pleth[[tmpindex[i * 4 + 2]]] - pleth[[tmpindex[i * 4]]]
    h3 = pleth[[tmpindex[i * 4 + 3]]] - pleth[[tmpindex[i * 4]]]
    h4 = pleth[[tmpindex[i * 4 + 3]]] - pleth[[tmpindex[i * 4 + 2]]]
    t1 = elapse[[tmpindex[i * 4 + 1]]] - elapse[[tmpindex[i * 4]]]
    t2 = elapse[[tmpindex[i * 4 + 2]]] - elapse[[tmpindex[i * 4]]]
    t3 = elapse[[tmpindex[i * 4 + 3]]] - elapse[[tmpindex[i * 4]]]
    t = elapse[[tmpindex[i * 4 + 4]]] - elapse[[tmpindex[i * 4]]]
    #h=0.5h1
    h = h1 / 2 + pleth[[tmpindex[i * 4]]]
    plethAB = pleth[tmpindex[i * 4]: tmpindex[i * 4 + 1]]
    elapseAB = elapse[tmpindex[i * 4]: tmpindex[i * 4 + 1]]
    error = np.array([])
    for j in range(len(plethAB)):
        error = np.append(error, np.power(plethAB[j] - h, 2))
    errorindex = np.argmin(error)
    halfAB_value = np.append(halfAB_value, plethAB[errorindex])
    halfAB_elapse_value = np.append(halfAB_elapse_value, elapseAB[errorindex])
    t4 = elapseAB[errorindex] - elapse[[tmpindex[i * 4]]]
    
    ###标签label
    abp_tmp = np.concatenate((abp[[tmpindex[i * 4]]],  abp[[tmpindex[i * 4 + 1]]], abp[[tmpindex[i * 4 + 2]]], \
                        abp[[tmpindex[i * 4 + 3]]], abp[[tmpindex[i * 4 + 4]]]), axis=0)

    features[i] = np.concatenate((h1, h2, h3, h4, t1, t2, t3, t4, t, abp_tmp), axis=0)    
    #数组拼接函数concatenate,axis=0为拼接方向，即对0轴的数组进行纵向拼接
    
print(features.shape)
features=np.reshape(features,[1015,14])
print(features.shape)
fea_data=DataFrame(features,columns=['h1','h2','h3','h4','t1','t2','t3','t4','t','abp1','abp2','abp3','abp4','abp5'])
fea_data.to_csv('Fe08.csv',index=False,header=True)    ###输出特征文件FeaData

#------------------------------------------------------------------
#在交互模式下输入featuresplot()可以看提取出来的极值图
localmin_elapse_value = np.array([])
localmax_elapse_value = np.array([])

for p in range(len(localmin_index)):
    localmin_elapse_value = np.append(localmin_elapse_value, elapse[localmin_index[p]])

for q in range(len(localmax_index)):
    localmax_elapse_value = np.append(localmax_elapse_value, elapse[localmax_index[q]])

def featuresplot():
    plt.figure(figsize=(26, 8))   #图片尺寸
    plt.plot(elapse, pleth)
    plt.plot(localmin_elapse_value, localmin_value, 'g^', localmax_elapse_value, localmax_value, 'r^', halfAB_elapse_value, halfAB_value, 'b^')
    # plt.show()
    plt.savefig('123.png', dpi=600)   #图片保存路径和像素值
