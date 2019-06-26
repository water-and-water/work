#  数据预处理

from scipy import signal
from pandas import read_csv
import numpy as np
import matplotlib.pyplot as plt
from pandas import Series,DataFrame
import pandas as pd
filename=r'C:\Users\CYY\Desktop\data_12h\samples.csv'
data1=read_csv(filename)
peek=data1.head(5)
print(peek)
frame=DataFrame(data1)
# print(frame)
data=data1.iloc[:,6]            #从第0行起，到最后一个数据；第7列
print(len(data))
data=data.astype(float)           #类型转换
b,a=signal.butter(6,[0.005,0.16],'bandpass')    #巴特沃斯滤波器（阶数，截止频率，滤波器类型）
filtedData=signal.filtfilt(b,a,data)
#fig=plt.figure()
#plt.plot(data,color='g')
#fig=plt.figure()
#plt.plot(filtedData,'k')     #滤波后信号
#plt.show()

###  用滤波后的数据替换原始数据中的pleth值
filtedData=list(filtedData)

deData = []        #此时dedata是空的
for item in filtedData:
    item=round(item,4)
    deData.append(item)

print(len(deData))
print(len(frame))
frame['PLETH(mV)']=deData   #数据长度不一致会报错
#print(frame)
frame.to_csv('processData.csv',index=False,header=True)
