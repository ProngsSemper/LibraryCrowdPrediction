# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


# def print_hi(name):
#     # Use a breakpoint in the code line below to debug your script.
#     print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
# if __name__ == '__main__':
#     print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, LSTM, Dropout, Flatten
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math
from matplotlib.font_manager import FontProperties  # 画图时可以使用中文

# f = pd.read_csv('F:\AE86.csv')
f = pd.read_csv('F:\lib.csv')


# 从新设置列标
def set_columns():
    columns = []

    for i in f.loc[2]:
        # append()接受一个对象参数，把对象添加到列表的尾部 .strip（）用来删除空白符
        # columns.append(i.strip())
        columns.append(i)
    return columns


# f.columns = set_columns()
# drop函数默认删除行，列需要加axis = 1
# drop方法有一个可选参数inplace，表明可对原数组作出修改并返回一个新数组。不管参数默认为False还是设置为True，原数组的内存值是不会改变的，区别在于原数组的内容是否直接被修改。
f.drop([0, 1, 2], inplace=True)

# data 包含要操作的列
data = pd.DataFrame()
# data['datetime'] = f['Local Date'] + ' ' + f['Local Time']
# data['total_flow'] = f['Total Carriageway Flow']
# # data['speed'] = f['Speed Value']
# data['datetime'] = pd.to_datetime(data['datetime'])
#
# data['month'] = data['datetime'].apply(lambda date: date.month)
# data['day'] = data['datetime'].apply(lambda date: date.day)
# data['hour'] = data['datetime'].apply(lambda date: date.hour)
# data['minute'] = data['datetime'].apply(lambda date: date.minute)

# 代码适应图书馆的数据
data['datetime'] = f['DATE']
data['human_traffic'] = f['INCOUNT']

data['year'] = data['datetime'].apply(lambda date: date.split('/')[0]).astype('int')
data['month'] = data['datetime'].apply(lambda date: date.split('/')[1]).astype('int')
data['day'] = data['datetime'].apply(lambda date: date.split('/')[2]).astype('int')

data['human_traffic'] = np.array(data['human_traffic']).astype(np.float64)

# 数据转格式
# data['total_flow'] = np.array(data['total_flow']).astype(np.float64)
# 一月第25天第一个时间的索引值
d25 = data.query('day==25').index[0]
# 训练集  2211个数据，2018年一月前三周
train_set = data.iloc[:d25, 1:2]
# 检测集  669个数据，2018年最后一周
test_set = data.iloc[d25:, 1:2]
# 归一化 训练神经网络模型归一化，对预测结果进行反归一化便于与原始标签进行比较，衡量模型的性能
sc = MinMaxScaler(feature_range=(0, 1))
train_set_sc = sc.fit_transform(train_set)
test_set_sc = sc.transform(test_set)

# 按照time_step划分时间步长
time_step = 5
x_train = []
y_train = []
x_test = []
y_test = []
for i in range(time_step, len(train_set_sc)):
    x_train.append(train_set_sc[i - time_step:i])
    y_train.append(train_set_sc[i:i + 1])
for i in range(time_step, len(test_set_sc)):
    x_test.append(test_set_sc[i - time_step:i])
    y_test.append(test_set_sc[i:i + 1])
x_test, y_test = np.array(x_test), np.array(y_test)

# 随机化，这部分可以不要
np.random.seed(7)
np.random.shuffle(x_train)
np.random.seed(7)
np.random.shuffle(y_train)
tf.random.set_seed(7)

# 转为array格式
x_train, y_train = np.array(x_train), np.array(y_train)
x_test, y_test = np.array(x_test), np.array(y_test)
x_train = np.reshape(x_train, (x_train.shape[0], time_step, 1))
x_test = np.reshape(x_test, (x_test.shape[0], time_step, 1))

# LSTM模型
model = tf.keras.Sequential([
    LSTM(80, return_sequences=True),
    Dropout(0.2),
    LSTM(80),
    Dropout(0.2),
    Dense(1)
])
model.compile(optimizer='adam',
              loss='mse', )

# 训练模型， 其中epochs，batch_size 可以自己更改
history = model.fit(x_train, y_train,
                    epochs=5,
                    validation_data=(x_test, y_test))
# 模型预测
pre_flow = model.predict(x_test)
# 反归一化
pre_flow = sc.inverse_transform(pre_flow)
real_flow = sc.inverse_transform(y_test.reshape(y_test.shape[0], 1))

# 计算误差
mse = mean_squared_error(pre_flow, real_flow)
rmse = math.sqrt(mean_squared_error(pre_flow, real_flow))
mae = mean_absolute_error(pre_flow, real_flow)
print('均方误差---', mse)
print('均方根误差---', rmse)
print('平均绝对误差--', mae)

# 画出预测结果图
font_set = FontProperties(fname=r"C:\Windows\Fonts\simsun.ttc", size=15)  # 中文字体使用宋体，15号
plt.figure(figsize=(15, 10))
plt.plot(real_flow, label='Real_Flow', color='r', )
plt.plot(pre_flow, label='Pre_Flow')
plt.xlabel('测试序列', fontproperties=font_set)
# plt.ylabel('交通流量/辆', fontproperties=font_set)
plt.ylabel('人流量/人数', fontproperties=font_set)
plt.legend()
# 预测储存图片
plt.savefig('F:\345.jpg')
