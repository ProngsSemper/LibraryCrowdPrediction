import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib.font_manager import FontProperties
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense, LSTM, Dropout

f = pd.read_csv('F:\lib3.csv')

# data 包含要操作的列
data = pd.DataFrame()

# 代码适应图书馆的数据
data['datetime'] = f['DATE']
data['human_traffic'] = f['INCOUNT']

data['year'] = data['datetime'].apply(lambda date: date.split('/')[0]).astype('int')
data['month'] = data['datetime'].apply(lambda date: date.split('/')[1]).astype('int')
data['day'] = data['datetime'].apply(lambda date: date.split('/')[2]).astype('int')
# 将人流量数据转换为float64类型
data['human_traffic'] = np.array(data['human_traffic']).astype(np.float64)

# 用1/6的数据做测试集，本数据集中2021年1月30日后的数据作为测试集
d_final = data.query('year==2021').query('month==1').query('day==30').index[0]
# 训练集
# : 表示所有，如果左右有数字则表示左闭右开[ ) 下面这个1：2 就是取数据集里的第一列，从这里开始下面的数据集都是只包含人流量了
train_set = data.iloc[:d_final, 1:2]
# 检测集
test_set = data.iloc[d_final:, 1:2]

# 数据预处理归一化 训练神经网络模型归一化，对预测结果进行反归一化便于与原始标签进行比较，衡量模型的性能 可以加快求解速度
# 数据归一化就是将数据中的每一个元素映射到一个较小的区间，这样，多维数据间的数字差距就会减小，消除量纲的影响。特别是在分析多维数据对标签的影响时更为重要。
sc = MinMaxScaler(feature_range=(0, 1))
# fit(): 求得训练集X的均值，方差，最大值，最小值,这些训练集X固有的属性。
# transform(): 在fit的基础上，进行标准化，降维，归一化等操作
# fit_transform是fit和transform的组合，既包括了训练又包含了转换。
# transform()和fit_transform()二者的功能都是对数据进行某种统一处理（比如标准化~N(0,1)，将数据缩放(映射)到某个固定区间，归一化，正则化等）
train_set_sc = sc.fit_transform(train_set)
test_set_sc = sc.transform(test_set)

# 按照time_step划分时间步长（使用前time_step个时间段预测下一个）
time_step = 5
x_train = []
y_train = []
x_test = []
y_test = []
# range() 从time_step 到 len - 1，下面则是time_step到d_final
for i in range(time_step, len(train_set_sc)):
    # 第一次循环append 0：time_step（0、1、2、...） 第二次append 1：time_step + 1（1、2、3、...）
    x_train.append(train_set_sc[i - time_step:i])
    # 每次循环从time_step开始每一轮添加一个
    y_train.append(train_set_sc[i:i + 1])
for i in range(time_step, len(test_set_sc)):
    x_test.append(test_set_sc[i - time_step:i])
    y_test.append(test_set_sc[i:i + 1])

# 随机化
np.random.seed(7)
np.random.shuffle(x_train)
np.random.seed(7)
np.random.shuffle(y_train)
tf.random.set_seed(7)

# 转为array格式
x_test, y_test = np.array(x_test), np.array(y_test)
x_train, y_train = np.array(x_train), np.array(y_train)
# 将x_train与x_test处理为新的格式 y.shape[0]代表行数，y.shape[1]代表列数。
x_train = np.reshape(x_train, (x_train.shape[0], time_step, 1))
x_test = np.reshape(x_test, (x_test.shape[0], time_step, 1))

# LSTM模型
model = tf.keras.Sequential([
    LSTM(90, return_sequences=True),
    Dropout(0.3),
    LSTM(110),
    Dropout(0.32),
    Dense(1)
])
model.compile(optimizer='adam',
              loss='mse')

# 训练模型
history = model.fit(x_train, y_train,
                    # 调试程序用
                    # epochs=5,
                    # 正式训练用
                    epochs=500,
                    # batch_size=32,
                    validation_data=(x_test, y_test))
# 模型预测
pre_flow = model.predict(x_test)
# 反归一化 转换为原始数据
pre_flow = sc.inverse_transform(pre_flow)
real_flow = sc.inverse_transform(y_test.reshape(y_test.shape[0], 1))

# 计算误差
mse = mean_squared_error(pre_flow, real_flow)
rmse = math.sqrt(mean_squared_error(pre_flow, real_flow))
mae = mean_absolute_error(pre_flow, real_flow)
# 均方误差是指参数估计值与参数真值之差平方的期望值，记为MSE。MSE是衡量“平均误差”的一种较方便的方法，MSE可以评价数据的变化程度，MSE的值越小，说明预测模型描述实验数据具有更好的精确度。
print('均方误差---', mse)
# RMSE是精确度的度量，用于比较特定数据集的不同模型的预测误差，而不是数据集之间的预测误差，因为它与比例相关。
# RMSE始终是非负的，值0（实际上几乎从未实现）表明数据非常合适。通常，较低的RMSE优于较高的RMSE。但是，跨不同类型数据的比较将无效，因为该度量取决于所使用数字的比例。
print('均方根误差---', rmse)
# 平均绝对误差的计算方法是，将各个样本的绝对误差汇总，然后根据数据点数量求出平均误差。通过将模型的所有绝对值加起来，可以避免因预测值比真实值过高或过低而抵销误差，并能获得用于评估模型的整体误差指标
print('平均绝对误差--', mae)

# 画出预测结果图
font_set = FontProperties(fname=r"C:\Windows\Fonts\simsun.ttc", size=15)  # 中文字体使用宋体，15号
plt.figure(figsize=(15, 10))
plt.plot(real_flow, label='Real_Flow', color='r', )
plt.plot(pre_flow, label='Pre_Flow')
plt.xlabel('测试序列', fontproperties=font_set)
plt.ylabel('人流量/人数', fontproperties=font_set)
plt.legend()
# 预测储存图片
plt.savefig('F:\libFull.jpg')
