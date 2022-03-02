import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math
from matplotlib.font_manager import FontProperties

f = pd.read_csv('F:\lib.csv')

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

# 用1/6的数据做测试集，本数据集中2021年3月21日后的数据作为测试集
d_final = data.query('year==2021').query('month==3').query('day==21').index[0]
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
    LSTM(160, return_sequences=True),
    Dropout(0.2),
    LSTM(160),
    Dropout(0.2),
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
                    batch_size=64,
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
print('均方误差---', mse)
print('均方根误差---', rmse)
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
plt.savefig('F:\libCorrect.jpg')
