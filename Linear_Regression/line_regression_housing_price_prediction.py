"""
@author: Colyn
@date: 2022-11-05
@desc: Multiple Linear Regression Homework 基于多元线性回归预测波斯顿房价问题
"""
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

# 超参数设定
config = {
    'alpha': 0.01,  # 学习率，即步长
    'epoch': 1000,  # 训练轮数
    'l': 50  # 正则化参数
}

# 导入文件
source_path = 'housing.data'
csv_path = 'housing.csv'
source_data = np.fromfile(source_path, sep=' ')
# 每条数据包括14项，其中前面13项是影响因素，第14项是相应的房屋价格中位数
label = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM',
         'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
data_size = len(label)
# 将原始数据进行Reshape，变成[N, 14]这样的形状
source_data = source_data.reshape([source_data.shape[0] // data_size, data_size])
np.savetxt(csv_path, source_data, delimiter=',')
data = pd.read_csv(csv_path, names=label)
# 特征缩放 （x-平均值）/标准差
data = (data - data.mean()) / data.std()
# 查看特征缩放后的数据
data.head(10)
print(data)

# 变量初始化
# 最后一列为y，其余为x
cols = data.shape[1]  # 列数 shape[0]行数 [1]列数
X = data.iloc[:, 0:cols - 1]  # 取前cols-1列，即输入向量
y = data.iloc[:, cols - 1:cols]  # 取最后一列，即目标变量
X.head(10)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 将数据转换成numpy矩阵
X_train = np.matrix(X_train.values)
y_train = np.matrix(y_train.values)
X_test = np.matrix(X_test.values)
y_test = np.matrix(y_test.values)
# 初始化theta矩阵
theta = np.matrix([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
# X_train.shape, X_test.shape, y_train.shape, y_test.shape

# 添加偏置列，值为1，axis = 1 添加列
X_train = np.insert(X_train, 0, 1, axis=1)
X_test = np.insert(X_test, 0, 1, axis=1)


# X_train.shape, X_test.shape, y_train.shape, y_test.shape

# 代价函数
def loss_func(X, y, theta):
    inner = np.power(X * theta.T - y, 2)
    return np.sum(inner) / (2 * len(X))


# 正则化代价函数
def reg_loss(X, y, theta, l):
    reg = (l / (2 * len(X))) * (np.power(theta, 2).sum())
    return loss_func(X, y, theta) + reg


# 梯度下降
def grad_des(X, y, theta, l, alpha, epoch):
    loss = np.zeros(epoch)  # 初始化一个ndarray，包含每次epoch的loss
    m = X.shape[0]  # 样本数量m
    for i in range(epoch):
        # 利用向量化一步求解
        theta = theta - (alpha / m) * (X * theta.T - y).T * X - (alpha * l / m) * theta  # 添加了正则项
        loss[i] = reg_loss(X, y, theta, l)  # 记录每次迭代后的代价函数值
    return theta, loss


# 运行梯度下降算法 并得出最终拟合的theta值 代价函数J(theta)
final_theta, loss = grad_des(X_train, y_train, theta, config['l'], config['alpha'], config['epoch'])
print(final_theta)

# 模型评估
train_y_est = X_train * final_theta.T
test_y_est = X_test * final_theta.T
mse = np.sum(np.power(test_y_est - y_test, 2)) / (len(X_test))
rmse = np.sqrt(mse)
R2_train = 1 - np.sum(np.power(train_y_est - y_train, 2)) / np.sum(np.power(np.mean(y_train) - y_train, 2))
R2_test = 1 - np.sum(np.power(test_y_est - y_test, 2)) / np.sum(np.power(np.mean(y_test) - y_test, 2))
print("均方误差[MSE]:", mse, "均方根误差[RMSE]:", rmse, "训练集R方:", R2_train, "测试集R方:", R2_test)

# 绘制迭代曲线
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(np.arange(config['epoch']), loss, 'r')  # np.arange()返回等差数组
ax.set_xlabel('epoch')
ax.set_ylabel('loss')
ax.set_title('loss on each epoch')
plt.show()

# 图例展示预测值与真实值的变化趋势
# plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
# plt.rcParams['axes.unicode_minus'] = False
plt.figure(facecolor='w')
t = np.arange(len(X_test))  # 创建等差数组
plt.plot(t, y_test, 'r-', linewidth=2, label=u'truth')
plt.plot(t, test_y_est, 'b-', linewidth=2, label=u'prediction')
plt.legend(loc='upper right')
plt.title(u'Boston Housing Price Prediction by MLR', fontsize=18)
plt.grid(b=True, linestyle='--')
plt.show()
