#!/usr/bin/python3
# coding: utf-8
"""
@author: Colyn
@group: NJUST
@desc: Logistic Regression & Softmax Regression & BP Neural Network
@date: 2022-12-11
"""
import numpy as np


class DataSet(object):
    """
    数据集类
    """

    def __init__(self, file_X, file_y):
        """
        :param file_X: 数据集x文件
        :param file_y: 数据集y文件
        """
        self.epoch = 1000
        self.alpha = 1e-6
        self.data_X = np.loadtxt(file_X, unpack=False)
        self.data_y = np.loadtxt(file_y, unpack=False)
        self.N = self.data_X.shape[0]
        self.ones = [[1] for _ in range(self.N)]
        self.train_X = np.mat(np.append(self.data_X, self.ones, axis=1))
        self.train_y = np.mat(self.data_y).T


class Logistic(DataSet):
    """
    Logistic回归模型
    """

    def __init__(self, file_X, file_y):
        """
        :param file_X: 数据集x文件
        :param file_y: 数据集y文件
        """
        super().__init__(file_X, file_y)

    def func_sigmoid(self, X_i, theta):
        """
        sigmoid函数
        :param X_i: 第i条数据样本
        :param theta: 参数
        :return: sigmoid
        """
        return 1 / (1 + np.exp(-theta * X_i.T))

    def func_loss(self, theta):
        """
        交叉熵损失函数
        :param theta:
        :return: loss
        """
        tmp = 0
        for i in range(self.N):
            tmp += (-self.train_y[i] * np.log(self.func_sigmoid(self.train_X[i], theta)) -
                    (1 - self.train_y[i]) * np.log(1 - self.func_sigmoid(self.train_X[i], theta)))
        return np.array(tmp / self.N)[0][0]

    def func_gd(self, theta, alpha, epoch=100):
        """
        梯度下降算法
        :param alpha: 学习率
        :param theta: 参数
        :param epoch: 训练轮数
        :return: theta & loss
        """
        pre_theta, self.alpha, loss, self.epoch = theta, alpha, [], epoch
        for e in range(self.epoch):
            print(f"Logistic-GD: epoch [{e + 1}] loss:", self.func_loss(theta))
            sum = 0
            for i in range(self.N):
                sum += (self.train_y[i] - self.func_sigmoid(self.train_X[i], theta)) * self.train_X[i]
            theta = pre_theta + sum * self.alpha / self.N
            pre_theta = theta
            loss.append(self.func_loss(theta))
        return np.array(theta)[0], loss

    def func_sgd(self, theta, alpha, epoch=100):
        """
        随机梯度下降算法
        :param alpha: 学习率
        :param theta: 参数
        :param epoch: 训练轮数
        :return: theta & loss
        """
        pre_theta, self.alpha, loss, self.epoch = theta, alpha, [], epoch
        for e in range(self.epoch):
            print(f"Logistic-SGD: epoch [{e + 1}] loss:", self.func_loss(theta))
            i = np.random.randint(0, 80)
            theta = pre_theta + self.alpha * (self.train_y[i] - self.func_sigmoid(self.train_X[i], theta)) * \
                    self.train_X[i]
            pre_theta = theta
            loss.append(self.func_loss(theta))
        return np.array(theta)[0], loss


class Softmax(DataSet):
    """
    Softmax回归模型
    """

    def __init__(self, file_X, file_y):
        """
        :param file_X: 数据集x文件
        :param file_y: 数据集y文件
        """
        super().__init__(file_X, file_y)
        # 迭代器，将y转换成one-hot形式
        one_hot = [[0, 1] if self.train_y[i] == 1 else [1, 0] for i in range(self.N)]
        self.train_y = np.mat(one_hot)

    def func_softmax(self, X_i, theta):
        """
        softmax函数
        :param X_i: 第i条数据样本
        :param theta: 参数
        :return: softmax
        """
        sum = 0
        for i in range(theta.shape[0]):
            sum += np.exp(theta[i] * X_i.T)
        return np.mat(np.exp(theta * X_i.T) / sum)  # 2x1

    def func_loss(self, theta):
        """
        损失函数
        :param theta: 参数
        :return: loss
        """
        tmp = 0
        for i in range(self.N):
            tmp += (-self.train_y[i] * np.log(self.func_softmax(self.train_X[i], theta)))
        return np.array(tmp / self.N)[0][0]  # 1x1

    def func_gd(self, theta, alpha, epoch=100):
        """
        梯度下降算法
        :param alpha: 学习率
        :param theta: 参数
        :param epoch: 训练轮数
        :return: theta & loss
        """
        pre_theta, self.alpha, loss, self.epoch = theta, alpha, [], epoch
        for e in range(self.epoch):
            print(f"Softmax-GD: epoch [{e + 1}] loss:", self.func_loss(theta))
            sum = 0
            for i in range(self.N):
                tmp = np.mat((self.train_y[i].T - self.func_softmax(self.train_X[i], theta)))  # 2X1
                sum += tmp * self.train_X[i]  # 2x3
            theta = pre_theta + sum * self.alpha / self.N
            pre_theta = theta
            loss.append(self.func_loss(theta))
        return np.array(theta), loss

    def func_sgd(self, theta, alpha, epoch=100):
        """
        随机梯度下降算法
        :param alpha: 学习率
        :param theta: 参数
        :param epoch: 训练轮数
        :return: theta & loss
        """
        pre_theta, self.alpha, loss, self.epoch = theta, alpha, [], epoch
        for e in range(self.epoch):
            print(f"Softmax-SGD: epoch [{e + 1}] loss:", self.func_loss(theta))
            i = np.random.randint(0, 80)
            tmp = np.mat(self.alpha * (self.train_y[i] - self.func_softmax(self.train_X[i], theta).T))
            theta = pre_theta + tmp.T * self.train_X[i]
            pre_theta = theta
            loss.append(self.func_loss(theta))
        return np.array(theta), loss


class BP(DataSet):
    """
    BP神经网络算法模型
    """

    def __init__(self, file_X, file_y):
        """
        :param file_X: 数据集x文件
        :param file_y: 数据集y文件
        """
        self.K = 1
        self.index = 1
        self.w1 = None  # 存储输入层到隐藏层的权重参数
        self.w2 = None  # 存储隐藏层到输出层的权重参数
        super().__init__(file_X, file_y)
        # 迭代器，将y转换成one-hot形式
        one_hot = [[0, 1] if self.train_y[i] == 1 else [1, 0] for i in range(self.N)]
        self.train_y = np.mat(one_hot)

    def set_args(self, K, index, h_n=2, o_n=2, alpha=1e-6, epoch=1000) -> None:
        """
        超参数设置
        :param h_n: 隐藏层神经元个数
        :param o_n: 输出层神经元个数
        :param K: K折划分
        :param alpha: 学习率
        :param epoch: 训练轮数
        :param index: 选定的测试集下标
        """
        # 初始化参数w1
        self.w1 = np.mat([[0 for _ in range(h_n)] for _ in range(3)])  # h_n=2, o_n=2时, 3x2
        # 初始化参数w2
        self.w2 = np.mat([[0 for _ in range(o_n)] for _ in range(h_n + 1)])  # h_n=2, o_n=2时, 3x2
        self.alpha = alpha
        self.epoch = epoch
        self.index = index
        self.K = K
        self.index = index
        assert 1 <= self.index <= self.K, "The index must be lower than K"

    def func_sigmoid(self, X):
        """
        sigmoid函数
        :param X: 净输入值
        :return: sigmoid净输出值
        """
        return 1 / (1 + np.exp(-X))

    def func_softmax(self, Z):
        """
        softmax函数
        :param Z: 净输入值
        :return: softmax净输出值
        """
        sum = 0
        for i in range(Z.size):
            sum += np.exp(Z[i])
        return np.exp(Z) / sum  # 2x1

    def output_loss(self, a, pos):
        """
        返回单条数据的损失值
        :param a: 净输出值向量
        :param pos: 该输出值对应的输入值所在的索引
        :return: loss
        """
        return float(-self.train_y[pos] * np.log(a))  # (1xn)x(nx1) = 1x1

    def func_hadamard(self, A1, A2):
        """
        矩阵的Hadamard积
        :param A1: 矩阵1
        :param A2: 矩阵2
        :return: A1oA2
        """
        assert A1.shape == A2.shape, "The A1’s dimension must be the same as A2's!"
        for i in range(A1.shape[0]):
            for j in range(A1.shape[1]):
                A1[i][j] = A1[i][j] * A2[i][j]
        return A1

    def func_train(self):
        """
        训练模型
        :return self.w1: 输入层->隐藏层参数
        :return self.w2: 隐藏层->输出层参数
        """
        # 测试集索引
        length = int(self.N // self.K)
        index = np.linspace((self.index - 1) * length, self.index * length - 1, length)
        ret_loss = []
        for e in range(self.epoch):
            loss = 0  # 每轮训练的输出层损失值
            for i in range(self.N):
                if i in index:
                    pass
                else:
                    # 计算隐藏层神经元
                    z1 = self.w1.T * self.train_X[i].T  # 2x1
                    a1 = self.func_sigmoid(z1)  # 2x1
                    # 计算输出层神经元
                    z2 = self.w2.T * np.r_[a1, [[1]]]  # 新增偏置项b, 2x1
                    a2 = self.func_softmax(z2)  # 2x1
                    # 计算输出层误差值
                    delta_L = a2 - self.train_y[i].T  # 2x1
                    # 更新隐藏层->输出层参数
                    self.w2 = self.w2 - self.alpha * np.r_[a1, [[1]]] * delta_L.T  # (3x1)x(1x2)=3x2
                    # 计算隐藏层误差值
                    # sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
                    g_ = self.func_hadamard(self.func_sigmoid(z1),
                                            1 - self.func_sigmoid(z1))  # 2x1
                    delta_l = self.func_hadamard(g_, self.w2[:2] * delta_L)  # 2x1
                    # 更新输入层->隐藏层参数
                    self.w1 = self.w1 - self.alpha * self.train_X[i].T * delta_l.T  # (3x1)x(1x2)=3x2
                    # 计算输出层的loss和
                    loss += self.output_loss(a2, i)
            loss /= (self.N - length)
            ret_loss.append(loss)
            print(f"BP Algorithm: epoch [{e + 1}] loss:", loss)
        return self.w1, self.w2, self.train_X[(self.index - 1) * length:(self.index * length)], ret_loss

    def func_predict(self, w1, w2, test_X, n):
        """
        预测函数
        :param w1: 输入层->隐藏层参数
        :param w2: 隐藏层->输出层参数
        :return yield result: 预测值
        """
        for j in range(n):
            # 计算隐藏层神经元
            z1 = w1.T * test_X[j].T
            a1 = self.func_sigmoid(z1)
            # 计算输出层神经元
            z2 = w2.T * np.r_[a1, [[1]]]  # 新增偏置项b
            a2 = self.func_softmax(z2)  # 2x1
            yield 1 if a2[0] >= a2[1] else 0


# !/usr/bin/python3
# coding: utf-8
"""
@author: Colyn
@group: NJUST
@desc: Logistic Regression & Softmax Regression & BP Neural Network
@date: 2022-12-11
"""

import numpy as np


class DataSet(object):
    """
    数据集类
    """

    def __init__(self, file_X, file_y):
        """
        :param file_X: 数据集x文件
        :param file_y: 数据集y文件
        """
        self.epoch = 1000
        self.alpha = 1e-6
        self.data_X = np.loadtxt(file_X, unpack=False)
        self.data_y = np.loadtxt(file_y, unpack=False)
        self.N = self.data_X.shape[0]
        self.ones = [[1] for _ in range(self.N)]
        self.train_X = np.mat(np.append(self.data_X, self.ones, axis=1))
        self.train_y = np.mat(self.data_y).T


class Logistic(DataSet):
    """
    Logistic回归模型
    """

    def __init__(self, file_X, file_y):
        """
        :param file_X: 数据集x文件
        :param file_y: 数据集y文件
        """
        super().__init__(file_X, file_y)

    def func_sigmoid(self, X_i, theta):
        """
        sigmoid函数
        :param X_i: 第i条数据样本
        :param theta: 参数
        :return: sigmoid
        """
        return 1 / (1 + np.exp(-theta * X_i.T))

    def func_loss(self, theta):
        """
        交叉熵损失函数
        :param theta:
        :return: loss
        """
        tmp = 0
        for i in range(self.N):
            tmp += (-self.train_y[i] * np.log(self.func_sigmoid(self.train_X[i], theta)) -
                    (1 - self.train_y[i]) * np.log(1 - self.func_sigmoid(self.train_X[i], theta)))
        return np.array(tmp / self.N)[0][0]

    def func_gd(self, theta, alpha, epoch=100):
        """
        梯度下降算法
        :param alpha: 学习率
        :param theta: 参数
        :param epoch: 训练轮数
        :return: theta & loss
        """
        pre_theta, self.alpha, loss, self.epoch = theta, alpha, [], epoch
        for e in range(self.epoch):
            print(f"Logistic-GD: epoch [{e + 1}] loss:", self.func_loss(theta))
            sum = 0
            for i in range(self.N):
                sum += (self.train_y[i] - self.func_sigmoid(self.train_X[i], theta)) * self.train_X[i]
            theta = pre_theta + sum * self.alpha / self.N
            pre_theta = theta
            loss.append(self.func_loss(theta))
        return np.array(theta)[0], loss

    def func_sgd(self, theta, alpha, epoch=100):
        """
        随机梯度下降算法
        :param alpha: 学习率
        :param theta: 参数
        :param epoch: 训练轮数
        :return: theta & loss
        """
        pre_theta, self.alpha, loss, self.epoch = theta, alpha, [], epoch
        for e in range(self.epoch):
            print(f"Logistic-SGD: epoch [{e + 1}] loss:", self.func_loss(theta))
            i = np.random.randint(0, 80)
            theta = pre_theta + self.alpha * (self.train_y[i] - self.func_sigmoid(self.train_X[i], theta)) * \
                    self.train_X[i]
            pre_theta = theta
            loss.append(self.func_loss(theta))
        return np.array(theta)[0], loss


class Softmax(DataSet):
    """
    Softmax回归模型
    """

    def __init__(self, file_X, file_y):
        """
        :param file_X: 数据集x文件
        :param file_y: 数据集y文件
        """
        super().__init__(file_X, file_y)
        # 迭代器，将y转换成one-hot形式
        one_hot = [[0, 1] if self.train_y[i] == 1 else [1, 0] for i in range(self.N)]
        self.train_y = np.mat(one_hot)

    def func_softmax(self, X_i, theta):
        """
        softmax函数
        :param X_i: 第i条数据样本
        :param theta: 参数
        :return: softmax
        """
        sum = 0
        for i in range(theta.shape[0]):
            sum += np.exp(theta[i] * X_i.T)
        return np.mat(np.exp(theta * X_i.T) / sum)  # 2x1

    def func_loss(self, theta):
        """
        损失函数
        :param theta: 参数
        :return: loss
        """
        tmp = 0
        for i in range(self.N):
            tmp += (-self.train_y[i] * np.log(self.func_softmax(self.train_X[i], theta)))
        return np.array(tmp / self.N)[0][0]  # 1x1

    def func_gd(self, theta, alpha, epoch=100):
        """
        梯度下降算法
        :param alpha: 学习率
        :param theta: 参数
        :param epoch: 训练轮数
        :return: theta & loss
        """
        pre_theta, self.alpha, loss, self.epoch = theta, alpha, [], epoch
        for e in range(self.epoch):
            print(f"Softmax-GD: epoch [{e + 1}] loss:", self.func_loss(theta))
            sum = 0
            for i in range(self.N):
                tmp = np.mat((self.train_y[i].T - self.func_softmax(self.train_X[i], theta)))  # 2X1
                sum += tmp * self.train_X[i]  # 2x3
            theta = pre_theta + sum * self.alpha / self.N
            pre_theta = theta
            loss.append(self.func_loss(theta))
        return np.array(theta), loss

    def func_sgd(self, theta, alpha, epoch=100):
        """
        随机梯度下降算法
        :param alpha: 学习率
        :param theta: 参数
        :param epoch: 训练轮数
        :return: theta & loss
        """
        pre_theta, self.alpha, loss, self.epoch = theta, alpha, [], epoch
        for e in range(self.epoch):
            print(f"Softmax-SGD: epoch [{e + 1}] loss:", self.func_loss(theta))
            i = np.random.randint(0, 80)
            tmp = np.mat(self.alpha * (self.train_y[i] - self.func_softmax(self.train_X[i], theta).T))
            theta = pre_theta + tmp.T * self.train_X[i]
            pre_theta = theta
            loss.append(self.func_loss(theta))
        return np.array(theta), loss


class BP(DataSet):
    """
    BP神经网络算法模型
    """

    def __init__(self, file_X, file_y):
        """
        :param file_X: 数据集x文件
        :param file_y: 数据集y文件
        """
        self.K = 1
        self.index = 1
        self.w1 = None  # 存储输入层到隐藏层的权重参数
        self.w2 = None  # 存储隐藏层到输出层的权重参数
        super().__init__(file_X, file_y)
        # 迭代器，将y转换成one-hot形式
        one_hot = [[0, 1] if self.train_y[i] == 1 else [1, 0] for i in range(self.N)]
        self.train_y = np.mat(one_hot)

    def set_args(self, K, index, h_n=2, o_n=2, alpha=1e-6, epoch=1000) -> None:
        """
        超参数设置
        :param h_n: 隐藏层神经元个数
        :param o_n: 输出层神经元个数
        :param K: K折划分
        :param alpha: 学习率
        :param epoch: 训练轮数
        :param index: 选定的测试集下标
        """
        # 初始化参数w1
        self.w1 = np.mat([[0 for _ in range(h_n)] for _ in range(3)])  # h_n=2, o_n=2时, 3x2
        # 初始化参数w2
        self.w2 = np.mat([[0 for _ in range(o_n)] for _ in range(h_n + 1)])  # h_n=2, o_n=2时, 3x2
        self.alpha = alpha
        self.epoch = epoch
        self.index = index
        self.K = K
        self.index = index
        assert 1 <= self.index <= self.K, "The index must be lower than K"

    def func_sigmoid(self, X):
        """
        sigmoid函数
        :param X: 净输入值
        :return: sigmoid净输出值
        """
        return 1 / (1 + np.exp(-X))  # [xi]mxn

    def func_softmax(self, Z):
        """
        softmax函数
        :param Z: 净输入值
        :return: softmax净输出值
        """
        sum = 0
        for i in range(Z.size):
            sum += np.exp(Z[i])
        return np.exp(Z) / sum  # nx1

    def output_loss(self, a, pos):
        """
        返回单条数据的损失值
        :param a: 净输出值向量
        :param pos: 该输出值对应的输入值所在的索引
        :return: loss
        """
        return float(-self.train_y[pos] * np.log(a))  # (1xn)x(nx1) = 1x1

    def func_hadamard(self, A1, A2):
        """
        矩阵的Hadamard积
        :param A1: 矩阵1
        :param A2: 矩阵2
        :return: A1oA2
        """
        assert A1.shape == A2.shape, "The A1’s dimension must be the same as A2's!"
        for i in range(A1.shape[0]):
            for j in range(A1.shape[1]):
                A1[i][j] = A1[i][j] * A2[i][j]
        return A1

    def func_train(self):
        """
        训练模型
        :return self.w1: 输入层->隐藏层参数
        :return self.w2: 隐藏层->输出层参数
        """
        # 测试集索引
        length = int(self.N // self.K)
        index = np.linspace((self.index - 1) * length, self.index * length - 1, length)
        ret_loss = []
        for e in range(self.epoch):
            loss = 0  # 每轮训练的输出层损失值
            for i in range(self.N):
                if i in index:
                    pass
                else:
                    # 计算隐藏层神经元
                    z1 = self.w1.T * self.train_X[i].T  # 2x1
                    a1 = self.func_sigmoid(z1)  # 2x1
                    # 计算输出层神经元
                    z2 = self.w2.T * np.r_[a1, [[1]]]  # 新增偏置项b, 2x1
                    a2 = self.func_softmax(z2)  # 2x1
                    # 计算输出层误差值
                    delta_L = a2 - self.train_y[i].T  # 2x1
                    # 更新隐藏层->输出层参数
                    self.w2 = self.w2 - self.alpha * np.r_[a1, [[1]]] * delta_L.T  # (3x1)x(1x2)=3x2
                    # 计算隐藏层误差值
                    # sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
                    g_ = self.func_hadamard(self.func_sigmoid(z1),
                                            1 - self.func_sigmoid(z1))  # 2x1
                    delta_l = self.func_hadamard(g_, self.w2[:2] * delta_L)  # 2x1
                    # 更新输入层->隐藏层参数
                    self.w1 = self.w1 - self.alpha * self.train_X[i].T * delta_l.T  # (3x1)x(1x2)=3x2
                    # 计算输出层的loss和
                    loss += self.output_loss(a2, i)
            loss /= (self.N - length)
            ret_loss.append(loss)
            print(f"BP Algorithm: epoch [{e + 1}] loss:", loss)
        return self.w1, self.w2, self.train_X[(self.index - 1) * length:(self.index * length)], ret_loss

    def func_predict(self, w1, w2, test_X, n):
        """
        预测函数
        :param w1: 输入层->隐藏层参数
        :param w2: 隐藏层->输出层参数
        :return yield result: 预测值
        """
        for j in range(n):
            # 计算隐藏层神经元
            z1 = w1.T * test_X[j].T
            a1 = self.func_sigmoid(z1)
            # 计算输出层神经元
            z2 = w2.T * np.r_[a1, [[1]]]  # 新增偏置项b
            a2 = self.func_softmax(z2)  # 2x1
            yield 1 if a2[0] >= a2[1] else 0
