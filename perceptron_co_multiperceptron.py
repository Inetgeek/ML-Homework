#!/usr/bin/python3
# coding: utf-8
"""
@author: Colyn
@group: NJUST
@desc: Logistic Regression & Softmax Regression
@date: 2022-11-18
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
        self.epoch = None
        self.data_X = np.loadtxt(file_X, unpack=False)
        self.data_y = np.loadtxt(file_y, unpack=False)
        self.N = self.data_X.shape[0]
        self.ones = [[1] for _ in range(self.N)]
        self.train_X = np.mat(np.append(self.data_X, self.ones, axis=1))
        self.train_y = np.mat(self.data_y).T

    def get_train(self):
        """
        获取预处理后的数据
        :return: train_X, train_y
        """
        return self.train_X, self.train_y

    def set_params(self, epoch):
        """
        设置超参数
        :param epoch: 训练轮数
        :return:NULL
        """
        self.epoch = epoch


class Logistic(DataSet):
    """
    Logistic回归模型
    """

    def __init__(self, file_X, file_y):
        """
        :param file_X: 数据集x文件
        :param file_y: 数据集y文件
        """
        self.alpha = None
        super().__init__(file_X, file_y)

    def func_sigmod(self, X_i, theta):
        """
        sigmod函数
        :param X_i: 第i条数据样本
        :param theta: 参数
        :return: sigmod
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
            tmp += (-self.train_y[i] * np.log(self.func_sigmod(self.train_X[i], theta)) -
                    (1 - self.train_y[i]) * np.log(1 - self.func_sigmod(self.train_X[i], theta)))
        return np.array(tmp / self.N)[0][0]

    def func_gd(self, theta, alpha):
        """
        梯度下降算法
        :param alpha: 学习率
        :param theta: 参数
        :return: theta & loss
        """
        pre_theta, self.alpha = theta, alpha
        for e in range(self.epoch):
            print(f"Logistic-GD: epoch [{e + 1}] loss:", self.func_loss(theta))
            sum = 0
            for i in range(self.N):
                sum += (self.train_y[i] - self.func_sigmod(self.train_X[i], theta)) * self.train_X[i]
            theta = pre_theta + sum * self.alpha / self.N
            pre_theta = theta
        return np.array(theta)[0], self.func_loss(theta)

    def func_sgd(self, theta, alpha):
        """
        随机梯度下降算法
        :param alpha: 学习率
        :param theta: 参数
        :return: theta & loss
        """
        pre_theta, self.alpha = theta, alpha
        for e in range(self.epoch):
            print(f"Logistic-SGD: epoch [{e + 1}] loss:", self.func_loss(theta))
            i = np.random.randint(0, 80)
            theta = pre_theta + self.alpha * (self.train_y[i] - self.func_sigmod(self.train_X[i], theta)) * \
                    self.train_X[i]
            pre_theta = theta
        return np.array(theta)[0], self.func_loss(theta)


class Softmax(DataSet):
    """
    Softmax回归模型
    """

    def __init__(self, file_X, file_y):
        """
        :param file_X: 数据集x文件
        :param file_y: 数据集y文件
        """
        self.alpha = None
        super().__init__(file_X, file_y)
        one_hot = [[0, 1] if self.train_y[i] == 1 else [1, 0] for i in range(self.N)]  # 迭代器，将y转换成one-hot形式
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

    def func_gd(self, theta, alpha):
        """
        梯度下降算法
        :param alpha: 学习率
        :param theta: 参数
        :return: theta & loss
        """
        pre_theta, self.alpha = theta, alpha
        for e in range(self.epoch):
            print(f"Softmax-GD: epoch [{e + 1}] loss:", self.func_loss(theta))
            sum = 0
            for i in range(self.N):
                tmp = np.mat((self.train_y[i].T - self.func_softmax(self.train_X[i], theta)))  # 2X1
                sum += tmp * self.train_X[i]  # 2x3
            theta = pre_theta + sum * self.alpha / self.N
            pre_theta = theta

        return np.array(theta), self.func_loss(theta)

    def func_sgd(self, theta, alpha):
        """
        随机梯度下降算法
        :param alpha: 学习率
        :param theta: 参数
        :return: theta & loss
        """
        pre_theta, self.alpha = theta, alpha
        for e in range(self.epoch):
            print(f"Softmax-SGD: epoch [{e + 1}] loss:", self.func_loss(theta))
            i = np.random.randint(0, 80)
            tmp = np.mat(self.alpha * (self.train_y[i] - self.func_softmax(self.train_X[i], theta).T))
            theta = pre_theta + tmp.T * self.train_X[i]
            pre_theta = theta
        return np.array(theta), self.func_loss(theta)


class Perceptron(DataSet):
    """
    二分类感知机模型
    """

    def __init__(self, file_X, file_y):
        """
        :param file_X: 数据集x文件
        :param file_y: 数据集y文件
        """
        self.alpha = None
        super().__init__(file_X, file_y)
        tmp_y = [[-1] if self.train_y[i] == 0 else [1] for i in range(self.N)]
        self.train_y = np.mat(tmp_y)

    def func_perceptron(self, X_i, theta):
        """
        假设函数
        :param X_i: 第i条数据样本
        :param theta: 参数
        :return: theta^T*X_i
        """
        return theta * X_i.T

    def func_loss(self, theta):
        """
        损失函数
        :param theta: 参数
        :return: loss
        """
        tmp = 0
        for i in range(self.N):
            multi = self.train_y[i] * self.func_perceptron(self.train_X[i], theta)
            tmp += (np.zeros([1, 1]) if multi >= 0 else -multi)
        return np.array(tmp / self.N)[0][0]

    def func_gd(self, theta, alpha):
        """
        梯度下降算法
        :param alpha: 学习率
        :param theta: 参数
        :return: theta & loss
        """
        pre_theta, self.alpha = theta, alpha
        for e in range(self.epoch):
            self.func_loss(theta)
            print(f"Perceptron-GD: epoch [{e + 1}] loss:", self.func_loss(theta))
            sum = 0
            for i in range(self.N):
                multi = self.train_y[i] * self.func_perceptron(self.train_X[i], theta)
                sum += (np.zeros([1, 3]) if multi > 0 else self.train_y[i] * self.train_X[i])
            theta = pre_theta + sum * self.alpha / self.N
            pre_theta = theta
        return np.array(theta)[0], self.func_loss(theta)

    def func_sgd(self, theta, alpha):
        """
        随机梯度下降算法
        :param alpha: 学习率
        :param theta: 参数
        :return: theta & loss
        """
        pre_theta, self.alpha = theta, alpha
        for e in range(self.epoch):
            print(f"Perceptron-SGD: epoch [{e + 1}] loss:", self.func_loss(theta))
            i = np.random.randint(0, 80)
            multi = self.train_y[i] * self.func_perceptron(self.train_X[i], theta)
            theta = pre_theta + self.alpha * (np.zeros([1, 3]) if multi > 0 else self.train_y[i] * self.train_X[i])
            pre_theta = theta
        return np.array(theta)[0], self.func_loss(theta)


class MultiPerceptron(DataSet):
    """
    多分类感知机模型
    """

    def __init__(self, file_X, file_y):
        """
        :param file_X: 数据集x文件
        :param file_y: 数据集y文件
        """
        self.alpha = None
        super().__init__(file_X, file_y)

    def func_perceptron(self, X_i, theta):
        """
        假设函数
        :param X_i: 第i条数据样本
        :param theta: 参数
        :return: theta^T*X_i
        """
        return theta * X_i.T  # 2x1

    def max_args(self, X_i, theta):
        """
        获取最大组参数
        :param X_i: 第i条数据样本
        :param theta: 参数
        :return: arg_max(theta)
        """
        multi = self.func_perceptron(X_i, theta)
        # print("multi:", multi)
        return 0 if multi[0] > multi[1] else 1

    def func_loss(self, theta):
        """
        损失函数
        :param theta: 参数 2x3
        :return: loss
        """
        tmp = 0
        for i in range(self.N):
            c_, y_j = self.max_args(self.train_X[i], theta), int(self.train_y[i])
            multi = theta[c_] * self.train_X[i].T - theta[y_j] * self.train_X[i].T  # 1x1
            tmp += (np.zeros([1, 1]) if c_ == y_j else multi)
            # print("c_:", c_, "y_j:", y_j, "multi:", multi, "tmp:", tmp)
        return np.array(tmp / self.N)[0][0]  # 1x1

    def func_gd(self, theta, alpha):
        """
        梯度下降算法
        :param alpha: 学习率
        :param theta: 参数
        :return: theta & loss
        """
        pre_theta, self.alpha = theta, alpha
        for e in range(self.epoch):
            print(f"MultiPerceptron-GD: epoch [{e + 1}] loss:", self.func_loss(theta))
            for c in range(theta.shape[0]):
                sum, gdn = 0, 0
                for i in range(self.N):
                    c_, y_j = self.max_args(self.train_X[i], theta), int(self.train_y[i])
                    if c_ == y_j:
                        gdn = 0
                    elif c_ != y_j and c == c_:
                        gdn = self.train_X[i]
                    elif c_ != y_j and c == y_j:
                        gdn = -self.train_X[i]
                    elif c_ != y_j and c != c_ and c != y_j:
                        gdn = 0
                    else:
                        pass
                    sum += gdn
                theta[c] = pre_theta[c] - sum * self.alpha / self.N
                pre_theta[c] = theta[c]
        return np.array(theta), self.func_loss(theta)

    def func_sgd(self, theta, alpha):
        """
        随机梯度下降算法
        :param alpha: 学习率
        :param theta: 参数
        :return: theta & loss
        """
        pre_theta, self.alpha = theta, alpha
        for e in range(self.epoch):
            print(f"MultiPerceptron-GD: epoch [{e + 1}] loss:", self.func_loss(theta))
            for c in range(theta.shape[0]):
                gdn = 0
                i = np.random.randint(0, 80)
                c_, y_j = self.max_args(self.train_X[i], theta), int(self.train_y[i])
                if c_ == y_j:
                    gdn = 0
                elif c_ != y_j and c == c_:
                    gdn = self.train_X[i]
                elif c_ != y_j and c == y_j:
                    gdn = -self.train_X[i]
                elif c_ != y_j and c != c_ and c != y_j:
                    gdn = 0
                else:
                    pass
                theta[c] = pre_theta[c] - gdn * self.alpha / self.N
                pre_theta[c] = theta[c]
        return np.array(theta), self.func_loss(theta)


if __name__ == '__main__':
    file_X = 'ex4Data/ex4x.dat'
    file_y = 'ex4Data/ex4y.dat'
    l_theta = np.mat([0, 0, 0])
    p_theta = np.mat([0, 0, 0])
    s_theta = np.mat([[0, 0, 0], [0, 0, 0]])
    mp_theta = np.mat([[0.1, 0.1, 0.1], [0.1, 0.1, 0.1]])
    l_train_set = Logistic(file_X=file_X, file_y=file_y)  # 实例化训练集
    l_train_set.set_params(epoch=10000)  # 设置超参数
    s_train_set = Softmax(file_X=file_X, file_y=file_y)  # 实例化训练集
    s_train_set.set_params(epoch=10000)  # 设置超参数
    p_train_set = Perceptron(file_X=file_X, file_y=file_y)  # 实例化训练集
    p_train_set.set_params(epoch=10000)  # 设置超参数
    mp_train_set = MultiPerceptron(file_X=file_X, file_y=file_y)  # 实例化训练集
    mp_train_set.set_params(epoch=10000)  # 设置超参数
    print("loss:", mp_train_set.func_loss(mp_theta))

    # 开始训练
    # Logistic模型
    l_theta_gd = l_train_set.func_gd(theta=l_theta, alpha=1e-3)  # 采用GD算法优化模型
    l_theta_sgd = l_train_set.func_sgd(theta=l_theta, alpha=1e-6)  # 采用SGD算法优化模型

    # Softmax模型
    s_theta_gd = s_train_set.func_gd(theta=s_theta, alpha=1e-4)  # 采用GD算法优化模型
    s_theta_sgd = s_train_set.func_sgd(theta=s_theta, alpha=1e-6)  # 采用SGD算法优化模型

    # Perceptron模型
    p_theta_gd = p_train_set.func_gd(theta=p_theta, alpha=3e-7)  # 采用GD算法优化模型
    p_theta_sgd = p_train_set.func_sgd(theta=p_theta, alpha=3e-7)  # 采用SGD算法优化模型

    # MultiPerceptron模型
    mp_theta_gd = mp_train_set.func_gd(theta=mp_theta, alpha=3e-7)  # 采用GD算法优化模型
    mp_theta_sgd = mp_train_set.func_sgd(theta=mp_theta, alpha=5e-5)  # 采用SGD算法优化模型

    # 打印结果
    print("l_theta_gd:", l_theta_gd[0], "loss:", l_theta_gd[1])
    print("l_theta_sgd:", l_theta_sgd[0], "loss:", l_theta_sgd[1])
    print("s_theta_gd:", s_theta_gd[0], "loss:", s_theta_gd[1])
    print("s_theta_sgd:", s_theta_sgd[0], "loss:", s_theta_sgd[1])
    print("p_theta_gd:", p_theta_gd[0], "loss:", p_theta_gd[1])
    print("p_theta_sgd:", p_theta_sgd[0], "loss:", p_theta_sgd[1])
    print("mp_theta_gd:", mp_theta_gd[0], "loss:", mp_theta_gd[1])
    print("mp_theta_sgd:", mp_theta_sgd[0], "loss:", mp_theta_sgd[1])
