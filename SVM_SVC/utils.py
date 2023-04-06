#!/usr/bin/python3
# coding: utf-8
"""
@author: Colyn
@group: NJUST
@desc: Logistic Regression & SVM
@date: 2022-11-28
"""
import numpy as np
from libsvm.svmutil import *


class DataSet(object):
    """
    数据集类
    """

    def __init__(self, file_X, file_y):
        """
        :param file_X: 数据集x文件
        :param file_y: 数据集y文件
        """
        self.data_X = np.loadtxt(file_X, unpack=False)
        self.data_y = np.loadtxt(file_y, unpack=False)
        self.N = self.data_X.shape[0]
        self.train_X = self.data_X
        self.train_y = self.data_y

    def get_train(self):
        """
        获取预处理后的数据
        :return: train_X, train_y
        """
        return self.train_X, self.train_y


class Logistic(DataSet):
    """
    Logistic回归模型
    """

    def __init__(self, file_X, file_y, alpha=None):
        """
        :param file_X: 数据集x文件
        :param file_y: 数据集y文件
        """
        self.epoch = None
        self.alpha = alpha
        super().__init__(file_X, file_y)
        self.ones = [[1] for _ in range(self.N)]
        self.train_X = np.mat(np.append(self.data_X, self.ones, axis=1))
        self.train_y = np.mat(self.data_y).T

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
                sum += (self.train_y[i] - self.func_sigmod(self.train_X[i], theta)) * self.train_X[i]
            theta = pre_theta + sum * self.alpha / self.N
            pre_theta = theta
            loss.append(self.func_loss(theta))
        return np.array(theta)[0], self.func_loss(theta), loss

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
            theta = pre_theta + self.alpha * (self.train_y[i] - self.func_sigmod(self.train_X[i], theta)) * \
                    self.train_X[i]
            pre_theta = theta
            loss.append(self.func_loss(theta))
        return np.array(theta)[0], self.func_loss(theta), loss


class SVM(DataSet):
    """
    支持向量机模型
    """

    def __init__(self, file_X, file_y, alpha=None):
        """
        :param file_X: 数据集x文件
        :param file_y: 数据集y文件
        """
        self.alpha = alpha
        self.opt = '-t 0'
        super().__init__(file_X, file_y)
        tmp = self.train_y
        self.train_y = [-1 if tmp[i] == 0 else 1 for i in range(tmp.size)]

    def func_train(self, args: str, path: str) -> None:
        """
        模型训练
        :param args: 训练参数
        :param path: 模型参数保存路径
        """
        _model = svm_train(self.train_y, self.train_X, args)
        svm_save_model(path, _model)
