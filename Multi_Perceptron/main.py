#!/usr/bin/python3
# coding: utf-8
"""
@author: Colyn
@group: NJUST
@desc: Logistic Regression & Softmax Regression
@date: 2022-11-18
"""
from utils import *
from pylab import *
from matplotlib import pyplot as plt


def plt_loss(loss: list, **kwargs) -> None:
    y = loss
    size, fig = len(loss), plt.figure(dpi=300, figsize=(24, 8))
    X = linspace(0, size, size)

    axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    axes.plot(X, y, 'r')
    axes.set_xlabel(kwargs['x_label'])
    axes.set_ylabel(kwargs['y_label'])
    axes.set_title(kwargs['title'])

    plt.show()


if __name__ == '__main__':
    file_X = 'ex4Data/ex4x.dat'
    file_y = 'ex4Data/ex4y.dat'
    l_theta = np.mat([0, 0, 0])
    p_theta = np.mat([0, 0, 0])
    s_theta = np.mat([[0, 0, 0], [0, 0, 0]])
    mp_theta = np.mat([[0.1, 0.1, 0.1], [0.1, 0.1, 0.1]])
    l_train_set = Logistic(file_X=file_X, file_y=file_y)  # 实例化训练集
    l_train_set.set_params(epoch=1000)  # 设置超参数
    s_train_set = Softmax(file_X=file_X, file_y=file_y)  # 实例化训练集
    s_train_set.set_params(epoch=1000)  # 设置超参数
    p_train_set = Perceptron(file_X=file_X, file_y=file_y)  # 实例化训练集
    p_train_set.set_params(epoch=1000)  # 设置超参数
    mp_train_set = MultiPerceptron(file_X=file_X, file_y=file_y)  # 实例化训练集
    mp_train_set.set_params(epoch=1000)  # 设置超参数

    # 开始训练
    # Logistic模型
    # l_theta_gd = l_train_set.func_gd(theta=l_theta, alpha=1e-3)  # 采用GD算法优化模型
    # l_theta_sgd = l_train_set.func_sgd(theta=l_theta, alpha=1e-6)  # 采用SGD算法优化模型
    #
    # # Softmax模型
    # s_theta_gd = s_train_set.func_gd(theta=s_theta, alpha=1e-4)  # 采用GD算法优化模型
    # s_theta_sgd = s_train_set.func_sgd(theta=s_theta, alpha=1e-6)  # 采用SGD算法优化模型
    #
    # # Perceptron模型
    p_theta_gd = p_train_set.func_gd(theta=p_theta, alpha=3e-7)  # 采用GD算法优化模型
    # p_theta_sgd = p_train_set.func_sgd(theta=p_theta, alpha=3e-7)  # 采用SGD算法优化模型

    # MultiPerceptron模型
    # mp_theta_gd = mp_train_set.func_gd(theta=mp_theta, alpha=3e-7)  # 采用GD算法优化模型
    # mp_theta_sgd = mp_train_set.func_sgd(theta=mp_theta, alpha=1e-4)  # 采用SGD算法优化模型

    # 打印结果
    # print("l_theta_gd:", l_theta_gd[0], "loss:", l_theta_gd[1])
    # print("l_theta_sgd:", l_theta_sgd[0], "loss:", l_theta_sgd[1])
    # print("s_theta_gd:", s_theta_gd[0], "loss:", s_theta_gd[1])
    # print("s_theta_sgd:", s_theta_sgd[0], "loss:", s_theta_sgd[1])
    # print("p_theta_gd:", p_theta_gd[0], "loss:", p_theta_gd[1])
    # print("p_theta_sgd:", p_theta_sgd[0], "loss:", p_theta_sgd[1])
    # print("mp_theta_gd:", mp_theta_gd[0], "loss:", mp_theta_gd[1])
    # print("mp_theta_sgd:", mp_theta_sgd[0], "loss:", mp_theta_sgd[1])

    # 结果比较
    # plt_loss(l_theta_sgd[2], x_label="epoch", y_label="loss", title="Logistic")
    # plt_loss(s_theta_sgd[2], x_label="epoch", y_label="loss", title="Softmax")
    # plt_loss(p_theta_sgd[2], x_label="epoch", y_label="loss", title="Perceptron")
    # plt_loss(mp_theta_sgd[2], x_label="epoch", y_label="loss", title="MultiPerceptron")
    plt_loss(p_theta_gd[2], x_label="epoch", y_label="loss", title="Perceptron")
