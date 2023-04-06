#!/usr/bin/python3
# coding: utf-8
"""
@author: Colyn
@group: NJUST
@desc: Logistic Regression & SVM
@date: 2022-11-28
"""
from pylab import *
from utils import *
from matplotlib import pyplot as plt


def func_plt(data: list, label: list, **kwargs) -> None:
    title, x_label, y_label, N = "Figure", "x", "y", len(label)
    for key in kwargs:
        if 'title' in key:
            title = kwargs[key]
        elif 'x_label' in key:
            x_label = kwargs[key]
        elif 'y_label' in key:
            y_label = kwargs[key]
        else:
            pass

    plt.figure(dpi=300, figsize=(8, 8))
    pos, neg = [[], []], [[], []]
    for i in range(N):
        if label[i] == 1:
            pos[0].append(data[i][0]), pos[1].append(data[i][1])
        else:
            neg[0].append(data[i][0]), neg[1].append(data[i][1])

    plt.scatter(pos[0][:], pos[1][:], c="blue", label="pos", alpha=0.8, edgecolors='white')
    plt.scatter(neg[0][:], neg[1][:], c="red", label="neg", alpha=0.8, edgecolors='white')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend(loc='best')
    plt.show()


if __name__ == "__main__":
    # 数据集
    file_X = 'ex4Data/ex4x.dat'
    file_y = 'ex4Data/ex4y.dat'

    # # ############################## Logistic模型 ##########################################
    # logistic = Logistic(file_X=file_X, file_y=file_y)
    # theta = np.mat([0, 0, 0]) # 初始化参数
    # theta_gd = logistic.func_gd(theta, alpha=1e-3, epoch=1000)  # 采用GD算法优化模型
    # print("theta_gd:", theta_gd[0], "loss:", theta_gd[1])
    # theta_sgd = logistic.func_sgd(theta, alpha=1e-6, epoch=1000)  # 采用SGD算法优化模型
    # print("theta_sgd:", theta_sgd[0], "loss:", theta_sgd[1])
    # ################################ SVM模型 #############################################
    svm = SVM(file_X=file_X, file_y=file_y)
    # 线性不可分SVM
    # svm.func_train(
    #     args='-s 0 -c 4 -b 1',  # 惩罚系数C=4
    #     path='SVC_4'
    # )
    # svm.func_train(
    #     args='-s 0 -c 0.5 -b 1',  # 惩罚系数C=0.5
    #     path='SVC_0.5'
    # )
    # 线性不可分SVM核技巧
    # svm.func_train(
    #     args='-t 0 -b 1',    # 线性核
    #     path='SVC_linear_kernel'
    # )
    # svm.func_train(
    #     args='-t 1 -b 1',  # 多项式性核
    #     path='SVC_polynomial_kernel'
    # )
    svm.func_train(
        args='-t 2 -b 1',  # 高斯核
        path='SVC_gaussian_kernel'
    )
    # # ############################## 可视化数据集 ############################################
    # func_plt(
    #     svm.train_X,
    #     svm.train_y,
    #     title="Data Distribution",
    #     x_label="score1",
    #     y_label="score2"
    # )
