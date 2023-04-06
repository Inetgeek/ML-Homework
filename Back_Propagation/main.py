#!/usr/bin/python3
# coding: utf-8
"""
@author: Colyn
@group: NJUST
@desc: Logistic Regression & Softmax Regression & BP Neural Network
@date: 2022-12-11
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


if __name__ == "__main__":
    # 数据集
    file_X = 'ex4Data/ex4x.dat'
    file_y = 'ex4Data/ex4y.dat'

    # Logistic回归
    l_theta = np.mat([0, 0, 0])
    logistic = Logistic(file_X=file_X, file_y=file_y)
    l_gd_args, l_gd_loss = logistic.func_gd(l_theta, alpha=1e-3, epoch=10000)
    l_sgd_args, l_sgd_loss = logistic.func_sgd(l_theta, alpha=1e-6, epoch=10000)

    # Sfotmax回归
    s_theta = np.mat([[0, 0, 0], [0, 0, 0]])
    softmax = Softmax(file_X=file_X, file_y=file_y)
    s_gd_args, s_gd_loss = softmax.func_gd(s_theta, alpha=1e-4, epoch=10000)
    s_sgd_args, s_sgd_loss = softmax.func_sgd(s_theta, alpha=1e-6, epoch=10000)

    # BP神经网络
    bp = BP(file_X=file_X, file_y=file_y)
    bp.set_args(
        K=5,  # K折划分
        index=5,  # 测试集索引
        h_n=2,  # 隐藏层神经元
        o_n=2,  # 输出层神经元
        alpha=1e-6,  # 学习率
        epoch=10000  # 训练轮数
    )
    w1, w2, test_X, b_loss = bp.func_train()

    # 预测
    index = int(bp.N // bp.K) * (bp.index - 1)
    for i in bp.func_predict(w1, w2, test_X, n=16):
        print(f"data[{index + 1}] predict:", i, ", truth:", int(bp.data_y[index]))
        index += 1

    # ######################## Logistic、Softmax及BP算法比较 ########################
    print("*" * 40)
    print("[Logistic Regression]\n",
          "[GD] theta:", l_gd_args,
          "[GD] loss:", l_gd_loss[-1]
          )
    print("*" * 40)
    print("[Logistic Regression]\n",
          "[SGD] theta:", l_sgd_args,
          "[SGD] loss:", l_sgd_loss[-1]
          )
    print("*" * 40)
    print("[Softmax Regression]\n",
          "[GD] theta:", s_gd_args,
          "[GD] loss:", s_gd_loss[-1]
          )
    print("*" * 40)
    print("[Softmax Regression]\n",
          "[SGD] theta:", s_sgd_args,
          "[SGD] loss:", s_sgd_loss[-1]
          )
    print("*" * 40)
    print("[Back Propagation]\n",
          "[delta_hidden:", w1,
          "delta_output:", w2,
          "loss:", b_loss[-1]
          )
    print("*" * 40)
    # ######################## 画图 ########################
    plt_loss(b_loss,
             x_label="epoch",
             y_label="loss",
             title="BP Algorithm"
             )
