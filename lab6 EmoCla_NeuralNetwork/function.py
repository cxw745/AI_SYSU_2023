import numpy as np


def step_function(x):
    y = x > 0
    return y.astype(np.int32)  # 支持numpy向量运算


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def relu(x):
    return np.maximum(0, x)


def identity_function(x):
    return x


def softmax(a):
    # -----可能会溢出-----
    # exp_a = np.exp(a)
    # sum_exp_a = np.sum(exp_a)
    # y = exp_a / sum_exp_a
    # -----溢出对策-----可以证明与原式得出的结果一模一样
    exp_a = np.exp(a - np.max(a))
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y


# 均方误差函数
def mean_square_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    batch_size = y.shape[0]
    return 0.5 * np.sum((y - t) ** 2) / batch_size


# 交叉熵误差函数
def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    batch_size = y.shape[0]
    delta = 1e-7  # 保护措施
    return -np.sum(t * np.log(y + delta)) / batch_size
