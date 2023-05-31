import numpy as np
class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr

    def update(self, params, grads):
        for key in params.keys():
            params[key] -= self.lr * grads[key]


class Momentum:
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None

    def update(self, params, grads):
        if self.v is None:
            self.v = {}
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)
        for key in params.keys():
            self.v[key] = self.momentum * self.v[key] - self.lr * grads[key]
            params[key] += self.v[key]


class AdaGrad:
    def __init__(self, lr=0.01):
        """
        构造函数，初始化 AdaGrad 参数

        :param lr: 学习率（Learning Rate），默认为 0.01
        """
        self.lr = lr
        self.h = None  # 保存以前所有梯度值的平方和

    def update(self, params, grads):
        """
        更新参数

        :param params: 需要更新的参数
        :param grads: 模型计算出的梯度
        """
        if self.h is None:
            # 如果还没有记录过前面的梯度平方和，则初始化
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)
        else:
            # 如果已经有了前面的梯度平方和，则累加当前梯度平方和
            for key in params.keys():
                # 更新梯度平方和
                self.h[key] += grads[key] * grads[key]

                # 计算学习率，这里使用 AdaGrad 的自适应学习率方法
                lr = self.lr / (np.sqrt(self.h[key]) + 1e-7)

                # 更新参数
                params[key] -= lr * grads[key]


class Adam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.m = None
        self.v = None

    def update(self, params, grads):
        if self.m is None:
            self.m, self.v = {}, {}
            for key, val in params.items():
                self.m[key] = np.zeros_like(val)
                self.v[key] = np.zeros_like(val)

        self.iter += 1
        lr_t = self.lr * np.sqrt(1.0 - self.beta2 ** self.iter) / (1.0 - self.beta1 ** self.iter)

        for key in params.keys():
            self.m[key] += (1 - self.beta1) * (grads[key] - self.m[key])
            self.v[key] += (1 - self.beta2) * (grads[key] ** 2 - self.v[key])

            params[key] -= lr_t * self.m[key] / (np.sqrt(self.v[key]) + 1e-7)
