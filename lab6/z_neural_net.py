from collections import OrderedDict

import numpy as np
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D


# 收集并理解数据
# 处理数据：清洗和整理
# 训练模型
# 测试模型
# 提高准确率
def read_file(file):
    # 返回数据集 的 单词和标签
    words = []
    label_name = []
    label_idx = []
    with open(file) as f:
        data_title = f.readline()
        print('The data name is %s' % data_title)
        dataset = f.readlines()
        for data in dataset:
            tmp_data = data.replace('\n', '').split(' ')
            words.append(tmp_data[3:])
            label_name.append(tmp_data[2])
            label_idx.append(int(tmp_data[1]))
    return words, label_name, label_idx


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


def tf_idf(word, doc, docs):
    # word是要计算的单词，doc是当前文档存有所有单词，docs是所有的文档
    word_doc = sum(1 for doc in docs if word in docs)  # 计算所有文档中包含该单词的文档数
    tf = doc.count(word) / len(doc)
    idf = np.log(len(docs) / (word_doc + 1))
    return tf * idf


def numerical_diff(f, x):
    h = 1e-4
    return (f(x + h) - f(x - h)) / (2 * h)


def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)
    for idx in range(x.shape[0]):  # 求第idx个变量的偏导
        # print(x[idx])
        tmp_val = x[idx]
        x[idx] = tmp_val + h
        fxh1 = f(x)
        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2 * h)

        x[idx] = tmp_val
    return grad


# # 梯度下降法，寻找损失函数最小值的方向
# def gradient_descent(f, init_x, lr=0.01, step_num=100):
#     x = init_x
#     for i in range(step_num):
#         grad = numerical_gradient(f, x)
#         x -= lr * grad
#     return x
#
#
# class SimpleNet:
#     def __init__(self):
#         self.W = np.random.randn(2, 3)
#
#     def predict(self, x):
#         return np.dot(x, self.W)
#
#     def loss(self, x, t):
#         z = self.predict(x)
#         y = softmax(z)
#         loss = mean_square_error(y, t)
#         return loss


# batch_nomalization 将Affine层输出的数据
class BatchNormalization:
    def __init__(self, gamma, beta, momentum=0.9, running_mean=None, running_var=None):
        self.gamma = gamma
        self.beta = beta
        self.momentum = momentum
        self.input_shape = None  # Conv层的情况下为4维，全连接层的情况下为2维

        # 测试时使用的平均值和方差
        self.running_mean = running_mean
        self.running_var = running_var

        # backward时使用的中间数据
        self.batch_size = None
        self.xc = None
        self.std = None
        self.dgamma = None
        self.dbeta = None

    def forward(self, x, train_flg=True):
        self.input_shape = x.shape
        if x.ndim != 2:
            N, C, H, W = x.shape
            x = x.reshape(N, -1)

        out = self.__forward(x, train_flg)

        return out.reshape(*self.input_shape)

    def __forward(self, x, train_flg):
        if self.running_mean is None:
            N, D = x.shape
            self.running_mean = np.zeros(D)
            self.running_var = np.zeros(D)

        if train_flg:
            mu = x.mean(axis=0)
            xc = x - mu
            var = np.mean(xc ** 2, axis=0)
            std = np.sqrt(var + 10e-7)
            xn = xc / std

            self.batch_size = x.shape[0]
            self.xc = xc
            self.xn = xn
            self.std = std
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mu
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
        else:
            xc = x - self.running_mean
            xn = xc / ((np.sqrt(self.running_var + 10e-7)))

        out = self.gamma * xn + self.beta
        return out

    def backward(self, dout):
        if dout.ndim != 2:
            N, C, H, W = dout.shape
            dout = dout.reshape(N, -1)

        dx = self.__backward(dout)

        dx = dx.reshape(*self.input_shape)
        return dx

    def __backward(self, dout):
        dbeta = dout.sum(axis=0)
        dgamma = np.sum(self.xn * dout, axis=0)
        dxn = self.gamma * dout
        dxc = dxn / self.std
        dstd = -np.sum((dxn * self.xc) / (self.std * self.std), axis=0)
        dvar = 0.5 * dstd / self.std
        dxc += (2.0 / self.batch_size) * self.xc * dvar
        dmu = np.sum(dxc, axis=0)
        dx = dxc - dmu / self.batch_size

        self.dgamma = dgamma
        self.dbeta = dbeta

        return dx


class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std):
        # 初始化权重
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size) / np.sqrt(train_size)
        self.params['b1'] = weight_init_std * np.random.randn(hidden_size) / np.sqrt(train_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, hidden_size) / np.sqrt(hidden_size)
        self.params['b2'] = weight_init_std * np.random.randn(hidden_size) / np.sqrt(hidden_size)
        self.params['W3'] = weight_init_std * np.random.randn(hidden_size, output_size) / np.sqrt(hidden_size)
        self.params['b3'] = weight_init_std * np.random.randn(output_size) / np.sqrt(hidden_size)

        # ------反向传播误差计算部分
        # 生成层
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['BatchNorm1'] = BatchNormalization(1, 0)  # 标准化层
        self.layers['Relu1'] = Relu()
        self.layers['Dropout1'] = Dropout(0.3)  # Dropout方法

        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
        self.layers['BatchNorm2'] = BatchNormalization(1, 0)  # 标准化层
        self.layers['Relu2'] = Relu()
        self.layers['Dropout2'] = Dropout(0.3)  # Dropout方法

        self.layers['Affine3'] = Affine(self.params['W3'], self.params['b3'])
        self.lastLayer = SoftmaxWithLoss()

    def predict(self, x, train_flg):
        # 梯度计算法 面向过程
        # W1, W2 = self.params['W1'], self.params['W2']
        # b1, b2 = self.params['b1'], self.params['b2']
        #
        # a1 = np.dot(x, W1) + b1
        # z1 = sigmoid(a1)
        # a2 = np.dot(z1, W2) + b2
        # y = softmax(a2)
        # return y
        # ---------------------------------------------------
        # 反向传播 用面向对象的方式实现层与层之间的传播
        for layer in self.layers.values():
            x = layer.forward(x, train_flg)
        return x

    # x:输入数据, t:监督数据
    def loss(self, x, t):
        y = self.predict(x, True)
        # return cross_entropy_error(y, t)

        # 权值衰减
        weight_decay = 0
        W1 = self.params['W1']
        W2 = self.params['W2']
        W3 = self.params['W3']
        weight_decay += 0.5 * 0.1 * (np.sum(W1 ** 2) + np.sum(W2 ** 2) + np.sum(W3 ** 2))

        loss_ = self.lastLayer.forward(y, t)
        return loss_ + weight_decay

    def accuracy(self, x, t, train_flg):
        y = self.predict(x, train_flg)
        print('original predict\n', y, '\n', 'original teacher\n', t)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)
        print('label predict\n', y, '\n', 'label teacher\n', t)
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    # x:输入数据, t:监督数据
    # 数值微分计算梯度
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)
        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
        grads['W3'] = numerical_gradient(loss_W, self.params['W3'])
        grads['b3'] = numerical_gradient(loss_W, self.params['b3'])
        # print('numerical_grads:')
        # print('W1 grad:', np.mean(grads['W1']))
        # print('b1 grad:', np.mean(grads['b1']))
        # print('W2 grad:', np.mean(grads['W2']))
        # print('b2 grad:', np.mean(grads['b2']))
        return grads

    # 反向传播计算梯度
    def backpropagation_gradient(self, x, t):
        # backward
        dout = self.lastLayer.backward(self.loss(x, t))
        layers = list(self.layers.values())
        layers.reverse()

        for layer in layers:
            dout = layer.backward(dout)
        # 每一层的梯度
        grads = {}
        grads['W1'] = self.layers['Affine1'].dW
        grads['b1'] = self.layers['Affine1'].db
        grads['W2'] = self.layers['Affine2'].dW
        grads['b2'] = self.layers['Affine2'].db
        grads['W3'] = self.layers['Affine3'].dW
        grads['b3'] = self.layers['Affine3'].db
        # print('backprogagation_grads:')
        # print('W1 grad:', np.mean(grads['W1']))
        # print('b1 grad:', np.mean(grads['b1']))
        # print('W2 grad:', np.mean(grads['W2']))
        # print('b2 grad:', np.mean(grads['b2']))
        return grads


# 反向传播误差算法实现 每一层都进行归一化
# ReLU 层的作用就像电路中的开关一样
class Relu:
    def __init__(self):
        self.mask = None  # 把正向传播时的输入 x的元素中小于等于0的地方保存为 True，其他地方（大于0的元素）保存为 False。

    def forward(self, x, a=0):  # 输入的是numpy数组
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        # print('this is Relu backward,next is affine\n', np.average(dx,axis=0))
        return dx


# sigmoid 激活函数层
class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out
        return out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out
        return dx


# 仿射层
class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None

    def forward(self, x, a=0):
        self.x = x
        out = np.dot(x, self.W) + self.b
        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        # print('this is Affine backward,next is Relu\n', np.average(dx, axis=0))
        return dx


# 损失函数层 最后一层
class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size
        # print('this is loss backward,next is affine\n', np.average(dx,axis=0))
        return dx


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
        self.lr = lr
        self.h = None  # 保存以前所有梯度值的平方和

    def update(self, params, grads):
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)
        else:
            for key in params.keys():
                self.h[key] += grads[key] * grads[key]
                params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)


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
            # self.m[key] = self.beta1*self.m[key] + (1-self.beta1)*grads[key]
            # self.v[key] = self.beta2*self.v[key] + (1-self.beta2)*(grads[key]**2)
            self.m[key] += (1 - self.beta1) * (grads[key] - self.m[key])
            self.v[key] += (1 - self.beta2) * (grads[key] ** 2 - self.v[key])

            params[key] -= lr_t * self.m[key] / (np.sqrt(self.v[key]) + 1e-7)

            # unbias_m += (1 - self.beta1) * (grads[key] - self.m[key]) # correct bias
            # unbisa_b += (1 - self.beta2) * (grads[key]*grads[key] - self.v[key]) # correct bias
            # params[key] += self.lr * unbias_m / (np.sqrt(unbisa_b) + 1e-7)


class Dropout:
    def __init__(self, dropout_ratio=0.3):
        self.dropout_ratio = dropout_ratio
        self.mask = None

    def forward(self, x, train_flg=True):
        if train_flg:  # 训练的时候随机删除神经元
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            return x * self.mask
        else:  # 测试的时候乘上删除比例，实质上就是模拟集成神经网络学习
            return x * (1.0 - self.dropout_ratio)

    def backward(self, dout):
        return dout * self.mask


def simple_net_example():
    X = np.array([1.0, 0.5])
    W1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    B1 = np.array([0.1, 0.2, 0.3])
    print(W1.shape)  # (2, 3)
    print(X.shape)  # (2,)
    print(B1.shape)  # (3,)
    A1 = np.dot(X, W1) + B1
    Z1 = sigmoid(A1)
    print(A1)  # [0.3, 0.7, 1.1]
    print(Z1)  # [0.57444252, 0.66818777, 0.75026011

    W2 = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    B2 = np.array([0.1, 0.2])
    print(Z1.shape)  # (3,)
    print(W2.shape)  # (3, 2)
    print(B2.shape)  # (2,)
    A2 = np.dot(Z1, W2) + B2
    Z2 = sigmoid(A2)

    W3 = np.array([[0.1, 0.3], [0.2, 0.4]])
    B3 = np.array([0.1, 0.2])
    A3 = np.dot(Z2, W3) + B3
    Y = identity_function(A3)  # 或者Y = A3


# -----------------代码实现-------------------
'''
训练数据用于参数（权重和偏置）的学习，验证数据用于超参数的性
能评估。为了确认泛化能力，要在最后使用（比较理想的是只用一次）
测试数据。
'''
x_train = []
t_train = []

x_test = []
t_test = []
train_loss_list = []
train_acc_list = []
test_acc_list = []
# 平均每个epoch的重复次数

# 超参数
iters_num = 5000
learning_rate = 0.0002
# 读取文档得到 词 和 标签
words_train, emo_name_train, emo_number_train = read_file('./dataset/Classification/train.txt')
words_test, emo_name_test, emo_number_test = read_file('./dataset/Classification/test.txt')


# 给文件编码
def word2vec():
    # 建立词表 禁止停用词
    tmp = ['a, able, about, across, after, all, almost, also, am, among, an, and, any, are, as, at, be, because, '
           'been, but, by, can, cannot, could, dear, did, do, does, either, else, ever, every, for, from, get, '
           'got, had, has, have, he, her, hers, him, his, how, however, i, if, in, into, is, it, its, just, '
           'least, let, like, likely, may, me, might, most, must, my, neither, no, nor, not, of, off, often, on, '
           'only, or, other, our, own, rather, said, say, says, she, should, since, so, some, than, that, the, '
           'their, them, then, there, these, they, this, tis, to, too, twas, us, wants, was, we, were, what, '
           'when, where, which, while, who, whom, why, will, with, would, yet, you, your, a, about, above, across, '
           'actually, add, ago, all, almost, along, already, also, although, always, am, among, an, and, another, '
           'any, anyone, anything, anyway, anywhere, are, aren, around, as, ask, at, away, b, back, be, because, '
           'been, before, being, below, best, better, between, big, bit, both, but, by, c, called, can, came, cannot, '
           'case, certain, certainly, clear, clearly, come, common, concerning, consequently, consider, could, '
           'couldn, d, date, day, did,  different, do, does, doesn, doing, done, don, down, due, during, e, each, '
           'early, either, else, end, enough, especially, even, ever, every, everyone, everything, example, except, '
           'f, face, fact, far, few, find, first, for, form, four, from, full, further, g, general, get, give, go, '
           'going, good, got, great, h, had, hardly, has, hasn, have, having, he, her, here, hi, high, him, himself, '
           'his, hit, hold, home, how, however, i, if, in, indeed, information, interest, into, is, isn, issue, it, '
           'its, it,s, itself, j, just, k, keep, kind, know, known, l, large, last, late, later, least, left, less, '
           'let, letter, likely, long, look, m, made, make, many, may, maybe, me, mean, meets, member, mention, '
           'might, mine, miss, more, most, mostly, much, must, my, myself, n, name, namely, need, never, new, next, '
           'nine, no, nobody, none, nor, normally, not, nothing, now, o, of, off, often, oh, ok, okay, old, on, once, '
           'one, only, onto, or, other, our, ours, out, over, own, p, part, particular, past, people, perhaps, '
           'person, place, plus, point, possible, present, probably, program, provide, put, q, question, quickly, '
           'quite, r, rather, really, recent, regarding, regards, related, relatively, request, right, result, '
           'return, s, said, same, saw, say, saying, says, second, see, seem, seemed, seeming, seems, seen, self, '
           'send, sent, several, shall, she, should, shouldn, show, showed, shown, shows, side, since, six, small, '
           'so, some, somebody, somehow, someone, something, sometime, sometimes, somewhat, somewhere, soon, sorry, '
           'specific, specified, specify, still, stop, such, sure, t, take, taken, taking, tends, term, than, that, '
           'thats, the, their, theirs, them, themselves, then, there, therefore, these, they, thing, things, think, '
           'third, this, those, three, through, thus, time, to, together, too, took, toward, turned, two, u, under, '
           'understood, unfortunately, unless, unlike, unlikely, until, up, upon, us, use, used, useful, usually, v, '
           'value, various, very, via, video, view, w, want, was, wasn, way, we, well, were, what, whatever, when, '
           'whenever, where, whether, which, while, who, whole, whom, whose, why, will, with, within, without, won, '
           'work, would, wouldn, x, y, year, yes, yet, you, your, yours, yourself, yourselves, z']
    ban_words = set(tmp[0].replace(' ', '').split(','))

    dictionary = set()
    for words in words_train:
        for word in words:
            if word not in ban_words:
                dictionary.add(word)
    dictionary = list(dictionary)
    # 建立标签表
    label = set()
    for emo in emo_name_train:
        label.add(emo)
    label = list(label)

    # 处理训练集 得到one—hot表示
    # for words in words_train:
    #     vec = [0 for i in range(len(dictionary))]
    #     for word in words:
    #         if word in dictionary:
    #             vec[dictionary.index(word)] = 1
    #     x_train.append(vec)
    # tf_idf表示
    for words in words_train:
        vec = [0 for i in range(len(dictionary))]
        for word in words:
            if word in dictionary:
                vec[dictionary.index(word)] = tf_idf(word, words, words_train)
        x_train.append(vec)

    for emo in emo_name_train:
        vec = [0 for i in range(len(label))]
        if emo in label:
            vec[label.index(emo)] = 1
        t_train.append(vec)
    # 处理测试集 one_hot
    # for words in words_test:
    #     vec = [0 for i in range(len(dictionary))]
    #     for word in words:
    #         if word in dictionary:
    #             vec[dictionary.index(word)] = 1
    #     x_test.append(vec)

    # tf_idf
    for words in words_test:
        vec = [0 for i in range(len(dictionary))]
        for word in words:
            if word in dictionary:
                vec[dictionary.index(word)] = tf_idf(word, words, words_test)
        x_test.append(vec)

    for emo in emo_name_test:
        vec = [0 for i in range(len(label))]
        if emo in label:
            vec[label.index(emo)] = 1
        t_test.append(vec)


# 给数据预处理 划分为训练数据、测试数据、验证数据三部分，用验证数据调整超参数

word2vec()

# 转换成numpy数组 这里使用标签的名字来作为答案
x_train = np.array(x_train)
t_train = np.array(t_train)
x_test = np.array(x_test)
t_test = np.array(t_test)


# 分割训练集 用一部分来作为验证数据
# 打乱训练数据
def shuffle_dataset(x, t):
    """打乱数据集

    Parameters
    ----------
    x : 训练数据
    t : 监督数据

    Returns
    -------
    x, t : 打乱的训练数据和监督数据
    """
    permutation = np.random.permutation(x.shape[0])
    x = x[permutation, :] if x.ndim == 2 else x[permutation, :, :, :]
    t = t[permutation]

    return x, t


# 数据集参数
train_size = len(x_train)
test_size = len(x_test)
# 权重矩阵大小参数
input_size = len(x_train[0])
hidden_size = 100
output_size = len(t_train[0])
# mini_batch
batch_size = 100
iter_per_epoch = train_size // batch_size
print(train_size)
print(input_size, hidden_size, output_size)
network = TwoLayerNet(input_size, hidden_size, output_size, 4)
optimizer = AdaGrad(0.005)


class MultiLayerNet:
    """全连接的多层神经网络

    Parameters
    ----------
    input_size : 输入大小（MNIST的情况下为784）
    hidden_size_list : 隐藏层的神经元数量的列表（e.g. [100, 100, 100]）
    output_size : 输出大小（MNIST的情况下为10）
    activation : 'relu' or 'sigmoid'
    weight_init_std : 指定权重的标准差（e.g. 0.01）
        指定'relu'或'he'的情况下设定“He的初始值”
        指定'sigmoid'或'xavier'的情况下设定“Xavier的初始值”
    weight_decay_lambda : Weight Decay（L2范数）的强度
    """

    def __init__(self, input_size, hidden_size_list, output_size,
                 activation='relu', weight_init_std='relu', weight_decay_lambda=0):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size_list = hidden_size_list
        self.hidden_layer_num = len(hidden_size_list)
        self.weight_decay_lambda = weight_decay_lambda
        self.params = {}

        # 初始化权重
        self.__init_weight(weight_init_std)

        # 生成层
        activation_layer = {'sigmoid': Sigmoid, 'relu': Relu}
        self.layers = OrderedDict()
        for idx in range(1, self.hidden_layer_num + 1):
            self.layers['Affine' + str(idx)] = Affine(self.params['W' + str(idx)],
                                                      self.params['b' + str(idx)])
            self.layers['Activation_function' + str(idx)] = activation_layer[activation]()

        idx = self.hidden_layer_num + 1
        self.layers['Affine' + str(idx)] = Affine(self.params['W' + str(idx)],
                                                  self.params['b' + str(idx)])

        self.last_layer = SoftmaxWithLoss()

    def __init_weight(self, weight_init_std):
        """设定权重的初始值

        Parameters
        ----------
        weight_init_std : 指定权重的标准差（e.g. 0.01）
            指定'relu'或'he'的情况下设定“He的初始值”
            指定'sigmoid'或'xavier'的情况下设定“Xavier的初始值”
        """
        all_size_list = [self.input_size] + self.hidden_size_list + [self.output_size]
        for idx in range(1, len(all_size_list)):
            scale = weight_init_std
            if str(weight_init_std).lower() in ('relu', 'he'):
                scale = np.sqrt(2.0 / all_size_list[idx - 1])  # 使用ReLU的情况下推荐的初始值
            elif str(weight_init_std).lower() in ('sigmoid', 'xavier'):
                scale = np.sqrt(1.0 / all_size_list[idx - 1])  # 使用sigmoid的情况下推荐的初始值

            self.params['W' + str(idx)] = scale * np.random.randn(all_size_list[idx - 1], all_size_list[idx])
            self.params['b' + str(idx)] = np.zeros(all_size_list[idx])

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)

        return x

    def loss(self, x, t):
        """求损失函数

        Parameters
        ----------
        x : 输入数据
        t : 教师标签

        Returns
        -------
        损失函数的值
        """
        y = self.predict(x)
        # 权值衰减
        weight_decay = 0
        for idx in range(1, self.hidden_layer_num + 2):
            W = self.params['W' + str(idx)]
            weight_decay += 0.5 * self.weight_decay_lambda * np.sum(W ** 2)

        return self.last_layer.forward(y, t) + weight_decay

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1: t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def numerical_gradient(self, x, t):
        """求梯度（数值微分）

        Parameters
        ----------
        x : 输入数据
        t : 教师标签

        Returns
        -------
        具有各层的梯度的字典变量
            grads['W1']、grads['W2']、...是各层的权重
            grads['b1']、grads['b2']、...是各层的偏置
        """
        loss_W = lambda W: self.loss(x, t)

        grads = {}
        for idx in range(1, self.hidden_layer_num + 2):
            grads['W' + str(idx)] = numerical_gradient(loss_W, self.params['W' + str(idx)])
            grads['b' + str(idx)] = numerical_gradient(loss_W, self.params['b' + str(idx)])

        return grads

    def gradient(self, x, t):
        """求梯度（误差反向传播法）

        Parameters
        ----------
        x : 输入数据
        t : 教师标签

        Returns
        -------
        具有各层的梯度的字典变量
            grads['W1']、grads['W2']、...是各层的权重
            grads['b1']、grads['b2']、...是各层的偏置
        """
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 设定
        grads = {}
        for idx in range(1, self.hidden_layer_num + 2):
            grads['W' + str(idx)] = self.layers['Affine' + str(idx)].dW + self.weight_decay_lambda * self.layers[
                'Affine' + str(idx)].W
            grads['b' + str(idx)] = self.layers['Affine' + str(idx)].db

        return grads


class Trainer:
    """进行神经网络的训练的类
    """

    def __init__(self, network, x_train, t_train, x_test, t_test,
                 epochs=20, mini_batch_size=100,
                 optimizer='SGD', optimizer_param={'lr': 0.01},
                 evaluate_sample_num_per_epoch=None, verbose=True):
        self.network = network
        self.verbose = verbose
        self.x_train = x_train
        self.t_train = t_train
        self.x_test = x_test
        self.t_test = t_test
        self.epochs = epochs
        self.batch_size = mini_batch_size
        self.evaluate_sample_num_per_epoch = evaluate_sample_num_per_epoch

        # optimzer
        optimizer_class_dict = {'sgd': SGD, 'momentum': Momentum,
                                'adagrad': AdaGrad, 'adam': Adam}
        self.optimizer = optimizer_class_dict[optimizer.lower()](**optimizer_param)

        self.train_size = x_train.shape[0]
        self.iter_per_epoch = max(self.train_size / mini_batch_size, 1)
        self.max_iter = int(epochs * self.iter_per_epoch)
        self.current_iter = 0
        self.current_epoch = 0

        self.train_loss_list = []
        self.train_acc_list = []
        self.test_acc_list = []

    def train_step(self):
        batch_mask = np.random.choice(self.train_size, self.batch_size)
        x_batch = self.x_train[batch_mask]
        t_batch = self.t_train[batch_mask]

        grads = self.network.gradient(x_batch, t_batch)
        self.optimizer.update(self.network.params, grads)

        loss = self.network.loss(x_batch, t_batch)
        self.train_loss_list.append(loss)
        if self.verbose: print("train loss:" + str(loss))

        if self.current_iter % self.iter_per_epoch == 0:
            self.current_epoch += 1

            x_train_sample, t_train_sample = self.x_train, self.t_train
            x_test_sample, t_test_sample = self.x_test, self.t_test
            if not self.evaluate_sample_num_per_epoch is None:
                t = self.evaluate_sample_num_per_epoch
                x_train_sample, t_train_sample = self.x_train[:t], self.t_train[:t]
                x_test_sample, t_test_sample = self.x_test[:t], self.t_test[:t]

            train_acc = self.network.accuracy(x_train_sample, t_train_sample)
            test_acc = self.network.accuracy(x_test_sample, t_test_sample)
            self.train_acc_list.append(train_acc)
            self.test_acc_list.append(test_acc)

            if self.verbose: print(
                "=== epoch:" + str(self.current_epoch) + ", train acc:" + str(train_acc) + ", test acc:" + str(
                    test_acc) + " ===")
        self.current_iter += 1

    def train(self):
        for i in range(self.max_iter):
            self.train_step()

        test_acc = self.network.accuracy(self.x_test, self.t_test)

        if self.verbose:
            print("=============== Final Test Accuracy ===============")
            print("test acc:" + str(test_acc))


x_train, t_train = shuffle_dataset(x_train, t_train)
# 分割验证数据
validation_rate = 0.20
validation_num = int(x_train.shape[0] * validation_rate)
x_val = x_train[:validation_num]
t_val = t_train[:validation_num]
x_train = x_train[validation_num:]
t_train = t_train[validation_num:]


def __train(lr, weight_decay, epocs=50):
    network = MultiLayerNet(input_size, [hidden_size, hidden_size],
                            output_size, weight_decay_lambda=weight_decay)
    trainer = Trainer(network, x_train, t_train, x_val, t_val,
                      epochs=epocs, mini_batch_size=100,
                      optimizer='adagrad', optimizer_param={'lr': lr}, verbose=False)
    trainer.train()

    return trainer.test_acc_list, trainer.train_acc_list


# 超参数的随机搜索======================================
optimization_trial = 100
results_val = {}
results_train = {}
for _ in range(optimization_trial):
    # 指定搜索的超参数的范围===============
    weight_decay = 10 ** np.random.uniform(-8, -4)
    lr = 10 ** np.random.uniform(-6, -2)
    # ================================================

    val_acc_list, train_acc_list = __train(lr, weight_decay)
    print("val acc:" + str(val_acc_list[-1]) + " | lr:" + str(lr) + ", weight decay:" + str(weight_decay))
    key = "lr:" + str(lr) + ", weight decay:" + str(weight_decay)
    results_val[key] = val_acc_list
    results_train[key] = train_acc_list

# 绘制图形========================================================
print("=========== Hyper-Parameter Optimization Result ===========")
graph_draw_num = 20
col_num = 5
row_num = int(np.ceil(graph_draw_num / col_num))
i = 0

for key, val_acc_list in sorted(results_val.items(), key=lambda x: x[1][-1], reverse=True):
    print("Best-" + str(i + 1) + "(val acc:" + str(val_acc_list[-1]) + ") | " + key)

    plt.subplot(row_num, col_num, i + 1)
    plt.title("Best-" + str(i + 1))
    plt.ylim(0.0, 1.0)
    if i % 5: plt.yticks([])
    plt.xticks([])
    x = np.arange(len(val_acc_list))
    plt.plot(x, val_acc_list)
    plt.plot(x, results_train[key], "--")
    i += 1

    if i >= graph_draw_num:
        break

plt.show()

# ---------------------------------------
# 梯度确认 可能是反向传播的时候梯度爆炸了
x_batch = x_train[:2]
t_batch = t_train[:2]
grad_numerical = network.numerical_gradient(x_batch, t_batch)
grad_backprop = network.backpropagation_gradient(x_batch, t_batch)
# 求各个权重的绝对误差的平均值
for key in grad_numerical.keys():
    diff = np.average(np.abs(grad_backprop[key] - grad_numerical[key]))
    print('numerical' + key + ':', np.average(grad_numerical[key]))
    print('backprop' + key + ':', np.average(grad_backprop[key]))
    print(key + ":" + str(diff))
    print('\n')
# ---------------------------------------
for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)

    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # 计算梯度
    # grads = network.numerical_gradient(x_batch, t_batch)
    grads = network.backpropagation_gradient(x_batch, t_batch)  # 高速版!
    # 更新参数
    optimizer.update(network.params, grads)
    # 记录学习过程
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)
    print('当前损失函数值为%f' % loss)
    # 计算每个epoch的识别精度
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train, True)
        test_acc = network.accuracy(x_test, t_test, False)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc))
    print('第几次循环？%d' % i)

fig, ax = plt.subplots(1, 2)
# 画第1个图：折线图
ax[0].plot(range(iters_num), train_loss_list, label='loss')

# 画第2个图：散点图
ax[1].plot(range(len(train_acc_list)), train_acc_list, label='train')
ax[1].plot(range(len(test_acc_list)), test_acc_list, label='test')

ax[0].legend()
ax[1].legend()
plt.show()
