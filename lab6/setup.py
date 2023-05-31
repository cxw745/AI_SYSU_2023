import numpy as np

from encode import *
from network import *
from train import *

# 读取数据
x_train, t_train, t2_train = read_file('./dataset/Classification/train.txt')
x_test, t_test, t2_test = read_file('./dataset/Classification/test.txt')
# 读取文件方式
ban = True
function = 'one_hot'
x_train, t_train, x_test, t_test = word2vec(x_train=x_train, t_train=t_train, x_test=x_test, t_test=t_test,
                                            ban=ban, function=function)
# 转换成numpy数组
x_train, t_train, x_test, t_test = np.array(x_train), np.array(t_train), np.array(x_test), np.array(t_test)
# ----------------------------------------------------------
# 网络矩阵大小
input_size = len(x_train[0])
hidden_size_list = [100]
output_size = len(t_train[0])
# 网络参数
use_dropout = True  # dropout 方法
use_batchnorm = True  # 标准化
activation = 'relu'  # 激活函数
weight_init_std = activation  # 初始权值
weight_decay_lambda = 0.823  # 权值衰减系数
dropout_ration = 0.8  # dropout比例
# 创建网络
network = Network(input_size=input_size, hidden_size_list=hidden_size_list, output_size=output_size,
                  activation=activation, weight_init_std=weight_init_std, weight_decay_lambda=weight_decay_lambda,
                  use_dropout=use_dropout, dropout_ration=dropout_ration, use_batchnorm=use_batchnorm)
# -------------------------------------------------------------------
# 训练参数
epochs = 50
mini_batch_size = 40  # 40个 共246个数据 一个epochs会有两次循环 设置epochs为250 总共循环次数为500
optimizer = 'Adagrad'
optimizer_lr = 0.011
evaluate_sample_num_per_epoch = None
# 训练并绘制损失函数和正确率的变化图
train = Train(network=network, x_train=x_train, t_train=t_train, x_test=x_test, t_test=t_test,
              epochs=epochs, mini_batch_size=mini_batch_size,
              optimizer=optimizer, optimizer_lr=optimizer_lr,
              evaluate_sample_num_per_epoch=evaluate_sample_num_per_epoch)
# 训练结束 测试准确率 得到训练过程准确率 和 损失函数值的变化 绘图
test_acc = train.train()
train.draw_acc_loss()
# -----------------------------------------------------------------------
# 使用验证集调整超参数，寻找更优超参数
find = Find(network, train, x_train, t_train)
# find.find_lr(0.01,0.013,0.0005)
# find.find_dropout_ration(0.8,0.89,0.01)
# find.find_weight_decay_lambda(0.82,0.83,0.001)
'''
经过反复测试 得到较好的参数为
lr = 0.011
dropout_ration = 0.8
decay_lambda = 0.823
'''
