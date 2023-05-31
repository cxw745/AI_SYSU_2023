from optimizer import *
from network import *
import matplotlib.pyplot as plt


class Train:
    def __init__(self, network, x_train, t_train, x_test, t_test,
                 epochs, mini_batch_size,
                 optimizer, optimizer_lr, evaluate_sample_num_per_epoch):
        self.network = network
        self.x_train = x_train
        self.t_train = t_train
        self.x_test = x_test
        self.t_test = t_test
        self.epochs = epochs  # 多少个epochs 一个epochs当作一次输出
        self.batch_size = mini_batch_size  # 一个mini_batch大小
        self.evaluate_sample_num_per_epoch = evaluate_sample_num_per_epoch  # 小样本 用于快速评估网络性能
        self.optimizer_lr = optimizer_lr
        # 寻找最小梯度的方法
        optimizer_class_dict = {'sgd': SGD, 'momentum': Momentum,
                                'adagrad': AdaGrad}
        # optimizer_param是传入参数
        self.str_optimizer = optimizer
        self.optimizer = optimizer_class_dict[optimizer.lower()](lr=optimizer_lr)
        # 训练集大小
        self.train_size = x_train.shape[0]
        # 每个epoch有几个循环 epoch = train_size // mini_batch_size
        # 我们有246个训练样本 mini_batch_size设置为246 即一次循环一个epochs
        self.iter_per_epoch = max(self.train_size // mini_batch_size, 1)
        # 总的循环次数
        self.all_iter = int(self.epochs * self.iter_per_epoch)
        self.current_iter = 0
        self.current_epoch = 0

        self.train_loss_list = []
        self.train_acc_list = []
        self.test_acc_list = []

    def train_step(self):
        # 所有样本中随机抽取batch_size个样本量
        batch_mask = np.random.choice(self.train_size, self.batch_size)
        x_batch = self.x_train[batch_mask]
        t_batch = self.t_train[batch_mask]
        # 获取梯度并学习
        grads = self.network.bp_gradient(x_batch, t_batch)
        self.optimizer.update(self.network.params, grads)

        # 获得损失函数
        loss = self.network.loss(x_batch, t_batch)
        self.train_loss_list.append(loss)
        print("train loss:" + str(loss))

        # 每个epoch评估一次
        if self.current_iter % self.iter_per_epoch == 0:
            self.current_epoch += 1

            # 对网络性能进行快速评估 否则就是使用全部样本进行评估
            if not self.evaluate_sample_num_per_epoch is None:
                t = self.evaluate_sample_num_per_epoch
                x_train_sample, t_train_sample = self.x_train[:t], self.t_train[:t]
                x_test_sample, t_test_sample = self.x_test[:t], self.t_test[:t]
            else:
                x_train_sample, t_train_sample = self.x_train, self.t_train
                x_test_sample, t_test_sample = self.x_test, self.t_test
            # 计算准确率
            train_acc = self.network.accuracy(x_train_sample, t_train_sample)
            test_acc = self.network.accuracy(x_test_sample, t_test_sample)
            self.train_acc_list.append(train_acc)
            self.test_acc_list.append(test_acc)

            print(
                "=== epoch:" + str(self.current_epoch) + ", train acc:" + str(train_acc) + ", test acc:" + str(
                    test_acc) + " ===")
        self.current_iter += 1

    # 训练并测试准确率
    def train(self):
        for i in range(self.all_iter):
            self.train_step()
        test_acc = self.network.accuracy(self.x_test, self.t_test)
        print("=============== Final Test Accuracy ===============")
        print("test acc:" + str(test_acc))
        return test_acc

    def draw_acc_loss(self):
        epochs = np.arange(0, self.current_epoch)
        iters = np.arange(0, self.all_iter)
        fig, axes = plt.subplots(1, 2, figsize=(10, 6))

        axes[0].plot(iters, self.train_loss_list)
        axes[0].set_xlabel('iter'), axes[0].set_ylabel('train loss'), axes[0].set_title('loss')
        axes[0].legend(['loss'])

        axes[1].plot(epochs, self.train_acc_list)
        axes[1].plot(epochs, self.test_acc_list)
        axes[1].set_xlabel('epoch'), axes[1].set_ylabel('accuracy'), axes[1].set_title('accuracy')
        axes[1].legend(['train acc', 'test acc'])
        plt.savefig('./result/acc_loss', bbox_inches='tight', dpi=600)
        plt.show()


class Find:
    def __init__(self, network, train, x_train, t_train):
        # 打乱训练数据
        x_train, t_train = self.shuffle_dataset(x_train, t_train)
        # 取部分数据作为验证集
        validation_rate = 1
        validation_num = int(x_train.shape[0] * validation_rate)
        self.x_val = x_train[:validation_num]
        self.t_val = t_train[:validation_num]
        self.network = network
        self.train = train

    def draw_acc(self, x, y, x_label, y_label, title):
        plt.plot(x, y)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        plt.savefig('./result/' + title + '.png', dpi=600)
        plt.show()

    def find_lr(self, start, end, distance):
        # 创建不同学习率的多个训练方法
        # 根据给定创建范围数组
        x_list = np.arange(start, end, distance)  # lr
        y_list = []
        for x in x_list:
            train = Train(network=self.train.network, x_train=self.x_val, t_train=self.t_val,
                          x_test=self.train.x_test, t_test=self.train.t_test,
                          epochs=self.train.epochs, mini_batch_size=self.train.batch_size,
                          optimizer=self.train.str_optimizer, optimizer_lr=x,  # 这是变量
                          evaluate_sample_num_per_epoch=self.train.evaluate_sample_num_per_epoch)
            test_acc = train.train()
            y_list.append(test_acc)
        self.draw_acc(x_list, y_list, 'lr', 'acc', 'lr_acc')

    def find_weight_decay_lambda(self, start, end, distance):
        x_list = np.arange(start, end, distance)
        y_list = []
        for x in x_list:  # weight_decay_lambda
            network = Network(input_size=self.network.input_size, hidden_size_list=self.network.hidden_size_list,
                              output_size=self.network.output_size,
                              activation=self.network.activation, weight_init_std=self.network.weight_init_std,
                              weight_decay_lambda=x,  # 这是变量
                              use_dropout=self.network.use_dropout, dropout_ration=self.network.dropout_ration,
                              use_batchnorm=self.network.use_batchnorm)
            train = Train(network=network, x_train=self.x_val, t_train=self.t_val,
                          x_test=self.train.x_test, t_test=self.train.t_test,
                          epochs=self.train.epochs, mini_batch_size=self.train.batch_size,
                          optimizer=self.train.str_optimizer, optimizer_lr=self.train.optimizer_lr,
                          evaluate_sample_num_per_epoch=self.train.evaluate_sample_num_per_epoch)
            test_acc = train.train()
            y_list.append(test_acc)
        self.draw_acc(x_list, y_list, 'weight_decay_lambda', 'acc', 'weight_decay_lambda_acc')

    def find_dropout_ration(self, start, end, distance):
        x_list = np.arange(start, end, distance)
        y_list = []
        for x in x_list:  # dropout_ration
            network = Network(input_size=self.network.input_size, hidden_size_list=self.network.hidden_size_list,
                              output_size=self.network.output_size,
                              activation=self.network.activation, weight_init_std=self.network.weight_init_std,
                              weight_decay_lambda=self.network.weight_decay_lambda,
                              use_dropout=self.network.use_dropout, dropout_ration=x,  # 这是变量
                              use_batchnorm=self.network.use_batchnorm)
            train = Train(network=network, x_train=self.x_val, t_train=self.t_val,
                          x_test=self.train.x_test, t_test=self.train.t_test,
                          epochs=self.train.epochs, mini_batch_size=self.train.batch_size,
                          optimizer=self.train.str_optimizer, optimizer_lr=self.train.optimizer_lr,
                          evaluate_sample_num_per_epoch=self.train.evaluate_sample_num_per_epoch)
            test_acc = train.train()
            y_list.append(test_acc)
        print(x_list,y_list)
        self.draw_acc(x_list, y_list, 'dropout_ration', 'acc', 'dropout_ration_acc')

    def shuffle_dataset(self, x, t):
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
