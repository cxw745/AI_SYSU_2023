import copy
import math
import re
import random
import time
import matplotlib.pyplot as plt
from matplotlib import animation

city_list = list()  # 存储city类的实例


# 读取文件
def read_file():
    with open('test_bier127.tsp') as f:
        pat = re.compile(r'[0-9]+')
        test = f.readline()
        while test:
            city = re.findall(pat, test)
            city_list.append(City(int(city[0]), float(city[1]), float(city[2])))
            test = f.readline()


# 计算两个城市化之间的欧拉距离
def city_distance(city1, city2):
    return ((city1.x - city2.x) ** 2 + (city1.y - city2.y) ** 2) ** 0.5


def evaluate(cities):
    distance = 0
    for i in range(len(cities) - 1):
        distance += city_distance(city_list[cities[i]], city_list[cities[i + 1]])
    distance += city_distance(city_list[cities[0]], city_list[cities[-1]])
    return distance


# 定义城市类
class City:
    # 城市序号，横坐标、纵坐标
    def __init__(self, num, x, y):
        self.num = num
        self.x = x
        self.y = y


# 局部搜索
class Ls:
    def __init__(self):
        # ------参数
        # self.para_x = 1
        self.para_y = 1
        self.repeat = 0
        self.out_iter = 2000
        self.inn_iter = 100
        self.T_high = 2000
        self.T_start = 2000
        self.T_end = 1
        self.alpha = 0.998
        self.ans_len = len(city_list)  # 解的长度
        # ------距离记录-----
        self.best_distance = 0  # 全局最优解
        self.cur_distance = 0  # 当前解
        # -----解的序列记录集合记录-----
        self.cur_ans = [i for i in range(self.ans_len)]  # 当前解的序列
        self.best_ans = self.cur_ans  # 全局最优解的序列
        # -----每次解的序列记录-----
        self.cur_distance_list = list()  # 每次当前解的序列的 集合
        self.best_distance_list = list()  # 每次最好解的序列的 集合
        # -----初始化-----
        # 打乱
        random.shuffle(self.cur_ans)
        self.best_ans = self.cur_ans
        self.init_ans = self.cur_ans
        self.best_ans_list = [self.best_ans]
        # 得到初始化最好距离
        self.cur_distance = evaluate(self.cur_ans)
        self.best_distance = evaluate(self.best_ans)
        # 加入解的集合
        self.cur_distance_list.append(self.cur_distance)
        self.best_distance_list.append(self.best_distance)

    def climb(self):
        for i in range(self.out_iter):
            # while self.T_start >= self.T_end:
            for j in range(self.inn_iter):
                # 随机生成两个点
                idx1 = random.randint(0, self.ans_len - 2)
                idx2 = random.randint(idx1, self.ans_len - 1)
                '''
                # 交换两个点
                new_ans = copy.deepcopy(self.cur_ans)
                new_ans[idx1], new_ans[idx2] = new_ans[idx2], new_ans[idx1]
                new_distance = evaluate(new_ans)
                '''
                # 两个点之间的变成逆序列
                new_ans = copy.deepcopy(self.cur_ans)
                new_ans[idx1:idx2 + 1] = list(reversed(new_ans[idx1:idx2 + 1]))
                new_distance = evaluate(new_ans)

                # 如果新距离比最好的都小，就更新最好的距离，否则把当前解加入集合中
                if new_distance < self.best_distance:
                    self.best_distance = new_distance
                    self.best_ans = new_ans
                    self.best_distance_list.append(self.best_distance)
                    self.best_ans_list.append(self.best_ans)
                else:
                    self.best_distance_list.append(self.best_distance)

                # 与当前解比较，继续探索更好的解决
                if new_distance < self.cur_distance:
                    self.cur_distance = new_distance
                    self.cur_ans = new_ans
                    self.cur_distance_list.append(self.cur_distance)
                else:
                    self.cur_distance_list.append(self.cur_distance)
            print('循环次数为%d , 目前解为:%f , 全局最优解为:%f' % (
            (j + 1) * self.out_iter + j - 99, self.cur_distance, self.best_distance))

    # 画图
    def draw(self):
        fig, axes = plt.subplots(2, 2)
        ax_cur_ans_list = axes[0][0]
        ax_best_ans_list = axes[0][1]
        ax_init_ans = axes[1][0]
        ax_best_ans = axes[1][1]
        # -----得到城市在地图上的横纵坐标
        init_x, init_y = [], []
        best_x, best_y = [], []
        for init, best in zip(self.init_ans, self.best_ans):
            init_x.append(city_list[init].x)
            init_y.append(city_list[init].y)
            best_x.append(city_list[best].x)
            best_y.append(city_list[best].y)
        init_x.append(city_list[self.init_ans[0]].x)
        init_y.append(city_list[self.init_ans[0]].y)
        best_x.append(city_list[self.best_ans[0]].x)
        best_y.append(city_list[self.best_ans[0]].y)
        anneal_time = [i for i in range(len(self.cur_distance_list))]
        # 每次解的曲线图
        ax_cur_ans_list.set_title('cur_distance')
        ax_cur_ans_list.set_xlabel('annealing_time')
        ax_cur_ans_list.set_ylabel('cur_distance')
        ax_cur_ans_list.plot(anneal_time, self.cur_distance_list)
        # 每次最优解的曲线图
        ax_best_ans_list.set_title('best_distance')
        ax_best_ans_list.set_xlabel('annealing_time')
        ax_best_ans_list.set_ylabel('best_distance')
        ax_best_ans_list.plot(anneal_time, self.best_distance_list)
        # 初始解的路径可视化
        ax_init_ans.set_title('initial_ans')
        ax_init_ans.set_xlabel('x')
        ax_init_ans.set_ylabel('y')
        ax_init_ans.scatter(init_x, init_y)
        ax_init_ans.plot(init_x, init_y)
        # 最终解的路径可视化
        ax_best_ans.set_title('best_ans')
        ax_best_ans.set_xlabel('x')
        ax_best_ans.set_ylabel('y')
        ax_best_ans.scatter(best_x, best_y)
        ax_best_ans.plot(best_x, best_y)
        plt.tight_layout()
        fig.suptitle('LS')
        plt.show()

    # 画动图并保存
    def draw_gif(self):
        # 绘制路径变化动图并保存 注意fig不能共用
        fig2 = plt.figure(2)
        imgs = []
        # 间隔采样，最终取11帧数
        for i in range(0, len(self.best_ans_list), len(self.best_ans_list) // 50):
            best_x = []
            best_y = []
            for city in self.best_ans_list[i]:
                best_x.append(city_list[city].x)
                best_y.append(city_list[city].y)
            best_x.append(city_list[self.best_ans_list[i][0]].x)
            best_y.append(city_list[self.best_ans_list[i][0]].y)
            # 绘静态图 帧数
            plt.xlabel('x')
            plt.ylabel('y')
            im = plt.plot(best_x, best_y, marker='o', color='blue')
            imgs.append(im)

        # 将最终结果持续显示一段时间
        for i in range(200):
            best_x = []
            best_y = []
            for city in self.best_ans_list[-1]:
                best_x.append(city_list[city].x)
                best_y.append(city_list[city].y)
            best_x.append(city_list[self.best_ans_list[-1][0]].x)
            best_y.append(city_list[self.best_ans_list[-1][0]].y)
            # 绘静态图 帧数
            plt.xlabel('x')
            plt.ylabel('y')
            im = plt.plot(best_x, best_y, marker='o', color='blue')
            imgs.append(im)

        ani = animation.ArtistAnimation(fig2, imgs, interval=200, repeat_delay=1000)
        plt.suptitle('Path Changing')
        plt.show()
        ani.save('result\LS.gif', writer='pillow')

def main():
    read_file()
    ls = Ls()
    time1 = time.time()
    ls.climb()
    time2 = time.time()
    print(ls.best_ans)
    ls.draw()
    ls.draw_gif()
    print('running time:%f' % (time2 - time1))


if __name__ == '__main__':
    main()
