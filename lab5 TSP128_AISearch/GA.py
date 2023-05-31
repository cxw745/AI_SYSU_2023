import copy
import re
import random
import time
import matplotlib.pyplot as plt
from matplotlib import animation

# best answer bier127 : 118282


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


# 定义城市类
class City:
    # 城市序号，横坐标、纵坐标
    def __init__(self, num, x, y):
        self.num = num
        self.x = x
        self.y = y


# 个体 实值编码方式
class Individual:
    def __init__(self, gene_list=None):
        global city_list
        self.gene_len = len(city_list)

        # 随机生成序列 编码方式是实值编码
        if gene_list is None:
            gene_list = [i for i in range(0, self.gene_len)]
            random.shuffle(gene_list)
        self.gene_list = gene_list
        self.fitness = self.evaluate_fitness()

    # 适应度即以当前序列走完一个闭合曲线的路径之和
    def evaluate_fitness(self):
        # 计算个体适应度
        fitness = 0
        for city_idx in range(self.gene_len - 1):
            # 起始城市和目标城市的序号
            from_city_idx = self.gene_list[city_idx]
            to_city_idx = self.gene_list[city_idx + 1]
            fitness += city_distance(city_list[from_city_idx], city_list[to_city_idx])
        # 连接首尾【最后一个城市->起点城市】
        fitness += city_distance(city_list[self.gene_list[0]], city_list[self.gene_list[-1]])
        return fitness

    def __lt__(self, other):
        return self.fitness < other.fitness


# 遗传算法
class Ga:
    def __init__(self):
        global city_list
        self.mutate_prob = 0.4
        self.individual_num = 40
        self.generation_num = 20000
        self.gene_len = len(city_list)
        self.individual_list = None
        self.best_individual = None  # 最好的
        self.best_result_list = list()  # 存储每一代最好的个体的基因
        self.best_fitness_list = list()  # 存储每一代最好的个体的适应度适应度
        # self.repeat = 0
        # self.city_list = city_list  # 初始问题空间

    # 繁殖后代 交叉cross 变异mutate 筛选select
    def cross(self):
        new_gene = list()
        random.shuffle(self.individual_list) # 打乱种群中的排序，随机交配
        for ind_idx in range(0, len(self.individual_list) - 1, 2): # 两两个体交配
            # 得到两个交配个体的基因编码
            parent_gene1 = copy.deepcopy(self.individual_list[ind_idx].gene_list)
            parent_gene2 = copy.deepcopy(self.individual_list[ind_idx + 1].gene_list)

            # 随机生成两个基因点
            # randint[start,end]
            idx1 = random.randint(0, self.gene_len - 2)
            idx2 = random.randint(idx1, self.gene_len - 1)

            # 得到parent基因的 基因->下标 的字典
            gene_idx1 = {value: idx for idx, value in enumerate(parent_gene1)}
            gene_idx2 = {value: idx for idx, value in enumerate(parent_gene2)}

            # 交叉 方式为在自身的基因内对换，对换的对象用相同位置上另一个亲本的基因来确定，然后在自身寻找这个基因对换
            for i in range(idx1, idx2 + 1):
                if parent_gene1[i] == parent_gene2[i]:
                    continue
                value1, value2 = parent_gene1[i], parent_gene2[i]
                pos1, pos2 = gene_idx1[value2], gene_idx2[value1]
                # 对换基因
                parent_gene1[i], parent_gene1[pos1] = parent_gene1[pos1], parent_gene1[i]
                parent_gene2[i], parent_gene2[pos2] = parent_gene2[pos2], parent_gene2[i]
                # 更新字典
                gene_idx1[value1], gene_idx1[value2] = i, pos1
                gene_idx2[value2], gene_idx2[value1] = i, pos2
            new_gene.append(parent_gene1)
            new_gene.append(parent_gene2)
        return new_gene # 返回交配后的子代

    def mutate(self, new_gene):

        '''
        # 随机交换两个基因点
        for gene in new_gene:
            if random.random() <= self.mutate_prob:
                idx1 = random.randint(0, self.gene_len - 2)
                idx2 = random.randint(idx1 + 1, self.gene_len - 1)
                gene[idx1], gene[idx2] = gene[idx2], gene[idx1]
            self.individual_list.append(Individual(gene))
        '''


        # 两个基因点间逆序
        for gene in new_gene:
            if random.random() <= self.mutate_prob:
                idx1 = random.randint(0, self.gene_len - 2)
                idx2 = random.randint(idx1, self.gene_len - 1)
                gene[idx1:idx2 + 1] = list(reversed(gene[idx1:idx2 + 1]))
            self.individual_list.append(Individual(gene))


        '''
        # 两点插入 将随机基因片段插入尾部
        for gene in new_gene:
            if random.random() <= self.mutate_prob:
                idx1 = random.randint(0, self.gene_len - 2)
                idx2 = random.randint(idx1, self.gene_len - 1)
                # idx3 = random.randint(idx2 + 1, self.gene_len - 1)
                t = gene[:idx1]
                t += gene[idx2:]
                t += gene[idx1:idx2]
                gene = t
            self.individual_list.append(Individual(gene))
        '''
        '''
        # 彻底疯狂 完全翻转 随机
        for gene in new_gene:
            if random.random() <= self.mutate_prob:
                # gene = list(reversed(gene))
                random.shuffle(gene)
            self.individual_list.append(Individual(gene))
        '''
    def select(self):
        # 锦标赛筛选法 分小组，选择每个小组里最优秀的前几名个体组成下一代的种群
        group_num = 10  # 小组数
        group_size = 10  # 每小组人数
        group_winner = self.individual_num // group_num  # 每小组筛选出的individual【获胜者】数量
        winners = list()  # 锦标赛结果
        for i in range(group_num):
            group = list()
            for j in range(group_size):
                # 随机组成小组
                player = random.choice(self.individual_list)  # 随机选择参赛者
                group.append(player)
            group = sorted(group)  # 对本次锦标赛获胜者按适应度排序
            # 取出获胜者
            winners += group[:group_winner]
        self.individual_list = winners



    # 得到下一代的基因序列
    def get_next_gene_list(self):
        new_gene = self.cross()
        self.mutate(new_gene)
        self.select()
        # 找到最好的个体
        for individual in self.individual_list:
            if individual.fitness < self.best_individual.fitness:
                self.best_individual = individual

    # 创建初代种群 训练 返回每一代最好的个体，以字典存储，到时候以适应度大小排序
    def train(self):
        self.individual_list = [Individual() for i in range(self.individual_num)]
        self.best_individual = self.individual_list[0]
        for i in range(self.generation_num):
            print('generation %d best fitness %f ' % (i, self.best_individual.fitness))
            self.get_next_gene_list()
            self.best_result_list.append(self.best_individual.gene_list)
            self.best_fitness_list.append(self.best_individual.fitness)

    def draw(self):
        fig, axes = plt.subplots(2, 2)
        ax_cur_ans_list = axes[0][0]
        ax_best_ans_list = axes[0][1]
        ax_init_ans = axes[1][0]
        ax_best_ans = axes[1][1]
        # -----得到城市在地图上的横纵坐标
        init_x, init_y = [], []
        best_x, best_y = [], []
        for init, best in zip(self.best_result_list[0], self.best_result_list[-1]):
            init_x.append(city_list[init].x)
            init_y.append(city_list[init].y)
            best_x.append(city_list[best].x)
            best_y.append(city_list[best].y)
        init_x.append(city_list[self.best_result_list[0][0]].x)
        init_y.append(city_list[self.best_result_list[0][0]].y)
        best_x.append(city_list[self.best_result_list[-1][0]].x)
        best_y.append(city_list[self.best_result_list[-1][0]].y)
        anneal_time = [i for i in range(len(self.best_fitness_list))]
        # 每次解的曲线图
        ax_cur_ans_list.set_title('cur_distance')
        ax_cur_ans_list.set_xlabel('annealing_time')
        ax_cur_ans_list.set_ylabel('cur_distance')
        ax_cur_ans_list.plot(anneal_time, self.best_fitness_list)
        # 每次最优解的曲线图
        ax_best_ans_list.set_title('best_distance')
        ax_best_ans_list.set_xlabel('annealing_time')
        ax_best_ans_list.set_ylabel('best_distance')
        ax_best_ans_list.plot(anneal_time, self.best_fitness_list)
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
        fig.suptitle('GA')
        plt.show()

    # 画动图并保存
    def draw_gif(self):
        # 绘制路径变化动图并保存 注意fig不能共用
        fig2 = plt.figure(2)
        imgs = []
        # 间隔采样，最终取11帧数
        for i in range(0, len(self.best_result_list), len(self.best_result_list) // 50):
            best_x = []
            best_y = []
            for city in self.best_result_list[i]:
                best_x.append(city_list[city].x)
                best_y.append(city_list[city].y)
            best_x.append(city_list[self.best_result_list[i][0]].x)
            best_y.append(city_list[self.best_result_list[i][0]].y)
            # 绘静态图 帧数
            plt.xlabel('x')
            plt.ylabel('y')
            im = plt.plot(best_x, best_y, marker='o', color='blue')
            imgs.append(im)

        # 将最终结果持续显示一段时间
        for i in range(200):
            best_x = []
            best_y = []
            for city in self.best_result_list[-1]:
                best_x.append(city_list[city].x)
                best_y.append(city_list[city].y)
            best_x.append(city_list[self.best_result_list[-1][0]].x)
            best_y.append(city_list[self.best_result_list[-1][0]].y)
            # 绘静态图 帧数
            plt.xlabel('x')
            plt.ylabel('y')
            im = plt.plot(best_x, best_y, marker='o', color='blue')
            imgs.append(im)

        ani = animation.ArtistAnimation(fig2, imgs, interval=200, repeat_delay=1000)
        plt.suptitle('Path Changing')
        plt.show()
        ani.save('result\GA.gif', writer='pillow')

def main():
    # 读取文件
    read_file()
    # 遗传算法
    ga = Ga()
    # 训练 返回最终的字典结果
    time1 = time.time()
    ga.train()
    time2 = time.time()
    print('running time %f' % (time2 - time1))
    # 画图
    ga.draw()
    ga.draw_gif()


if __name__ == '__main__':
    main()
