import sys

inf = sys.maxsize
filename = 'Romania.txt'
diary = 'Romania_diary.txt'


class Romania:
    def __init__(self):
        with open(filename) as city_data:
            city_num, road_num = city_data.readline().split(' ')
            city_way = city_data.readlines()
            self.city_num = int(city_num)
            self.road_num = int(road_num)
            city_data.close()

        # 对外生成一个城市图 和 一些映射表
        # 因为是已经有的一个图，所以执行dijsktra算法需要的访问数组、距离数组、记录前驱的数组都作为类的成员
        # 这样每次查询的时候就可以利用已经记录的数据，查询速度会快很多
        self.city_graph = list(list(inf for j in range(self.city_num)) for i in range(self.city_num))
        self.city_title_index = dict()  # 城市首字母到下标，用来执行dijsktra算法
        self.city_index_name = dict()  # 城市下标到城市名，后面用来输出结果
        self.visit = list(0 for i in range(self.city_num))  # 访问
        self.distance = list(inf for i in range(self.city_num))  # 距离
        self.path = list(-1 for i in range(self.city_num))  # 前驱

        index = 0  # 作为映射的下标
        for way in city_way:  # 生成一个图，这里采用邻接矩阵的方法
            city_one, city_two, city_distance = way.split()
            city_distance = int(city_distance)
            if city_one.lower()[0] not in self.city_title_index:  # 首字母不在则添加
                city_name = city_one.lower()  # 小写规格化
                city_title = city_name[0]  # 记录城市的首字母
                self.city_title_index[city_title] = index  # 城市首字母映射为下标
                self.city_index_name[index] = city_one  # 下标映射为城市名
                index = index + 1
            if city_two.lower()[0] not in self.city_title_index:
                city_name = city_two.lower()  # 小写规格化
                city_title = city_name[0]  # 记录城市的首字母
                self.city_title_index[city_title] = index  # 城市首字母映射为下标
                self.city_index_name[index] = city_two  # 下标映射为城市名
                index = index + 1
            # 得到两个城市映射的下标，初始化邻接矩阵
            city_one_index = self.city_title_index[city_one.lower()[0]]
            city_two_index = self.city_title_index[city_two.lower()[0]]
            if city_distance < self.city_graph[city_one_index][city_two_index]:  # 初始化图
                self.city_graph[city_one_index][city_two_index] = city_distance
                self.city_graph[city_two_index][city_one_index] = city_distance

    def dij(self, start, end):  # dijskra算法，给出起点到终点，更新访问、距离、前驱数组
        for i in range(self.city_num):
            index = -1
            self.distance[start] = 0
            for j in range(self.city_num):
                if self.visit[j] == 0 and (index == -1 or self.distance[j] < self.distance[index]):
                    index = j

            if index == end:  # 如果找到了终点，则退出该函数
                return
            self.visit[index] = 1
            for j in range(self.city_num):
                if self.visit[j] == 0 and self.distance[index] + self.city_graph[index][j] < self.distance[j]:
                    self.distance[j] = self.distance[index] + self.city_graph[index][j]
                    self.path[j] = index

    def format_output(self, index_one, index_two):  # 格式化输出并且将结果记录到diary中
        self.dij(index_one, index_two)
        if self.distance[index_two] == inf:
            print("No way between the two cities!")
            return
        print("The shortest distance is: ", self.distance[index_two])
        print("The shortest path is: ", end='')
        my_path = list()
        index = index_two
        # 将前驱结点依次存入栈，利用栈FILO的性质就可以顺序输出路径了
        while index != -1:
            my_path.append(self.city_index_name[index])
            index = self.path[index]
        with open(diary, 'a') as f:
            f.write("The shortest distance is: " + str(self.distance[index_two]) + '\n')
            f.write("The shortest path is: ")
            f.close()
        while len(my_path):
            city = my_path.pop()
            print(city, end=' ')
            with open(diary, 'a') as f:
                f.write(city + ' ')
                f.close()
        print('\n')
        with open(diary, 'a') as f:
            f.write('\n')
            f.close()

    def format_input(self, city_one, city_two):  # 用来接受两个城市的输入，并且调用输出数组给出结果
        city_one = city_one.lower()[0]
        city_two = city_two.lower()[0]
        # 如果搜索的城市不存在，则返回
        if city_one not in self.city_title_index:
            print("No first city!")
            return
        if city_two not in self.city_title_index:
            print("No second city!")
            return
        index_one = self.city_title_index[city_one]
        index_two = self.city_title_index[city_two]
        self.format_output(index_one, index_two)

    def clear_diary(self):
        with open(diary, 'w') as f:
            f.write('')
