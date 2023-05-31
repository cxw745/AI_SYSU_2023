import copy
import time
import sys
from heuristic import *

PATH = list()  # 路径记录
Nodes_num = -1


# 计算逆序数的奇偶性
def inversion(cur_state):
    cnt = 0
    temp = list()
    for row in cur_state:
        for num in row:
            if num == 0:
                continue
            temp.append(num)
    for i in range(15):
        for j in range(i + 1, 15):
            if temp[i] > temp[j]:
                cnt += 1
    return cnt % 2


# 判断是否有解
def solvable(cur_state):
    # 4*4的棋盘 上下移动一次奇偶性必定改变 左右移动奇偶性不变
    cur_inver = inversion(cur_state)
    index = getblock(cur_state)
    k = (3 - index[0]) % 2
    # 奇偶性相同
    if cur_inver == 0:
        if k == 0:  # 上下移动偶数次不改变奇偶性
            return True
        else:
            return False
    # 奇偶性不同
    else:
        if k == 1:  # 上下移动奇数次改变奇偶性
            return True
        else:
            return False


# 获得空格的位置
def getblock(cur_state):
    for i in range(4):
        for j in range(4):
            if cur_state[i][j] == 0:
                return [i, j]


# 格式化输出过程
def format_output():
    step = 0
    for state in PATH:
        print('Step %d' % step)
        step += 1
        for row in state:
            for num in row:
                print(num, end='\t')
            print(end='\n')
        print(end='\n')


# 初始棋盘
# 初始棋盘
ini_state_set = [[[1, 2, 4, 8], [5, 7, 11, 10], [13, 15, 0, 3], [14, 6, 9, 12]],
                 [[5, 1, 3, 4], [2, 7, 8, 12], [9, 6, 11, 15], [0, 13, 10, 14]],
                 [[14, 10, 6, 0], [4, 9, 1, 8], [2, 3, 5, 11], [12, 1, 7, 15]],
                 [[6, 10, 3, 15], [14, 8, 7, 11], [5, 1, 0, 2], [13, 12, 9, 4]],
                 [[11, 3, 1, 7], [4, 6, 8, 2], [15, 9, 10, 13], [14, 12, 5, 0]],
                 [[0, 5, 15, 14], [7, 9, 6, 13], [1, 2, 12, 10], [8, 11, 4, 3]]]
# 最终棋盘

# 行动，只可以上下左右移动
dx = [0, 1, -1, 0]
dy = [1, 0, 0, -1]


# 拓展新棋盘
def get_state(cur_state, h):
    all_state = list()
    cur_index = getblock(cur_state)
    cur_x = cur_index[0]
    cur_y = cur_index[1]
    for i in range(4):
        new_x = cur_x + dx[i]
        new_y = cur_y + dy[i]
        if 0 <= new_x <= 3 and 0 <= new_y <= 3:
            new_state = copy.deepcopy(cur_state)
            new_state[cur_x][cur_y], new_state[new_x][new_y] = new_state[new_x][new_y], new_state[cur_x][cur_y]
            if new_state in PATH:  # or not solvable(new_state): 剪枝
                continue
            all_state.append(new_state)
    return sorted(all_state, key=lambda x: h(x))  # 按照启发式函数升序排序


# DFS
def IDAsearch(bound, ini_state, gn, h):  # 函数定义为搜索当前状态，并且深入搜索下一状态
    global Nodes_num
    Nodes_num += 1
    new_depth = sys.maxsize
    PATH.append(ini_state)  # 搜索路径
    # 设定base case
    if ini_state == end_state:
        return 0
    # 深度优先搜索 不是最终状态
    for cur_state in get_state(ini_state, h):
        hn = h(cur_state)
        fn = hn + gn
        # 深度限制 fn如果大于bound 就返回fn 用于搜索失败后更新深度
        if fn > bound:
            return fn
        # 否则继续深搜，并且得到子状态中超过bound最小的深度（如果有）
        cur_depth = IDAsearch(bound, cur_state, gn + 1, h)  # 继续搜索，注意gn+1
        if cur_depth == 0:
            return 0
        if cur_depth < new_depth:  # 最小更新bound
            new_depth = cur_depth
        # 当前棋盘不是解的路径上的
        PATH.pop()
    return new_depth


# 决策
#  估值函数fn = gn + hn gn定义为走的步数 hn定义为曼哈顿距离
# IDAstar 深度设定为启发式函数的价值
def IDAstar(h, ini_state):
    if not solvable(ini_state):
        return False
    global Nodes_num
    Nodes_num = -1
    bound = h(ini_state)
    while True:
        PATH.clear()
        depth = IDAsearch(bound, ini_state, 0, h)  # 更新深度
        if depth == 0:
            # 输出结果
            format_output()
            return True
        else:
            bound = depth


for step, ini_state in enumerate(ini_state_set):
    start = time.time()
    is_solv = IDAstar(manhattan_50, ini_state)
    end = time.time()
    if is_solv:
        print('total steps %d' % (len(PATH) - 1))
        print('expand nodes %d' % Nodes_num)
        print('test %d Running time %.3f\n' % (step + 1, end - start))
    else:
        print('test %d No solution!\n' % (step+1))
