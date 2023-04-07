import copy
import time
import tkinter as tk
from queue import PriorityQueue
from heuristic import *

# 初始棋盘
ini_state_set = [[[1, 2, 4, 8], [5, 7, 11, 10], [13, 15, 0, 3], [14, 6, 9, 12]],
                 [[5, 1, 3, 4], [2, 7, 8, 12], [9, 6, 11, 15], [0, 13, 10, 14]],
                 [[14, 10, 6, 0], [4, 9, 1, 8], [2, 3, 5, 11], [12, 1, 7, 15]],
                 [[6, 10, 3, 15], [14, 8, 7, 11], [5, 1, 0, 2], [13, 12, 9, 4]],
                 [[11, 3, 1, 7], [4, 6, 8, 2], [15, 9, 10, 13], [14, 12, 5, 0]],
                 [[0, 5, 15, 14], [7, 9, 6, 13], [1, 2, 12, 10], [8, 11, 4, 3]]]
# 最终棋盘
end_state = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 0]]
PATH = list()
PATH_GUI = list()
Nodes_num = 0
# 行动，只可以上下左右移动
dx = [0, 1, -1, 0]
dy = [1, 0, 0, -1]


class GUI:
    def __init__(self, state_set):
        self.state_set = state_set  # 解的路径的集合
        self.root = tk.Tk()
        self.root.title('15-Puzzle Solver')
        # 初始化布局，每个格子100*100，总共需要400*400的总界面
        self.board_canvas = tk.Canvas(self.root, width=400, height=400, borderwidth=0, relief='ridge')
        self.board_canvas.pack(side=tk.LEFT, padx=25, pady=25)
        # 将显示每一步的步骤在侧边栏
        self.step_listbox = tk.Listbox(self.root, width=25, height=20, relief='ridge')
        self.step_listbox.pack(side=tk.RIGHT, padx=20, pady=10)

    # 绘制每一个状态的图形界面
    def draw_cur_state(self, cur_state):
        # 设定格子的坐标
        for row_index, row in enumerate(cur_state):
            for col_index, num in enumerate(row):
                x0 = col_index * 100
                y0 = row_index * 100
                x1 = x0 + 100
                y1 = y0 + 100

                if num == 0:
                    # Draw an empty cell
                    self.board_canvas.create_rectangle(x0, y0, x1, y1, fill='white', outline='black')
                else:
                    # Draw a cell with text inside
                    self.board_canvas.create_rectangle(x0, y0, x1, y1, fill='gray', outline='black')
                    self.board_canvas.create_text(x0 + 50, y0 + 50, text=str(num), font=('Arial', 25, 'bold'))

    # 通过解的集合更新每一个状态的过程
    def update_cur_state(self):
        for step, cur_state in enumerate(self.state_set):
            self.step_listbox.insert(tk.END, 'move %d' % step)
            for row in cur_state:
                self.step_listbox.insert(tk.END, str(row))
            self.step_listbox.insert(tk.END, '')  # 相当于空行
            self.step_listbox.yview_moveto(1.0)  # 将列表框滚动到最底部
            # 插入文本后绘制界面
            self.draw_cur_state(cur_state)
            self.root.update()
            time.sleep(0.5)

    def run(self, runtime):
        self.update_cur_state()
        self.step_listbox.insert(tk.END, 'Runing time is %.2f' % runtime)
        self.step_listbox.yview_moveto(1.0)  # 将列表框滚动到最底部
        self.step_listbox.insert(tk.END, 'Total steps % d' % (len(PATH) - 1))
        self.step_listbox.yview_moveto(1.0)  # 将列表框滚动到最底部
        self.root.mainloop()


# 状态
class S:
    def __init__(self, gn=0, hn=0, state=None):
        self.gn = gn
        self.hn = hn
        self.fn = self.gn + self.hn
        self.state = state

    def __lt__(self, s):  # 定义小根堆，比较优先级是fn、hn、gn
        if self.fn == s.fn:
            return self.hn < s.hn  # 如果总代价相等，选择估值小的
        return self.fn < s.fn


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


# 启发式函数
def manhattan(cur_state):
    distance = 0
    for i in range(4):
        for j in range(4):
            if cur_state[i][j] == end_state[i][j] or cur_state[i][j] == 0:
                continue
            else:
                end_x = (cur_state[i][j] - 1) // 4
                end_y = (cur_state[i][j] - 1) % 4
                distance += abs(i - end_x) + abs(j - end_y)
    return distance


# 获得空格的位置
def getblock(cur_state):
    for i in range(4):
        for j in range(4):
            if cur_state[i][j] == 0:
                return [i, j]


# 格式化输出过程
def format_output(index_state, state_parent, cur_index):
    while cur_index != -1:
        PATH.append(cur_index)
        new_state = index_state[cur_index]
        cur_index = state_parent[str(new_state)]

    length = len(PATH)
    for i in range(length):
        cur_index = PATH[-i - 1]
        cur_state = index_state[cur_index]
        PATH_GUI.append(cur_state)
        print('Step %d' % i, end='\n')
        for i in range(4):
            for j in range(4):
                print(cur_state[i][j], end='\t')
            print(end='\n')
        print(end='\n')


# 决策
#  估值函数fn = gn + hn gn定义为走的步数 hn定义为曼哈顿距离
def Astar(h, ini_state):  # 参数是启发函数
    if not solvable(ini_state):
        return False
    # Close表 记录访问过的棋盘
    PATH.clear()  # 初始化路径记录列表
    CLOSE = list()
    # Open表，先采用列表，后面采用优先队列优化
    OPEN = PriorityQueue()
    # 哈希表 建立棋盘到下标的映射
    state_index = dict()
    index_state = dict()
    state_parent = dict()
    index = 0

    hn = h(ini_state)
    new_S = S(0, hn, ini_state)
    OPEN.put(new_S)
    state_parent[str(ini_state)] = -1  # 初始化最初态的父亲下标为-1
    while not OPEN.empty():
        cur_S = OPEN.get()
        cur_gn = cur_S.gn
        cur_state = cur_S.state
        # 得到当前棋盘空格的坐标
        cur_index = getblock(cur_state)
        cur_x, cur_y = cur_index[0], cur_index[1]
        CLOSE.append(cur_state)

        state_index[str(cur_state)] = index
        index_state[index] = cur_state

        index += 1
        # 产生新状态
        for i in range(4):
            new_x = cur_x + dx[i]
            new_y = cur_y + dy[i]
            if 0 <= new_x <= 3 and 0 <= new_y <= 3:
                new_state = copy.deepcopy(cur_state)
                # 移动空格
                new_state[cur_x][cur_y], new_state[new_x][new_y] = new_state[new_x][new_y], new_state[cur_x][cur_y]
                if new_state not in CLOSE:
                    new_hn = h(new_state)
                    new_gn = cur_gn + 1
                    new_S = S(new_gn, new_hn, new_state)
                    OPEN.put(new_S)
                    state_parent[str(new_state)] = state_index[str(cur_state)]
                    if new_hn == 0:  # 输出过程
                        # 加入PATH中寻找得出答案的路径 为了实现这个 需要将每一次的state建立一个哈希表 并且记录父亲的下标
                        index_state[index] = new_state  # 终点的下标
                        state_index[str(new_state)] = index
                        format_output(index_state, state_parent, index)

                        global Nodes_num
                        Nodes_num = OPEN.qsize() + len(CLOSE) - 1
                        return True


for step, ini_state in enumerate(ini_state_set):
    start = time.time()
    is_solv = Astar(manhattan_2, ini_state)
    end = time.time()
    if is_solv:
        gui = GUI(PATH_GUI)
        gui.run((end-start))
        print('total steps %d' % (len(PATH) - 1))
        print('expand nodes %d' % Nodes_num)
        print('test %d Running time %.3f\n' % (step + 1, end - start))
    else:
        print('test %d No solution!\n' % (step + 1))
