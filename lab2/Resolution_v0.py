# ¬
import re
import copy
from queue import Queue

S = list()
'''
5
On(aa,bb)
On(bb,cc)
Green(aa)
¬Green(cc)
(¬On(x,y), ¬Green(x), Green(y))
'''


def is_var(item):
    if len(item) == 1 and item.isalpha():
        return True
    return False


# 读取输入字符串，并转化为列表
def format_input(clause):
    global S
    pat = re.compile(r'[¬\w]+\(.*?\)')
    st = re.findall(pat, clause)
    for i in range(len(st)):
        st[i] = st[i].replace('(', ',').replace(')', '').replace(' ', '').split(',')
    S.append(st)


'''''
# 判断是否可以合一
def syncreticable(cmp_words, words):
    for wi in range(len(words)):
        if wi > 0 and words[wi] != cmp_words[wi]:
            if not is_var(words[wi]) and not is_var(cmp_words[wi]):
                return False
            else:
                continue
    return True
# 判断是否可以归结: 对已经变量替换后的处理
# 情况一：后面部分全部相同
# 情况二：不同位置 有一个是变量
def combinable(w1, w2):
    if w1[1:] == w2[1:]:
        return True
    for wi in range(len(w1)):
        if wi > 0 and w1[wi] != w2[wi] and (not is_var(w1[wi]) and not is_var(w2[wi])):
            return False
    return True
'''''


# 判断能否合一的函数
def syncreticable(clause_one, clause_two):
    # 一一匹配，只要匹配到两个都是常量并且不等就不合一,两个都是变量也不用合一，归结即可
    for i in range(1, len(clause_one)):
        if not is_var(clause_one[i]) and not is_var(clause_two[i]) and clause_one[i] != clause_two[i]:
            return False
        if is_var(clause_one[i]) and is_var(clause_two[i]):  # 两个变量 情况留给归结处理，这里视为不能合一
            return False
    return True


# 判断能否归结
# 句子已经合一，不能归结的情况只有 项不一样，注意两个都是变量的情况是可以归结的
def combinable(clause_one, clause_two):
    if syncreticable(clause_one,clause_two):
        # 项都一样，可以归结
        if clause_one[1:] == clause_two[1:]:
            return True
        '''
        # 都是变量可以归结
        for n in range(1,len(clause_one)):
            if is_var(clause_one[n]) and is_var(clause_two[n]):
                continue
            return False
        '''
    return False


def format_output(assignment, parents, r):
    # 去除无用子句
    index = list()
    q = Queue()
    q.put(parents[-1])
    while not q.empty():
        t = q.get()
        if t != (0, 0, 0, 0):
            q.put(parents[t[0]])
            q.put(parents[t[2]])
            index.append(t[0])
            index.append(t[2])

    # 重新编号的新子句
    new_S = copy.deepcopy(S[:r])
    new_parents = copy.deepcopy(parents[:r])
    new_assignment = copy.deepcopy(assignment[:r])
    # 标记更换的变量 0表示无变量 1表示在第一个子句中 2表示在第二个子句中 3表示在两个子句中
    new_index = list(0 for i in range(10000))
    # 创建新的，一一对应 重新编号
    for i in reversed(index):
        if i >= r:  # 更新新的列表
            new_S.append(S[i])
            new_assignment.append(assignment[i])
            # 主要是重新编号后，parents也要更新
            temp_parent = list()
            if parents[i][0] < r:
                temp_parent.append(parents[i][0] + 1)

                if len(new_S[parents[i][0]]) > 1:
                    new_index[len(new_S) - 1] = 1

            elif parents[i][0] >= r:
                temp_index = new_S.index(S[parents[i][0]])  # 找到原子句的父子句在新子句中的下标
                temp_parent.append(temp_index + 1)

                if len(new_S[temp_index]) > 1:
                    new_index[len(new_S) - 1] = 1

            if parents[i][2] < r:
                temp_parent.append(parents[i][2] + 1)

                if len(new_S[parents[i][2]]) > 1:
                    if new_index[len(new_S) - 1] == 1:
                        new_index[len(new_S) - 1] = 3
                    else:
                        new_index[len(new_S) - 1] = 2

            elif parents[i][2] >= r:
                temp_index = new_S.index(S[parents[i][2]])  # 父子句的下标
                temp_parent.append(temp_index + 1)
                if len(new_S[temp_index]) > 1:
                    if new_index[len(new_S) - 1] == 1:
                        new_index[len(new_S) - 1] = 3
                    else:
                        new_index[len(new_S) - 1] = 2

            temp_parent.append(parents[i][1] + 97)
            temp_parent.append(parents[i][3] + 97)
            new_parents.append(temp_parent)

    # 输出 parents格式为[父1，父2，父1句，父2句]
    for i in range(r, len(new_S) + 1):
        if i < len(new_S):
            if new_index[i] == 1:
                print('R[%d%s,%d]' % (new_parents[i][0], chr(new_parents[i][2]), new_parents[i][1]), end='')
            elif new_index[i] == 2:
                print('R[%d,%d%s]' % (new_parents[i][0], new_parents[i][1], chr(new_parents[i][3])), end='')
            elif new_index[i] == 3:
                print('R[%d%s,%d%s]' % (
                    new_parents[i][0], chr(new_parents[i][2]), new_parents[i][1], chr(new_parents[i][2])), end='')
            else:
                print('R[%d,%d]' % (new_parents[i][0], new_parents[i][1]), end='')

            print('(', end='')
            for j in range(len(new_assignment[i])):
                if j == len(new_assignment[i]) - 1:
                    print('%s=%s' % (new_assignment[i][j][1], new_assignment[i][j][0]), end='')
                else:
                    print('%s=%s,' % (new_assignment[i][j][1], new_assignment[i][j][0]), end='')
            print(')' + ' = ', end='')

            for j in range(len(new_S[i])):
                print(new_S[i][j][0] + '(', end='')
                if j == len(new_S[i]) - 1:
                    for k in range(1, len(new_S[i][j])):
                        if k == len(new_S[i][j]) - 1:
                            print(new_S[i][j][k], end='')
                        else:
                            print(new_S[i][j][k] + ',', end='')
                    print(')', end='\n')
                else:
                    for k in range(1, len(new_S[i][j])):
                        if k == len(new_S[i][j]) - 1:
                            print(new_S[i][j][k], end='')
                        else:
                            print(new_S[i][j][k] + ',', end='')
                    print(')' + ',', end='')
        else:  # 结尾
            print('R[%d,%d] = []' % (len(new_S) - 1, len(new_S)), end='')


def Resolution(r):
    parents = list()
    assignment = list()
    for i in range(len(S)):
        parents.append((0, 0, 0, 0))
        assignment.append([])
    res = True
    end = False
    cur_len = 0
    while True:
        S_len = len(S)
        if end:
            break
        for i in range(S_len):
            for j in range(cur_len, S_len):
                if i >= j:
                    continue
                for ki in range(len(S[i])):
                    for kj in range(len(S[j])):
                        set_one = copy.deepcopy(S[i])
                        set_two = copy.deepcopy(S[j])
                        clause_one = set_one[ki]
                        clause_two = set_two[kj]
                        # 合一归结
                        predicate_one = clause_one[0]
                        predicate_two = clause_two[0]
                        if ('¬' + predicate_one == predicate_two or predicate_one == '¬' + predicate_two) \
                                and (len(clause_one) == len(clause_two)):  # 找到谓词与其否定
                            # 合一
                            syn = syncreticable(clause_one, clause_two)  # 判断能否合一
                            if syn:
                                parents.append((i, ki, j, kj))  # 记录谁替换谁
                                assignment.append([])
                                # 分情况合一，找变量，难点在于怎么处理变量对变量的情况？
                                index = list(0 for n1 in range(len(clause_one)))  # 判断哪个项需要作变量替换
                                rep = False  # 是否替换？

                                for m in range(1, len(clause_one)):
                                    if is_var(clause_one[m]) and not is_var(clause_two[m]):  # 变量在第一个子句
                                        index[m] = 1
                                        rep = True
                                    elif not is_var(clause_one[m]) and is_var(clause_two[m]):  # 变量在第二个子句
                                        index[m] = 2  # 记录第二个子句中要替换的变量
                                        rep = True
                                    '''''
                                    # 两个都是变量的情况，放在哪处理？
                                    elif is_var(clause_one[m]) and is_var(clause_two[m]):
                                        index[m] = 3
                                        rep = True
                                    '''
                                if rep:  # 替换 并且用assignment记录
                                    clause_len = len(clause_one)
                                    replace_clause = list()
                                    for m in range(clause_len):
                                        if index[m] == 1:  # 变量在第一个式子中
                                            replace_clause.append((clause_two[m], clause_one[m]))
                                            for n in range(len(set_one)):
                                                set_one[n] = [
                                                    clause_two[m] if set_one[n][k] == clause_one[m] else set_one[n][k]
                                                    for k in range(len(set_one[n]))]
                                        elif index[m] == 2:  # 变量在第二个式子中
                                            replace_clause.append((clause_one[m], clause_two[m]))
                                            for n in range(len(set_two)):
                                                set_two[n] = [
                                                    clause_one[m] if set_two[n][k] == clause_two[m] else set_two[n][k]
                                                    for k in range(len(set_two[n]))]
                                        '''''
                                        elif index[m] == 3:  # 两个式子都有变量，这里按照第二种情况处理，将句一的变量看作常量
                                            replace_clause.append((clause_one[m], clause_two[m]))
                                            for n in range(len(set_two)):
                                                set_two[n] = [
                                                    clause_one[m] if set_two[n][k] == clause_two[m] else set_two[n][k]
                                                    for k in range(len(set_two[n]))]
                                        '''''
                                    assignment[-1] = replace_clause

                            # 归结 需要处理两个都是变量的情况
                            com = combinable(set_one[ki], set_two[kj])  # 判断两个子句能不能归结
                            if com:
                                combine_clause = list()
                                for k in range(len(set_one)):
                                    if set_one[k] != set_one[ki] and set_one[k] not in combine_clause:
                                        combine_clause.append(set_one[k])
                                for k in range(len(set_two)):
                                    if set_two[k] != set_two[kj] and set_two[k] not in combine_clause:
                                        combine_clause.append(set_two[k])
                                if len(combine_clause) == 0:
                                    end = True
                                elif combine_clause not in S:
                                    S.append(combine_clause)
                                elif combine_clause in S:
                                    assignment.pop()
                                    parents.pop()
                                if end:
                                    break
                    if end:
                        break
                if end:
                    break
            if end:
                break
        if cur_len == S_len:
            res = False
            end = True
        else:
            cur_len = S_len
    # 格式化输出
    if res:
        format_output(assignment, parents, r)
    else:
        print('no resolution!')


r = int(input())
for i in range(r):
    s = input()
    format_input(s)
Resolution(r)
