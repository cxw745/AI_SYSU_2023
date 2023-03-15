# Copyright (c) Xxxxic. All rights reserved.
import re
import copy
from queue import Queue


# 返回谓词的非
def opposite(word):
    if '¬' in word:
        return word.replace('¬', '')
    else:
        return '¬' + word


# 判断是否为变量
def is_value(word):
    if word.isalpha() and len(word) == 1:
        return True
    else:
        return False


# 判断是否可以合一
def judge_unification(cmp_words, words):
    for wi in range(len(words)):
        if wi > 0 and words[wi] != cmp_words[wi]:
            if not is_value(words[wi]) and not is_value(cmp_words[wi]):
                return False
            else:
                continue
    return True


# 返回其变量的下标
def find_value(words):
    words_value_list = []
    for wi in range(len(words)):
        # 注意谓词也可能是单字母
        if wi > 0 and is_value(words[wi]):
            words_value_list.append(wi)
    return words_value_list


# 判断是否可以归结: 对已经变量替换后的处理
# 情况一：后面部分全部相同
# 情况二：不同位置 有一个是变量
def judge_resolution(w1, w2):
    if w1[1:] == w2[1:]:
        return True
    for wi in range(len(w1)):
        if wi > 0 and w1[wi] != w2[wi] and (not is_value(w1[wi]) and not is_value(w2[wi])):
            return False
    return True


if __name__ == '__main__':
    n = input()
    n = int(n)
    ss = []
    for t in range(n):
        ts = input()
        ss.append(ts)
    # ss = ['On(aa,bb)', 'On(bb,cc)', 'Green(aa)', '¬Green(cc)', '(¬On(x,y), ¬Green(x), Green(y))']

    s = []
    for sss in ss:
        pat = re.compile(r"[¬a-zA-Z]+\(.*?\)")
        ll = re.findall(pat, sss)
        ans = []
        for i in range(len(ll)):
            st = ll[i].replace(' ', '').replace("(", ',').replace(")", '')
            # print(st)
            ans.append(st.split(','))
        s.append(ans)
    # print(s)

    anss = copy.deepcopy(s)
    assignment = [[] for i in range(len(s))]
    parents = [(0, 0, 0, 0) for i in range(len(s))]
    end = False
    temp_len = 0
    while True:
        if end:
            break
        # 遍历子句集合S，进行子句的合一、归结生成新子句，直到归结出空子句
        new_len = copy.deepcopy(len(s))
        for i in range(new_len):
            for j in range(temp_len, new_len):
                if i < j:  # 自己和自己不能生成子句
                    for ki in range(len(s[i])):
                        for kj in range(len(s[j])):
                            if s[i][ki][0] == opposite(s[j][kj][0]):
                                replaced_si = copy.deepcopy(s[i])
                                replaced_sj = copy.deepcopy(s[j])
                                # 合一
                                # 判断可以合一后，只要不是变量对变量 都可以进行替换
                                if judge_unification(s[i][ki], s[j][kj]):
                                    wl1 = find_value(s[i][ki])
                                    wl2 = find_value(s[j][kj])
                                    if len(set(wl1) & set(wl2)) != 0:
                                        continue

                                    parents.append((i, ki, j, kj))
                                    assignment.append([])
                                    # 记录变量替换assignment(包括x换y吗?)
                                    if len(set(wl1) & set(wl2)) == 0:
                                        for wl1i in wl1:
                                            assignment[-1].append((s[i][ki][wl1i], s[j][kj][wl1i]))
                                        for wl2i in wl2:
                                            assignment[-1].append((s[j][kj][wl2i], s[i][ki][wl2i]))
                                        # 变量替换不能冲突 否则也不能合一
                                        # 待完成

                                    # 进行变量替换
                                    for replaced_unit_idx in range(len(replaced_si)):
                                        for replaced_words in assignment[-1]:
                                            replaced_si[replaced_unit_idx] = [
                                                replaced_words[1] if t == replaced_words[0] else t
                                                for t in
                                                replaced_si[replaced_unit_idx]]
                                    for replaced_unit_idx in range(len(replaced_sj)):
                                        for replaced_words in assignment[-1]:
                                            replaced_sj[replaced_unit_idx] = [
                                                replaced_words[1] if t == replaced_words[0] else t
                                                for t in
                                                replaced_sj[replaced_unit_idx]]

                                # 归结
                                if judge_resolution(replaced_si[ki], replaced_sj[kj]):
                                    temp_str = []
                                    for combine_i in range(len(replaced_si)):
                                        if combine_i != ki:
                                            temp_str.append(replaced_si[combine_i])
                                    for combine_j in range(len(replaced_sj)):
                                        if combine_j != kj:
                                            temp_str.append(replaced_sj[combine_j])
                                    new_str = []
                                    [new_str.append(n) for n in temp_str if n not in new_str]

                                    if new_str in s:
                                        parents.pop()
                                        assignment.pop()
                                    else:
                                        s.append(new_str)

                                    if not new_str:
                                        end = True
                                        break
                        if end:
                            break
                    if end:
                        break
            if end:
                break
        temp_len = new_len

    # 去除无用子句
    q1 = Queue()
    q2 = Queue()
    q1.put(parents[-1])
    q2.put(parents[-1])
    t_ans_idx = []
    while q1.qsize():
        top = q1.get()
        if top != (0, 0, 0, 0):
            q1.put(parents[top[0]])
            # print(top, end=' ')
            t_ans_idx.append(top[0])
    while q2.qsize():
        top = q2.get()
        if top != (0, 0, 0, 0):
            q2.put(parents[top[2]])
            # print(top, end=' ')
            t_ans_idx.append(top[2])

    # 存reindex 原id
    final_ans_idx = [i for i in range(len(anss))]
    for i in reversed(t_ans_idx):
        if i >= len(anss):
            final_ans_idx.append(i)
    final_ans_idx.append(len(parents) - 1)
    # print(final_ans_idx)

    # 输出
    for i in range(len(final_ans_idx)):
        if len(anss) <= i < len(final_ans_idx) - 1:
            # 长度为1就不用输出字母
            if len(s[parents[final_ans_idx[i]][0]]) <= 1:
                str1 = parents[final_ans_idx[i]][0] + 1
            else:
                str1 = str(1 + final_ans_idx.index(parents[final_ans_idx[i]][0])) + chr(
                    97 + parents[final_ans_idx[i]][1])
            if len(s[parents[final_ans_idx[i]][2]]) <= 1:
                str2 = parents[final_ans_idx[i]][2] + 1
            else:
                str2 = str(1 + final_ans_idx.index(parents[final_ans_idx[i]][2])) + chr(
                    97 + parents[final_ans_idx[i]][3])
            print('R[%s,%s]' % (str1, str2), end='')

            asgnm = ['%s=%s' % (assignment[final_ans_idx[i]][j][0], assignment[final_ans_idx[i]][j][1]) for j in
                     range(len(assignment[final_ans_idx[i]]))]
            if len(asgnm) > 0:
                print('(', end='')
                for j in range(len(asgnm)):
                    print(asgnm[j], end='')
                    if j < len(asgnm) - 1:
                        print(',', end='')
                print(')', end='')
            print(' = ', end='')
            for j in range(len(s[final_ans_idx[i]])):
                print(s[final_ans_idx[i]][j][0], end='')
                print('(', end='')
                for k in range(len(s[final_ans_idx[i]][j])):
                    if k > 0:
                        print(s[final_ans_idx[i]][j][k], end='')
                    if 0 < k < len(s[final_ans_idx[i]][j]) - 1:
                        print(',', end='')
                print(')', end='')
                if j < len(s[final_ans_idx[i]]) - 1:
                    print(',', end='')
            print('')
        # 处理最后输出
        elif i == len(final_ans_idx) - 1:
            print('R[%d,%d]' % (len(final_ans_idx) - 2, len(final_ans_idx) - 1), end='')
            print(' = []')
            break
