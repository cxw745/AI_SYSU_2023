# 中山大学计算机学院人工智能本科生实验报告

**（2023学年春季学期）**

课程名称：Artificial Intelligence

| 教学班级   | 人工智能         | 专业（方向） | 计算机科学与技术 |
| ------ | ------------ | ------ | -------- |
| **学号** | **21307387** | **姓名** | **陈雪玮**  |

# 一、实验题目

用归结算法求解逻辑推理问题

# 二、实验内容

## 1.算法原理

算法可以分为三个步骤：

一、读取初始子句集S，将其存为一个列表。

二、利用广度优先策略遍历子句集S，进行子句的合一和归结，生成新子句后加入子句集S，重复以上过程，直到归结出空子句。而每一次的合一和归结，都需要记录新子句是由哪些子句合一、归结而来。（记录路径，parents (i, ki, j, kj)，表示这个新的子句是由第i个子句中的第ki个原子公式和第j个子句中的第kj个原子公式归结而来；assignment（aa，x）表示合一时x替换为aa。）

三、去除归结过程中生成的无用子句，按格式输出。通过队列的层序遍历，利用parents列表回溯路径，将归结出空子句的子句重新编号，parents和assignment的记录也重新编号，然后按照标准格式输出答案。

## 2.伪代码

以下为三个步骤的伪代码：

一、读取初始子句集

二、合一和归结，记录路径

```python
while True do
    更新添加了新子句后的S大小 S_len = len(S)
    for i = 1 : S_len do
        for j = cur_len: S_len do
            if i >= j then 
                continue #不要重复遍历
            end if
            for ki = 1: len(S[i]) do    
                   for kj = 1: len(S[j]) do
                        if 存在原子及其的否定（如On和它的否定¬On）then
                            合一：相同谓词、不同参数项的两个原子用最一般合一算法进行合一
                            归结：归结生成新子句s，加入到S中
                            记录：记录其最一般合一时变量替换情况assignment(aa,x)及其parents (i, ki, j, kj)
                            if s == [] (s为空子句) then
                                退出循环，格式化输出
                        end if
                   end for 
            end for
        end for
    if 当前子句长度没有增加，也没有退出循环 then
        归结失败
    else 
        更新遍历层数 cur_len = S_len
    end for
end while
```

三、去除无用子句，格式化输出

```python
Function format_output(assignment, parents, num):
    去除无用子句
    重新编号的新子句，下面是用来存储更新后的内容
    new_S = copy.deepcopy(S[:num])
    new_parents = copy.deepcopy(parents[:num])
    new_assignment = copy.deepcopy(assignment[:num])
    格式化输出
```

## 3.关键代码展示（带注释）

**将输入转化，存储在列表里**

```python
# 读取输入字符串，并转化为列表
def format_input(clause):
    pat = re.compile(r'[¬\w]+\(.*?\)') #正则表达式的格式
    st = re.findall(pat, clause) #正则表达式匹配
    for i in range(len(st)): #将匹配后多余的内容删去后加入到S中
        st[i] = st[i].replace('(', ',').replace(')', '').replace(' ', '').split(',')
    S.append(st)
```

**关键的判断能否合一、归结的函数**

```python
# 判断能否合一的函数
def syncreticable(clause_one, clause_two):
    # 一一匹配，只要匹配到两个都是常量或两个都是变量不合一
    for i in range(1, len(clause_one)):
        if not is_var(clause_one[i]) and not is_var(clause_two[i]) and clause_one[i] != clause_two[i]:
            return False
        if is_var(clause_one[i]) and is_var(clause_two[i]):
            return False
    return True


# 判断能否归结的函数
# 句子已经合一，不能归结的情况只有项不一样
def combinable(clause_one, clause_two):
    # 归结的句子必然是合一后的，所以先判断能否合一
    if syncreticable(clause_one, clause_two):
        # 项都一样，可以归结
        if clause_one[1:] == clause_two[1:]:
            return True
    return False
```

**通过BFS遍历S，进行合一、归结、记录路径**

```python
def Resolution(num):
    parents = list()
    assignment = list()
    for i in range(len(S)):
        parents.append((0, 0, 0, 0))
        assignment.append([])
    res = True  # 是否能归结
    end = False  # 用于跳出多层循环
    cur_len = 0  # 用于层序遍历
    while True:
        S_len = len(S)  # 更新长度
        if end:
            break
        for i in range(S_len):
            for j in range(cur_len, S_len):
                if i >= j:  # 一一匹配
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
                                assignment.append([])  # 加入语句，和parents下标对齐
                                # 分情况合一，找变量
                                index = list(0 for n1 in range(len(clause_one)))  # 判断哪个项需要作变量替换
                                rep = False  # 是否替换？
                                for m in range(1, len(clause_one)):
                                    if is_var(clause_one[m]) and not is_var(clause_two[m]):  # 变量在第一个子句
                                        index[m] = 1
                                        rep = True
                                    elif not is_var(clause_one[m]) and is_var(clause_two[m]):  # 变量在第二个子句
                                        index[m] = 2  # 记录第二个子句中要替换的变量
                                        rep = True
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
                                    assignment[-1] = replace_clause  # 将assignment尾部用合一记录替换

                            # 归结
                            com = combinable(set_one[ki], set_two[kj])  # 判断两个子句能不能归结
                            if com:
                                combine_clause = list()
                                for k in range(len(set_one)):
                                    if set_one[k] != set_one[ki] and set_one[k] not in combine_clause:
                                        combine_clause.append(set_one[k])
                                for k in range(len(set_two)):
                                    if set_two[k] != set_two[kj] and set_two[k] not in combine_clause:
                                        combine_clause.append(set_two[k])
                                # 判断归结后的结果
                                if len(combine_clause) == 0:  # 空子句
                                    end = True
                                elif combine_clause not in S:  # S里没有
                                    S.append(combine_clause)
                                elif combine_clause in S:  # S里有了，需要把已经加入的记录弹出，维护与S的下标对应
                                    assignment.pop()
                                    parents.pop()
                    if end:
                        break
                if end:
                    break
            if end:
                break
        # 没有跳出循环且没有新子句加入了，说明无法归结出空子句
        if cur_len == S_len:
            res = False
            end = True
        # 更新层数
        else:
            cur_len = S_len
    # 格式化输出
    if res:
        format_output(assignment, parents, num)
    else:
        print('No resolution!')
```

**去除无用子句并进行格式化输出**

```python
def format_output(assignment, parents, num):
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
    new_S = copy.deepcopy(S[:num])
    new_parents = copy.deepcopy(parents[:num])
    new_assignment = copy.deepcopy(assignment[:num])
    # 标记更换的变量 0表示无变量 1表示在第一个子句中 2表示在第二个子句中 3表示在两个子句中
    new_index = list(0 for i in range(10000))
    # 创建新的，一一对应 重新编号
    for i in reversed(index):
        if i >= num:  # 更新新的列表
            new_S.append(S[i])
            new_assignment.append(assignment[i])
            # 主要是重新编号后，parents也要更新
            temp_parent = list()
            if parents[i][0] < num:
                temp_parent.append(parents[i][0] + 1)

                if len(new_S[parents[i][0]]) > 1:
                    new_index[len(new_S) - 1] = 1

            elif parents[i][0] >= num:
                temp_index = new_S.index(S[parents[i][0]])  # 找到原子句的父子句在新子句中的下标
                temp_parent.append(temp_index + 1)

                if len(new_S[temp_index]) > 1:
                    new_index[len(new_S) - 1] = 1

            if parents[i][2] < num:
                temp_parent.append(parents[i][2] + 1)

                if len(new_S[parents[i][2]]) > 1:
                    if new_index[len(new_S) - 1] == 1:
                        new_index[len(new_S) - 1] = 3
                    else:
                        new_index[len(new_S) - 1] = 2

            elif parents[i][2] >= num:
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
    一一对应格式化输出，比较冗长且不是很关键，可在源代码中查看
```

## 4.创新点&优化（如果有）



# 三、实验结果及分析

## 1.实验结果展示示例（可图可表可文字，尽量可视化）

实验结果显示，该算法能够对谓词逻辑句子进行合一和归结，并输出归结的过程和结果。

请运行main.py文件查看三个示例，结果如下图。可以看到符合格式化输出，且归结逻辑正确。

![](C:\Users\Chan\AppData\Roaming\marktext\images\2023-03-20-21-50-10-image.png)

![](C:\Users\Chan\AppData\Roaming\marktext\images\2023-03-20-21-50-40-image.png)

![](C:\Users\Chan\AppData\Roaming\marktext\images\2023-03-20-21-51-15-image.png)

## 2.评测指标展示及分析（机器学习实验必须有此项，其它可分析运行时间等）

本实验的评测指标主要包括以下两个方面：算法的正确性和算法的时间复杂度。

1. 算法的正确性：对三个测试样例，实验结果表明算法都能够正确地对谓词逻辑句子进行合一和归结，并输出归结的过程和结果。

2. 算法的时间复杂度：在一般情况下，谓词逻辑归结算法的时间复杂度为指数级别，即 $O(2^n)$，其中 n 是待归结的句子数量。如果输入的句子数量较大，该算法可能非常慢，因为随着句子数量的增加，需要归结的句子对数以指数级别增加。

因此，对于大型或复杂的问题，可能需要使用更高效的算法或优化技术。例如启发式搜索、限制深度的深度优先搜索。

# 四、思考题

无

# 五、参考资料

[正则表达式 – 匹配规则 | 菜鸟教程 (runoob.com)](https://www.runoob.com/regexp/regexp-rule.html)

[python基础_格式化输出（%用法和format用法） - fat39 - 博客园 (cnblogs.com)](https://www.cnblogs.com/fat39/p/7159881.html)

[伪代码书写规范_Blooming Life的博客-CSDN博客](https://blog.csdn.net/qq_40078121/article/details/88692887)

[教你写一手漂亮的伪代码（详细规则&简单实例）_算法伪代码怎么写__陈同学_的博客-CSDN博客](https://blog.csdn.net/Dan1374219106/article/details/106676043)

[科研基础3-伪代码规范 - Shuzang's Blog](https://shuzang.github.io/2021/pseudocode-specification/)
