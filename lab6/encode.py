import numpy as np


# 读取文件
def read_file(file):
    # 返回数据集 的 单词和标签
    words = []
    label_name = []
    label_idx = []
    with open(file) as f:
        data_title = f.readline()
        print('The data name is %s' % data_title)
        dataset = f.readlines()
        for data in dataset:
            tmp_data = data.replace('\n', '').split(' ')
            words.append(tmp_data[3:])
            label_name.append(tmp_data[2])
            label_idx.append(int(tmp_data[1]))
    return words, label_name, label_idx


# 词频逆文档频率计算一个词的重要性
def tf_idf(word, doc, docs):
    # word是要计算的单词，doc是当前文档存有所有单词，docs是所有的文档
    word_doc = sum(1 for doc in docs if word in docs)  # 计算所有文档中包含该单词的文档数
    tf = doc.count(word) / len(doc)
    idf = np.log(len(docs) / (word_doc + 1))
    return tf * idf


def word2vec(x_train, t_train, x_test, t_test, ban, function):
    # 返回转换好的 训练集和测试集
    x_train_vec = []
    t_train_vec = []
    x_test_vec = []
    t_test_vec = []

    # 禁用词表
    banwords = []  # 禁用词表
    if ban:  # 是否启用禁用词
        banwords.append(
            'a, able, about, across, after, all, almost, also, am, among, an, and, any, are, as, at, be, because, '
            'been, but, by, can, cannot, could, dear, did, do, does, either, else, ever, every, for, from, get, '
            'got, had, has, have, he, her, hers, him, his, how, however, i, if, in, into, is, it, its, just, '
            'least, let, like, likely, may, me, might, most, must, my, neither, no, nor, not, of, off, often, on, '
            'only, or, other, our, own, rather, said, say, says, she, should, since, so, some, than, that, the, '
            'their, them, then, there, these, they, this, tis, to, too, twas, us, wants, was, we, were, what, '
            'when, where, which, while, who, whom, why, will, with, would, yet, you, your, a, about, above, across, '
            'actually, add, ago, all, almost, along, already, also, although, always, am, among, an, and, another, '
            'any, anyone, anything, anyway, anywhere, are, aren, around, as, ask, at, away, b, back, be, because, '
            'been, before, being, below, best, better, between, big, bit, both, but, by, c, called, can, came, cannot,'
            'case, certain, certainly, clear, clearly, come, common, concerning, consequently, consider, could, '
            'couldn, d, date, day, did,  different, do, does, doesn, doing, done, don, down, due, during, e, each, '
            'early, either, else, end, enough, especially, even, ever, every, everyone, everything, example, except, '
            'f, face, fact, far, few, find, first, for, form, four, from, full, further, g, general, get, give, go, '
            'going, good, got, great, h, had, hardly, has, hasn, have, having, he, her, here, hi, high, him, himself, '
            'his, hit, hold, home, how, however, i, if, in, indeed, information, interest, into, is, isn, issue, it, '
            'its, it,s, itself, j, just, k, keep, kind, know, known, l, large, last, late, later, least, left, less, '
            'let, letter, likely, long, look, m, made, make, many, may, maybe, me, mean, meets, member, mention, '
            'might, mine, miss, more, most, mostly, much, must, my, myself, n, name, namely, need, never, new, next, '
            'nine, no, nobody, none, nor, normally, not, nothing, now, o, of, off, often, oh, ok, okay, old, on, once,'
            'one, only, onto, or, other, our, ours, out, over, own, p, part, particular, past, people, perhaps, '
            'person, place, plus, point, possible, present, probably, program, provide, put, q, question, quickly, '
            'quite, r, rather, really, recent, regarding, regards, related, relatively, request, right, result, '
            'return, s, said, same, saw, say, saying, says, second, see, seem, seemed, seeming, seems, seen, self, '
            'send, sent, several, shall, she, should, shouldn, show, showed, shown, shows, side, since, six, small, '
            'so, some, somebody, somehow, someone, something, sometime, sometimes, somewhat, somewhere, soon, sorry, '
            'specific, specified, specify, still, stop, such, sure, t, take, taken, taking, tends, term, than, that, '
            'thats, the, their, theirs, them, themselves, then, there, therefore, these, they, thing, things, think, '
            'third, this, those, three, through, thus, time, to, together, too, took, toward, turned, two, u, under, '
            'understood, unfortunately, unless, unlike, unlikely, until, up, upon, us, use, used, useful, usually, v, '
            'value, various, very, via, video, view, w, want, was, wasn, way, we, well, were, what, whatever, when, '
            'whenever, where, whether, which, while, who, whole, whom, whose, why, will, with, within, without, won, '
            'work, would, wouldn, x, y, year, yes, yet, you, your, yours, yourself, yourselves, z')
        banwords = set(banwords[0].replace(' ', '').split(','))

    dictionary = set()
    for words in x_train:
        for word in words:
            if word not in banwords:
                dictionary.add(word)
    dictionary = list(dictionary)  # 字典

    # 建立标签表
    label = set()
    for t in t_train:
        label.add(t)
    label = list(label)

    if function == 'tf_idf':
        for words in x_train:
            vec = [0 for _ in range(len(dictionary))]
            for word in words:
                if word in dictionary:
                    vec[dictionary.index(word)] = tf_idf(word, words, x_train)
            x_train_vec.append(vec)

        for words in x_test:
            vec = [0 for _ in range(len(dictionary))]
            for word in words:
                if word in dictionary:
                    vec[dictionary.index(word)] = tf_idf(word, words, x_test)
            x_test_vec.append(vec)

    elif function == 'one_hot':
        for words in x_train:
            vec = [0 for _ in range(len(dictionary))]
            for word in words:
                if word in dictionary:
                    vec[dictionary.index(word)] = 1
            x_train_vec.append(vec)

        for words in x_test:
            vec = [0 for _ in range(len(dictionary))]
            for word in words:
                if word in dictionary:
                    vec[dictionary.index(word)] = 1
            x_test_vec.append(vec)

    for t in t_train:
        vec = [0 for _ in range(len(label))]
        if t in label:
            vec[label.index(t)] = 1
        t_train_vec.append(vec)

    for t in t_test:
        vec = [0 for _ in range(len(label))]
        if t in label:
            vec[label.index(t)] = 1
        t_test_vec.append(vec)

    return x_train_vec, t_train_vec, x_test_vec, t_test_vec
