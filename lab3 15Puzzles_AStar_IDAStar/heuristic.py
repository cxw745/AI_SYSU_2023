end_state = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 0]]


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


def chebyshev(cur_state):
    distance = 0
    for i in range(4):
        for j in range(4):
            if cur_state[i][j] == end_state[i][j] or cur_state[i][j] == 0:
                continue
            else:
                end_x = (cur_state[i][j] - 1) // 4
                end_y = (cur_state[i][j] - 1) % 4
                distance += max(abs(i - end_x), abs(j - end_y))
    return distance


def euclidean(cur_state):
    distance = 0
    for i in range(4):
        for j in range(4):
            if cur_state[i][j] == end_state[i][j] or cur_state[i][j] == 0:
                continue
            else:
                end_x = (cur_state[i][j] - 1) // 4
                end_y = (cur_state[i][j] - 1) % 4
                distance += (i - end_x) ** 2 + (j - end_y) ** 2
    return distance ** 0.5


def misplaced(cur_state):
    distance = 0
    for i in range(4):
        for j in range(4):
            if cur_state[i][j] == end_state[i][j] or cur_state[i][j] == 0:
                continue
            else:
                end_x = (cur_state[i][j] - 1) // 4
                end_y = (cur_state[i][j] - 1) % 4
                if i != end_x or j != end_y:
                    distance += 1
    return distance


def manhattan_2(cur_state):
    distance = 0
    for i in range(4):
        for j in range(4):
            if cur_state[i][j] == end_state[i][j] or cur_state[i][j] == 0:
                continue
            else:
                end_x = (cur_state[i][j] - 1) // 4
                end_y = (cur_state[i][j] - 1) % 4
                distance += abs(i - end_x) + abs(j - end_y)
    return 2 * distance


def manhattan_5(cur_state):
    distance = 0
    for i in range(4):
        for j in range(4):
            if cur_state[i][j] == end_state[i][j] or cur_state[i][j] == 0:
                continue
            else:
                end_x = (cur_state[i][j] - 1) // 4
                end_y = (cur_state[i][j] - 1) % 4
                distance += abs(i - end_x) + abs(j - end_y)
    return 5 * distance


def manhattan_10(cur_state):
    distance = 0
    for i in range(4):
        for j in range(4):
            if cur_state[i][j] == end_state[i][j] or cur_state[i][j] == 0:
                continue
            else:
                end_x = (cur_state[i][j] - 1) // 4
                end_y = (cur_state[i][j] - 1) % 4
                distance += abs(i - end_x) + abs(j - end_y)
    return 10 * distance


def manhattan_50(cur_state):
    distance = 0
    for i in range(4):
        for j in range(4):
            if cur_state[i][j] == end_state[i][j] or cur_state[i][j] == 0:
                continue
            else:
                end_x = (cur_state[i][j] - 1) // 4
                end_y = (cur_state[i][j] - 1) % 4
                distance += abs(i - end_x) + abs(j - end_y)
    return 50 * distance
