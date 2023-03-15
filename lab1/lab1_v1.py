MAX = 1000000
if __name__ == '__main__':
    m, n = map(int, input().split())
    g = [[MAX for i in range(26)] for j in range(26)]
    dic = dict(zip([chr(i) for i in range(97, 97 + 26)], [i for i in range(26)]))
    for i in range(n):
        u1, v1, d = input().split()
        d = int(d)
        u = dic[u1]
        v = dic[v1]
        if d < g[u][v]:
            g[u][v] = d
            g[v][u] = d
    while True:
        u1, v1 = input().split()
        u = dic[u1]
        v = dic[v1]
        vis = list(0 for i in range(26))
        dis = list(MAX for i in range(26))
        path = list(-1 for i in range(26))
        dis[u] = 0
        for j in range(26):
            index = -1
            for i in range(26):
                if vis[i] == 0 and (index == -1 or dis[i] < dis[index]):
                    index = i
            if index == v:
                print(dis[v])
                road = list()
                pre = v
                for i in range(26):
                    road.append(chr(pre + 97))
                    if path[pre] == -1:
                        break
                    pre = path[pre]
                length = len(road)
                for i in range(length):
                    if i < length - 1:
                        print(road.pop(), end="->")
                    else:
                        print(road.pop())
                break
            vis[index] = 1
            for i in range(26):
                if vis[i] == 0 and dis[index] + g[index][i] < dis[i]:
                    dis[i] = dis[index] + g[index][i]
                    path[i] = index
