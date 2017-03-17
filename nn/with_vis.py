# 2 - 2 - 2 - 1 NN ReLu
import random
import matplotlib.pyplot as plt
import numpy as np

def rand():
    return random.random() - random.random()

'''=============== Sample ============'''
data = []
for i in range(500):
    x = i % 2 + rand()
    y = i % 2 + rand()
    t = i % 2
    data.append([x, y, t])

for p in data:
    x = p[0]
    y = p[1]
    t = p[2]
    mark = ['x', '.'][t]
    plt.plot(x, y, mark, color='#aaaaaa')

'''=============== NN ================='''
def ReLu(v):
    return max(0, v)

def layer_calc(x, w):
    ans = []
    for wi in w:
        v = 0
        for i in range(len(x)):
            v += wi[i] * x[i]
        ans.append(ReLu(v))
    return ans

def define(x):
    if x[0] < 0.2:
        return 0
    else:
        return 1

def fire(x):
    for wx in w:
        x = layer_calc(x, wx)
    return define(x)

def error_rate():
    error_sum = 0
    for p in data:
        x = p[:-1]
        true_value = p[-1]
        error_sum += abs(fire(x) - true_value)
    error_rate = error_sum / len(data)
    return error_rate

def matrix(row_n, col_n):
    mat = []
    for y in range(col_n):
        col = []
        for x in range(row_n):
            col.append(rand())
        mat.append(col)
    return mat

def print_matrix(w):
    for wx in w:
        for wxx in wx:
            print(wxx)
        print('-------')

def layer(nn):
    w = []
    for i in range(len(nn) - 1):
        w.append(matrix(nn[i], nn[i + 1]))
    return w


def get_slope(wx):
    slope = [[0 for y in x] for x in wx]
    for xi in range(len(wx)):
        for yi in range(len(wx[xi])):
            tmp = wx[xi][yi]
            inc = wx[xi][yi] + d
            dec = wx[xi][yi] - d
            wx[xi][yi] = inc
            inc_e = error_rate()
            wx[xi][yi] = dec
            dec_e = error_rate()
            dx = inc - dec
            dy = inc_e - dec_e
            slope[xi][yi] = dy / dx
            wx[xi][yi] = tmp
    return slope

def train(d, alpha, epoch):
    for i in range(epoch):
        for i in range(len(w)):
            slope = get_slope(w[i])
            w[i] = [[w[i][xi][yi] - slope[xi][yi] * alpha for yi in range(len(w[i][xi]))] for xi in range(len(w[i]))]
        print("E:", error_rate())

w = layer([2, 1])
d = 0.1
alpha = 0.2
epoch = 10
train(d, alpha, epoch)

'''=============== result ============='''
for x in np.arange(-1, 2, 0.1):
    for y in np.arange(-1, 2, 0.1):
        v = fire([x, y])
        c = ['#ffaaaa', '#aaaaff'][v]
        plt.plot(x, y, '.', color=c)
plt.show()

