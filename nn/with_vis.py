# 2 - 2 - 2 - 1 NN ReLu
import random
import matplotlib.pyplot as plt
import numpy as np

def rand():
    return random.random() - random.random()

'''=============== Sample ============'''
data = []
for i in range(100):
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

plt.show()

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
    return x[0]

def train(data):
    d = 0
    for p in data:
        x = [p[0], p[1]]
        t = p[2]
        x1 = layer_calc(x, w1)
        x1 = layer_calc(x, w2)
        x1 = layer_calc(x, w3)
        d += abs(define(x1) - t)
    error_rate = d / len(data)
    return error_rate

w1 = []
for i in range(2):
    w1.append([rand(), rand()])
w2 = []
for i in range(2):
    w2.append([rand(), rand()])
w3 = []
for i in range(1):
    w3.append([rand(), rand()])

def get_slope(w, data):
    slope = []
    for wi in w:
        dec = wi - d
        inc = wi + d
        dx = inc - dec
        dy = train(data)

d = 0.01
alpha = 0.01
for i in range(100):
    slope1 = get_slope(w1, data)
    slope2 = get_slope(w2, data)
    slope3 = get_slope(w3, data)
    dec = w1 - d
    inc = w1 + d
    dx = inc - dec
    dy = train(data) - train(data)
    slope = dy / dx
    w1 -= slope * alpha
    w1 = [ w1[i] - slope1[i] * alpha for i in range(len(w1))]
    w2 = [ w2[i] - slope2[i] * alpha for i in range(len(w2))]
    w3 = [ w3[i] - slope3[i] * alpha for i in range(len(w3))]
