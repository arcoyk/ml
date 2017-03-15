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

# plt.show()

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

for i in range(100):
    a1 = [0, 0]
    a2 = [0, 0]
    a3 = [0]
    
