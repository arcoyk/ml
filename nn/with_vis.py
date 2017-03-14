# 3 - 2 - 2 - 1 NN ReLu
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
    mark = ['x', 'o'][t]
    plt.plot(x, y, mark)

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

w1 = []
for i in range(2):
    w1.append([rand(), rand(), rand()])
w2 = []
for i in range(2):
    w2.append([rand(), rand()])
w3 = []
for i in range(1):
    w3.append([rand(), rand()])

x = [rand(), rand(), rand()]
print(x)
x = layer_calc(x, w1)
print(x)
x = layer_calc(x, w2)
print(x)
x = layer_calc(x, w3)
print(x)
