# 3 - 2 - 2 - 1 NN ReLu
import random

def rand():
    return random.random() - random.random()

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
for x in range(2):
    w1.append([rand(), rand(), rand()])
w2 = []
for x in range(2):
    w2.append([rand(), rand()])
w3 = []
for x in range(1):
    w3.append([rand(), rand()])

x = [rand(), rand(), rand()]
print(x)
x = layer_calc(x, w1)
print(x)
x = layer_calc(x, w2)
print(x)
x = layer_calc(x, w3)
print(x)
