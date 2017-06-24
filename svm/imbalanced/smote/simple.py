import matplotlib.pyplot as plt
import random
import numpy as np

def SMOTE(data, n=5):
    label = data[0][-1]
    rst = list()
    for i in range(n):
        m = random.sample(data, 1)[0]
        m_fs = m[:-1]
        X = [d[:-1] for d in data]
        X = sorted(X, key = lambda x : scala(np.array(x) - np.array(m_fs)))
        ns_fs = X[1:6]
        for n_fs in ns_fs:
            new_fs = list(np.array(m_fs) + (np.array(n_fs) - np.array(m_fs)) * random.random())
            new = new_fs + [label]
            rst.append(new)
    return rst

def scala(nparr):
    return np.power(np.power(nparr, 2).sum(), 0.5)
    
def test():
    with open('cloud.csv') as f:
        posi = list()
        nega = list()
        additional = list()
        for line in f.read().split('\n')[:-1]:
            line = [int(x) for x in line.split(',')]
            if line[-1] == 1:
                posi.append(line)
            else:
                nega.append(line)
        additional = SMOTE(posi)
        color_plot(posi, 'red')
        color_plot(nega, 'blue')
        color_plot(additional, 'green')
    plt.show()

def color_plot(points, c):
    for point in points:
        plt.plot(point[0], point[1], color=c, marker='.')

test()
