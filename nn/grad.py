import matplotlib.pyplot as plt
import numpy as np

def func(x):
    return x * x

x = np.arange(-1, 1, 0.1)
y = [func(xi) for xi in x]
i = -1
for k in range(100):
    dec = i - 0.01
    inc = i + 0.01
    dx = inc - dec
    dy = func(inc) - func(dec)
    slope = dy / dx
    i -= slope * 0.3
    plt.plot(i, func(i), '.')
plt.plot(x, y, 'x')
plt.show()

