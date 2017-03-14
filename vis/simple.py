import matplotlib.pyplot as plt
import numpy as np

x = np.arange(-4, 4, 0.1)
y = np.sin(x)
plt.plot(x, y, '.')
plt.text(1, 0, 'Hello World')
plt.show()

