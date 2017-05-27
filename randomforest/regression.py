from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import numpy as np

X = np.arange(0, np.pi * 10, np.pi / pow(2, 4))
y = [np.sin(x / 2) + np.sin(x) for x in X]

for i in range(len(X)):
    plt.plot(X[i], y[i], '.', color="#000000")

model = RandomForestRegressor()
model.fit(X, y)

plt.show()
