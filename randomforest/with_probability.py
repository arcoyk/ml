import sys
import numpy as np
import csv_io
import point
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

X, y = csv_io.X_y('cloud.csv')

model = RandomForestClassifier()
model.fit(X, y)

grid = []
for xx in np.arange(0, 500, 20):
    for yy in np.arange(0, 500, 20):
        grid.append([xx, yy])
pred = model.predict_proba(grid)

for i in range(len(X)):
    p = X[i]
    v = y[i]
    plt.plot(p[0], p[1], '.', color=["#ff0000", "#0000ff"][v])

for i in range(len(grid)):
    p = grid[i]
    v = pred[i]
    plt.plot(p[0], p[1], 's', alpha=max(v), color=["#ff0000", "#0000ff"][v[0] <= 0.5])

plt.show()
