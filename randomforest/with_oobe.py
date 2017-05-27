import sys
import numpy as np
import csv_io
import point
from sklearn.ensemble import RandomForestClassifier

X, y = csv_io.X_y('cloud.csv')
point.show_X_y(X, y)

params = [x[:-1] for x in ps]
labels = [x[-1] for x in ps]

model = RandomForestClassifier()
model.fit(params, labels)

grid = []
for x in np.arange(0, 500, 20):
    for y in np.arange(0, 500, 20):
        grid.append([x, y])
pred = model.predict(grid)

pred_grid = []
for i in range(len(t)):
    x = grid[i][0]
    y = grid[i][1]
    t = pred[i]
    pred_grid.append([x, y, t])


