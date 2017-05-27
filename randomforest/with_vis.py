import sys
import numpy as np
import csv_io
import point
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

X, y = csv_io.X_y('cloud.csv')

model = RandomForestClassifier()
model.fit(X, y)

grid = []
for xx in np.arange(0, 500, 20):
    for yy in np.arange(0, 500, 20):
        grid.append([xx, yy])
pred = model.predict(grid)

point.plot_X_y(X, y)
point.plot_X_y(grid, pred, 'x')
point.show()
