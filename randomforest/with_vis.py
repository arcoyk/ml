import sys
sys.path.append('../util')
from sklearn.ensemble import RandomForestClassifier
import point
import numpy as np
import 
ps = point.csv_read('cloud.csv')
# point.show(ps, ['#ff0000', '#00ff00'])

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

plt = point.plot(pred_grid, '#ff0000', '#00ff00')

