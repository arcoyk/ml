import sys
import numpy as np
import csv_io
import point
from sklearn.ensemble import RandomForestClassifier

X, y = csv_io.X_y('cloud.csv')

model = RandomForestClassifier(oob_score=True)
model.fit(X, y)

grid = []
for xx in np.arange(0, 500, 20):
    for yy in np.arange(0, 500, 20):
        grid.append([xx, yy])
pred = model.predict(grid)

# point.plot(X, y)
# point.plot(grid, pred, 'x')
# point.show()
print(model.oob_score_)
