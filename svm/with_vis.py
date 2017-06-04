import sys
import matplotlib.pyplot as plt
import numpy as np
import csv_io
from sklearn.svm import SVC

def heatmap(clf):
    gx = np.linspace(0, 500, 100)
    gy = np.linspace(0, 500, 100)
    pred = list()
    for yy in gy:
        row = list()
        for xx in gx:
            row.append([xx, yy])
        pred.append(clf.predict(row))
    plt.contourf(gx, gy, pred, 100, alpha=0.3)
    plt.colorbar()

X, y = csv_io.X_y('cloud.csv')

clf = SVC()
clf.fit(X, y)

colors = ['#0000ff', '#ff0000']

heatmap(clf)

for i in range(len(X)):
    xx = X[i]
    yy = y[i]
    plt.plot(xx[0], xx[1], color=colors[yy], marker='.')

plt.show()

# print(clf.score(X, y))

