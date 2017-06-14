import sys
import matplotlib.pyplot as plt
import numpy as np
import myutil
import random
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

X = list()
y = list()
with open('/users/yui/projects/accumu/feature/feature.csv') as f:
    lines = f.read().split('\n')
    random.shuffle(lines)
    for line in lines:
        if len(line) < 4:
            continue
        X.append([float(x) for x in line.split(',')[5:-1]])
        y.append(float(line.split(',')[-1]))

# X, y = myutil.X_y('cloud.csv', shuffled=True)

train_X, train_y, test_X, test_y = myutil.devide_train_test(X, y, 0.8)

# myutil.tune_hyperparameters(train_X, train_y)

clf = SVC(kernel='rbf', C=1, gamma=0.01)
clf.fit(train_X, train_y)

colors = ['#0000ff', '#ff0000']

# myutil.plot_heatmap(clf)

# for i in range(len(X)):
#     xx = X[i]
#     yy = y[i]
#     plt.plot(xx[0], xx[1], color=colors[yy], marker='.')

# plt.show()
myutil.profile(clf, train_X, train_y, test_X, test_y)
