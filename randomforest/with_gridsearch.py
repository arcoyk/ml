import sys
import numpy as np
import csv_io
import point
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV

def train_test(X, y, train_rate):
    train_amount = int(len(X) * train_rate)
    train_X, train_y = X[:train_amount], y[:train_amount]
    test_X, test_y = X[train_amount:], y[train_amount:]
    return train_X, train_y, test_X, test_y

X, y = csv_io.X_y('cloud.csv')
model = RandomForestClassifier(oob_score=True)
train_X, train_y, test_X, test_y = train_test(X, y, 0.8)

""" 
Random Forest has two hyperparameters: ntree and mtry.
ntree is the amount of trees of the forest.
mtry is the amount of features used for one tree.
First, mtry is roughly tuned, then ntrees is tuned.
"""

# Grid Search for mtry
# params = {'max_features': [2, 5, 10], 'n_jobs': [-1]}
# cv = GridSearchCV(model, params, cv=10, scoring='neg_mean_squared_error', n_jobs=1)
# cv.fit(train_X, train_y)
# print("Grid searched", model.oob_score_)
model.fit(X, y)
print(model.oob_score_)

params = {'n_estimators': [10, 50, 200], 'n_jobs': [-1]}
cv = GridSearchCV(model, params, cv=5, scoring='neg_mean_squared_error', n_jobs=1)
cv.fit(train_X, train_y)
print("Grid searched", model.oob_score_)

model.fit(X, y)
print(model.oob_score_)
