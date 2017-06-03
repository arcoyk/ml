import random
import sys
import numpy as np
from time import time
from scipy.stats import randint as sp_randint
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support

def X_y(fname, shuffled=False):
    X = []
    y = []
    with open(fname) as f:
        lines = f.read().split('\n')
        if (shuffled):
            random.shuffle(lines)
        for line in lines:
            if len(line) <= 1:
                continue
            line = [int(x) for x in line.split(',')]
            X.append(line[:-1])
            y.append(line[-1])
    return X, y

def devide_train_test(X, y, train_rate):
    train_amount = int(len(X) * train_rate)
    train_X, train_y = X[:train_amount], y[:train_amount]
    test_X, test_y = X[train_amount:], y[train_amount:]
    return train_X, train_y, test_X, test_y

def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")

def tune_hyperparameters(X, y, random_search=False, grid_search=False):
    clf = RandomForestClassifier(n_estimators=20, class_weight='balanced')
    if random_search:
        param_dist = {"max_depth": [3, None],
                      "max_features": sp_randint(1, 2),
                      "min_samples_split": sp_randint(2, 6),
                      "min_samples_leaf": sp_randint(1, 11),
                      "bootstrap": [True, False],
                      "criterion": ["gini", "entropy"]}
        n_iter_search = 20
        random_search = RandomizedSearchCV(clf, param_distributions=param_dist,
                                           n_iter=n_iter_search)
        start = time()
        random_search.fit(X, y)
        print("RandomizedSearchCV took %.2f seconds for %d candidates"
              " parameter settings." % ((time() - start), n_iter_search))
        report(random_search.cv_results_)
    
    if grid_search:
        param_grid = {"max_depth": [3, None],
                      "max_features": [1, 2],
                      "min_samples_split": [2, 6],
                      "min_samples_leaf": [1, 3, 10],
                      "bootstrap": [True, False],
                      "criterion": ["gini", "entropy"]}
        grid_search = GridSearchCV(clf, param_grid=param_grid)
        start = time()
        grid_search.fit(X, y)
        print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
              % (time() - start, len(grid_search.cv_results_['params'])))
        report(grid_search.cv_results_)

def validate(clf, test_X, test_y):
    pred_y = clf.predict(test_X)
    return precision_recall_fscore_support(test_y, pred_y, average='weighted')

def profile(clf, train_X, train_y, test_X, test_y):
    full = len(train_X) + len(test_X)
    trues = sum(train_y) + sum(test_y)
    falses = full - trues
    pr, rc, fs, su = validate(clf, test_X, test_y)
    print("Fulldata", full)
    print("Trues", trues, trues / full)
    print("Falses", falses, falses / full)
    print("Featuers", clf.n_features_)
    print("Feature Importance", clf.feature_importances_)
    print("Precision", pr)
    print("Recall", rc)
    print("Fscore", fs)
    print("Support", su)

