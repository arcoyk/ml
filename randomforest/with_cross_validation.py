from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation
from sklearn.decomposition import PCA
import random
import numpy as np
import matplotlib.pyplot as plt

features = []
for i in range(100):
    x = random.random()
    y = random.random()
    z = random.random()
    zero_one = i % 2
    features.append([x, y, z + zero_one, zero_one])

def X_y(features):
    X = np.array([f[0:-1] for f in features])
    y = np.array([f[-1] for f in features])
    return X, y

def learn(params, trues):
    model = RandomForestClassifier()
    model.fit(params, trues)
    return model

def predict(model, params):
    return [model.predict(param) for param in params]

def validate(features):
    X, y = X_y(features)
    k = (int)(1 + np.log(len(X)) / np.log(2)) * 4
    k = 5
    k_fold = cross_validation.KFold(n=len(X), n_folds=k, shuffle=True)
    print("%d fold validation" % k)
    train_scores = list()
    test_scores = list()
    model = RandomForestClassifier()
    for train, test in k_fold:
        model.fit(X[train], y[train])
        train_scores.append(model.score(X[train], y[train]))
        test_scores.append(model.score(X[test], y[test]))
        if False:
            plot_pca_features(X[train], predict(model, X[train]), 'x')
            plot_pca_features(X[test], predict(model, X[test]), 'o')
            plt.show()
    return np.mean(train_scores), np.mean(test_scores)

def plot_pca_features(X, y, mark):
    pca = PCA(n_components=2)
    # X = pca.fit_transform(X)
    for i in range(len(X)):
        c = ['red', 'blue'][y[i]]
        plt.plot(X[i][0], X[i][1], mark, color=c)

print(validate(features))
