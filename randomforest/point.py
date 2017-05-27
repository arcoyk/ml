import matplotlib.pyplot as plt
import random
from sklearn.decomposition import PCA
import numpy as np

def test():
    x = [random.random() + 1 * x % 2 for x in range(100)]
    y = [random.random() for y in range(100)]
    z = [random.random() for z in range(100)]
    t = [t % 2 for t in range(100)]
    X = [[x[i], y[i], z[i]] for i in range(100)]
    show(X, t)

def show_X_y(X, y):
    colors = ["#ff0000", "#0000ff"]
    X = np.array(X)
    pca = PCA(n_components=2)
    X = pca.fit_transform(X)
    for i in range(len(X)):
        x = X[i]
        plt.plot(x[0], x[1], '.', color=colors[y[i]])
    plt.show()

