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
    pca_plot(X, t)
    show()

def plot(X, y, mark='.'):
    colors = ["#ff0000", "#0000ff"]
    for i in range(len(X)):
        x = X[i]
        plt.plot(x[0], x[1], mark, color=colors[y[i]])
def pca_plot(X, y, mark='.'):
    colors = ["#ff0000", "#0000ff"]
    X = np.array(X)
    pca = PCA(n_components=2)
    X = pca.fit_transform(X)
    plot(X, y, mark=mark)

def show():
    plt.show()

