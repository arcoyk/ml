from sklearn.decomposition import PCA
import numpy as np
import random
import matplotlib.pyplot as plt

x = [random.random() for x in range(100)]
y = [random.random() for y in range(100)]
z = [z % 2 for z in range(100)]
X = [[x[i], y[i], z[i]] for i in range(100)]

X = np.array(X)
pca = PCA(n_components=2)
X = pca.fit_transform(X)

plt.plot(X[:,0], X[:,1], 'x')
plt.show()
