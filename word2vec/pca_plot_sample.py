from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import logging
from gensim.models import word2vec

TEXT_DIR = 'abcde.txt'
MODEL_DIR = 'abcde.model'

# Train
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
sentences = word2vec.LineSentence(TEXT_DIR)
model = word2vec.Word2Vec(sentences)

for epoch in range(20):
    model.train(sentences)
    model.alpha -= (0.025 - 0.0001) / 19
    model.min_alpha = model.alpha

model.save(MODEL_DIR)

X = [model[w] for w in model.wv.index2word]
T = [w for w in model.wv.index2word]

# PCA
X = np.array(X)
pca = PCA(n_components=2)
X = pca.fit_transform(X)

# Plot
plt.plot(X[:,0], X[:,1], 'x')
for i in range(len(X)):
    x = X[i][0]
    y = X[i][1]
    t = T[i]
    plt.text(x, y, t)

plt.show()
