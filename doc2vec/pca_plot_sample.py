from os import listdir
import logging
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from gensim.models.doc2vec import LabeledSentence
from gensim import models

DOCS_DIR = './sample/science/'
MODEL_DIR = 'science.model'

def read_doc(path):
    words = []
    with open(path) as f:
        words = f.read().split(' ')
    name = path.split('/')[-1]
    print(name)
    return LabeledSentence(words = words, tags=[name])

# Train
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
sentences = [read_doc(DOCS_DIR + x) for x in listdir(DOCS_DIR)]
model = models.Doc2Vec(sentences)

for epoch in range(200):
    model.train(sentences)
    model.alpha -= (0.025 - 0.0001) / 19
    model.min_alpha = model.alpha

model.save("science.model")
model = models.Doc2Vec.load("science.model")

X = [model.docvecs[tag] for tag in model.docvecs.doctags]
T = [tag for tag in model.docvecs.doctags]

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
