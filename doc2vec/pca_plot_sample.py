from os import listdir
import logging
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from gensim.models.doc2vec import LabeledSentence
from gensim import models

ROOT = 'documents'
DOCS_DIR = './' + ROOT + '/'
MODEL_DIR = ROOT + '.model'

def doc2words(path):
  words = []
  with open(path) as f:
    words = f.read().split(' ')
  return words

def read_doc(path):
    words = doc2words(path)
    print(name)
    return LabeledSentence(words = words, tags=[name])

# Train
def train(docs_dir):
  logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
  sentences = [read_doc(doc_dir + x) for x in listdir(doc_dir)]
  model = models.Doc2Vec(sentences)
  for epoch in range(20):
      model.train(sentences, total_examples=model.corpus_count, epochs=model.iter)
      model.alpha -= (0.025 - 0.0001) / 19
      model.min_alpha = model.alpha
  model.save(MODEL_DIR)
  model = models.Doc2Vec.load(MODEL_DIR)
  return model

# PCA and Plot
def pca_plot(X, T, show=True):
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
  if show:
    plt.show()
  return X, T

# Vector and tag
def vectors_and_tags(model):
  X = [model.docvecs[tag] for tag in model.docvecs.doctags]
  T = [tag for tag in model.docvecs.doctags]
  return X, T

# Most similar by tag
def get_most_similar(model, tag):
  return model.docvecs.most_similar(tag)

def unseen_doc2vec(model, path):
  words = doc2words(path)
  return model.infer_vector(words)

# model = train()
model = models.Doc2Vec.load(MODEL_DIR)
X, T = vectors_and_tags(model)

X.append(unseen_doc2vec(model, 'unseen.txt'))
T.append('feajfoejaiwofjeiawofjeiaofjioewajfieaw')

X, T = pca_plot(X, T)
