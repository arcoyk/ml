from os import listdir
import MeCab
import logging
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from gensim.models.doc2vec import LabeledSentence
from gensim import models
from sklearn.metrics.pairwise import cosine_similarity

ROOT = 'documents'
DOCS_DIR = './' + ROOT + '/'
MODEL_DIR = ROOT + '.model'

def doc2words(path, wakati=False):
  words = []
  if wakati:
    tagger = MeCab.Tagger("-Owakati")        
    words = tagger.parse(text).split(' ')
  else:
    with open(path) as f:
      words = f.read().split(' ')
  return words

def read_doc(path):
    words = doc2words(path)
    print(name)
    return LabeledSentence(words = words, tags=[name])

def get_paths(r):
  return [r + '/' + d for d in os.listdir(r)]

# Train
def train(paths):
  logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
  sentences = [read_doc(path) for path in paths]
  model = models.Doc2Vec(sentences)
  for epoch in range(20):
      model.train(sentences, total_examples=model.corpus_count, epochs=model.iter)
      model.alpha -= (0.025 - 0.0001) / 19
      model.min_alpha = model.alpha
  model.save(MODEL_DIR)
  model = models.Doc2Vec.load(MODEL_DIR)
  return model

def plot(X, T):
  plt.plot(X[:,0], X[:,1], 'x')
  for i in range(len(X)):
      x = X[i][0]
      y = X[i][1]
      t = T[i]
      plt.text(x, y, t)
  plt.show()

# PCA
def pca(X, T, show=False):
  # PCA
  X = np.array(X)
  pca = PCA(n_components=2)
  X = pca.fit_transform(X)
  # Plot
  if show:
    plot(X, T)
  return X, T

# Vector and tag
def vectors_and_tags(model):
  T = [tag for tag in model.docvecs.doctags]
  X = [model.docvecs[tag] for tag in T]
  return X, T

def unseen_pca(model, path, show=False):
  X, T = vectors_and_tags(model)
  X.append(unseen_vec(model, path))
  T.append(path)
  return pca(X, T, show)

# Vector of unseen document
def unseen_vec(model, path):
  words = doc2words(path)
  return model.infer_vector(words)

def similar_docs_by_vec(model, vec, top_n=5):
  X, T = vectors_and_tags(model)
  rst = list()
  for i in range(len(T)):
    rst.append([T[i], cosine_similarity(X[i], vec)])
  rst.sort(key=lambda x:x[1])
  rst.reverse()
  return rst[:top_n]

def similar_docs(model, tag):
  if not tag in model.docvecs.doctags:
    print("Tag not found in [similar_docs_by_tag]")
    return unseen_similars(model, tag)
  vec = model.docvecs[tag]
  return similar_docs_by_vec(model, vec)

def unseen_similars(model, path):
  vec = unseen_vec(model, path)
  return similar_docs_by_vec(model, vec)

# paths = get_paths(MODEL_DIR)
# model = train(paths)
model = models.Doc2Vec.load(MODEL_DIR)
# unseen_pca(model, 'unseen.txt', show=True)
# sims = similar_docs(model, 'unseen.txt')
