import pca_plot_sample as myutil
import os
import logging
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from gensim.models.doc2vec import LabeledSentence
from gensim import models
import MeCab

ROOT = '書類'
MODEL_DIR = ROOT + '.model'
paths = list()

def search(d, r):
  if r.find('.txt') != -1:
    d += [r]
    return
  else:
    for f in os.listdir(r):
      search(d, r + '/' + f)
  return d

def path2tags(path):
  return path.split('/')[:-1]

paths = search(list(), ROOT)
# model = myutil.train(paths)
# model.save(MODEL_DIR)
model = models.Doc2Vec.load(MODEL_DIR)
sims = myutil.similar_docs(model, 'サンプル.txt')

