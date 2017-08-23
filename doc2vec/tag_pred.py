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

def get_tags(path):
  return path.split('/')[:-1]

def get_paths_tags(paths):
  rst = {}
  for path in paths:
    rst[path] = get_tags(path)
  return rst

def doc_title(path):
  return path.split('/')[-1]

model = models.Doc2Vec.load(MODEL_DIR)
sims = myutil.similar_docs(model, 'サンプル.txt')
for sim in sims:
  print(sim)
