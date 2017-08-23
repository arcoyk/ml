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

def hash_path2tag(paths):
  rst = {}
  for path in paths:
    rst[path] = path2tags(path)
  return rst

def path2title(path):
  return path.split('/')[-1]

def title2path(title):
  paths = search(list(), ROOT)
  for path in paths:
    if path2title(path) == title:
      return path
  return None

paths = search(list(), ROOT)
model = models.Doc2Vec.load(MODEL_DIR)
sims = myutil.similar_docs(model, 'サンプル.txt')
sim_paths = [title2path(sim[0]) for sim in sims]
p2t = hash_path2tag(paths)
for k, v in p2t.items():
  print(k, v)
