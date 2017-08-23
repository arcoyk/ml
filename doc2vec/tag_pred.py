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

def add_or_new(h, key, val):
  if key in h:
    h[key] += val
  else:
    h[key] = 0
  return h

def get_tagprobs(paths, probs):
  tagprobs = {}
  for i in range(len(paths)):
    path = paths[i]
    prob = probs[i]
    tags = path2tags(path)
    for tag in tags:
      tagprobs = add_or_new(tagprobs, tag, prob)
  rst = []
  for k, v in tagprobs.items():
    rst.append([k, float(v)])
  rst.sort(key=lambda x:x[1])
  rst.reverse()
  return rst

paths = search(list(), ROOT)
# model = myutil.train(paths)
# model.save(MODEL_DIR)
model = models.Doc2Vec.load(MODEL_DIR)
paths, probs = myutil.similar_docs(model, 'サンプル.txt')
tagprobs = get_tagprobs(paths, probs)
for tagprob in tagprobs:
  print(tagprob)
