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

def add_or_new(h, key, val, default=0):
  if key in h:
    if type(h[key]) is list:
      h[key].append(val)
    else:
      h[key] += val
  else:
    h[key] = default
  return h

def hash2list(h):
  r = []
  for k, v in h.items():
    r.append([k, v])
  return r

def get_tagprobs(paths, probs):
  tagprobs = {}
  for i in range(len(paths)):
    path = paths[i]
    prob = probs[i]
    tags = path2tags(path)
    for tag in tags:
      tagprobs = add_or_new(tagprobs, tag, prob)
  rst = hash2list(tagprobs)
  rst.sort(key=lambda x:x[1])
  rst.reverse()
  return rst

def tags_menu(paths):
  tags_list = [path2tags(path) for path in paths]
  tmp = {}
  for tags in tags_list:
    for i in range(len(tags)):
      tag = tags[i]
      tmp = add_or_new(tmp, i, tag, default=[])
  rst = []
  for r in hash2list(tmp):
    rst.append(list(set(r[1])))
  return rst

def pred_tags(tags_menu, tagprobs):
  rst = []
  for i in range(len(tags_menu)):
    tags = tags_menu[i]
    tmp = []
    for tagprob in tagprobs:
      if tagprob[0] in tags:
        tmp.append(tagprob)
    tmp.sort(key=lambda x:x[1])
    tmp.reverse()
    rst.append(tmp)
  return rst

paths = search(list(), ROOT)
# model = myutil.train(paths)
# model.save(MODEL_DIR)
model = models.Doc2Vec.load(MODEL_DIR)
paths, probs = myutil.similar_docs(model, 'サンプル.txt')
tagprobs = get_tagprobs(paths, probs)
tags_menu = tags_menu(paths)

r = pred_tags(tags_menu, tagprobs)
for i in r:
  print(i)
