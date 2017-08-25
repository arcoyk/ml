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
      if not f == '.DS_Store':
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

def get_tags_menu(paths):
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

def pred_tagprobs_devided(tags_menu, tagprobs):
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

def get_tagprobs_all_combis(tagprobs_devided):
  rst = [[]]
  for tagprobs in tagprobs_devided:
    tmp = list()
    for r in rst:
      for tagprob in tagprobs:
        tmp.append(r + [tagprob])
    rst = tmp
  return rst

def unique(a):
  rst = []
  for i in a:
    if not i in rst:
      rst.append(i)
  return rst

def filter_tagprobs_possible_combis(possible_paths, tagprobs_all_combis):
  possible_tags = unique([path2tags(path) for path in possible_paths])
  rst = []
  for combi in tagprobs_all_combis:
    cand_tags = [tagprobs[0] for tagprobs in combi]
    if cand_tags in possible_tags:
      rst.append(combi)
  return rst

def combi2tags_and_prob(combi):
  tags = [tagprob[0] for tagprob in combi]
  prob = sum([tagprob[1] for tagprob in combi])
  return tags, prob

def pred_tags_and_prob(combis):
  cand = []
  for combi in combis:
    tags, prob = combi2tags_and_prob(combi)
    cand.append([tags, prob])
  cand.sort(key=lambda x:x[1])
  cand.reverse()
  return cand[0][0], cand[0][1]

def path2path(model, path):
  all_paths = search(list(), ROOT)
  paths, probs = myutil.similar_docs(model, path)
  tagprobs = get_tagprobs(paths, probs)
  tags_menu = get_tags_menu(paths)
  tagprobs_devided = pred_tagprobs_devided(tags_menu, tagprobs)
  tagprobs_all_combis = get_tagprobs_all_combis(tagprobs_devided)
  tagprobs_possible_combis = filter_tagprobs_possible_combis(all_paths, tagprobs_all_combis)
  tags, prob = pred_tags_and_prob(tagprobs_possible_combis)
  path = ('/').join(tags)
  return path, prob

# model = myutil.train(paths)
# model.save(MODEL_DIR)
model = models.Doc2Vec.load(MODEL_DIR)
path, prob = path2path(model, 'サンプル.txt')
print(path, prob)
