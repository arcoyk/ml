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

def doc2words(path, wakati=False):
  words = []
  if wakati:
    tagger = MeCab.Tagger("-Owakati")        
    with open(path) as f:
      words = tagger.parse(f.read()).split(' ')
  else:
    with open(path) as f:
      words = f.read().split(' ')
  return words

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

def read_doc(path):
  words = doc2words(path, wakati=True)
  return LabeledSentence(words = words, tags=[doc_title(path)])

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

def unseen_vec(model, path):
  words = doc2words(path, wakati=True)
  return model.infer_vector(words)

def unseen_similars(model, path):
  vec = unseen_vec(model, path)
  return model.similar_by_vector(vec)

paths = search(paths, ROOT)
paths_tags = get_paths_tags(paths)
# model = train(paths)
model = models.Doc2Vec.load(MODEL_DIR)
sims = unseen_similars(model, "サンプル.txt")
print(sims)


