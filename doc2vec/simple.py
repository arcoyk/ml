import os
import logging
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from subprocess import call
from gensim.models.doc2vec import LabeledSentence
from gensim import models

TRAIN = False
DIR = 'sample/science/'
MODEL = 'doc2vec.model'

def get_content(txt_path):
    content = ''
    with open(txt_path) as f:
        content = f.read()
    return content

def read_docs():
    rst = list()
    for d in os.listdir(DIR):
        if d.find('.txt') != -1:
            content = get_content(DIR + d)
            title = d
            words = content.split(' ')
            rst.append(LabeledSentence(words = words, tags=[title]))
    return rst

if TRAIN:
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    sentences = read_docs()
    model = models.Doc2Vec(sentences)
    for epoch in range(20):
        model.train(sentences, total_examples=model.corpus_count, epochs=model.iter)
        model.alpha -= (0.025 - 0.0001) / 19
        model.min_alpha = model.alpha
    model.save(MODEL)

