from os import listdir
import logging
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from gensim.models.doc2vec import LabeledSentence
from gensim import models

ROOT = '書類'
DOCS_DIR = './' + ROOT + '/'
MODEL_DIR = ROOT + '.model'

def read_doc(path):
    words = []
    with open(path) as f:
        words = f.read().split(' ')
    name = path.split('/')[-1]
    print(name)
    return LabeledSentence(words = words, tags=[name])

# Train
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
sentences = [read_doc(DOCS_DIR + x) for x in listdir(DOCS_DIR)]
model = models.Doc2Vec(sentences)

for epoch in range(20):
    model.train(sentences, total_examples=model.corpus_count, epochs=model.iter)
    model.alpha -= (0.025 - 0.0001) / 19
    model.min_alpha = model.alpha

model.save(MODEL_DIR)
model = models.Doc2Vec.load(MODEL_DIR)


