import logging
from gensim.models import word2vec

TEXT_DIR = 'abcde.txt'
MODEL_DIR = 'abcde.model'

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
sentences = word2vec.LineSentence(TEXT_DIR)
model = word2vec.Word2Vec(sentences)
model.save(MODEL_DIR)

for w in model.wv.index2word:
    print(w, model.most_similar(w)[:3])
