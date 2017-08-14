import logging
from gensim.models import word2vec

TEXT_DIR = 'abcde.txt'
MODEL_DIR = 'abcde.model'
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
sentences = word2vec.LineSentence(TEXT_DIR)
model = word2vec.Word2Vec(sentences)

for epoch in range(20):
    model.train(sentences)
    model.alpha -= (0.025 - 0.0001) / 19
    model.min_alpha = model.alpha

model.save(MODEL_DIR)

for w in model.wv.index2word:
    print(w, model.most_similar(w)[:3])
