from gensim.models.doc2vec import LabeledSentence
from gensim import models
from os import listdir

def read_doc(path):
    words = []
    with open(path) as f:
        words = f.read().split(' ')
    name = path.split('/')[-1]
    print(name)
    return LabeledSentence(words = words, tags=[name])

DIR = "./sample/science/"
sentences = [read_doc(DIR + x) for x in listdir(DIR)]
model = models.Doc2Vec(sentences)
for epoch in range(20):
    print('Epoch:', epoch + 1)
    model.train(sentences)
    model.alpha -= (0.025 - 0.0001) / 19
    model.min_alpha = model.alpha

model.save("science.model")
model = models.Doc2Vec.load("science.model")

for tag in model.docvecs.doctags:
    print(model.docvecs[tag])

