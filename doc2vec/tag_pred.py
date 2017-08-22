import os
root = '書類'
d = list()

def search(d, r):
  if r.find('.txt') != -1:
    d += [r]
    return
  else:
    for f in os.listdir(r):
      search(d, r + '/' + f)
  return d

def learn(list_of_tagged_docs):
  pass

def similar_tags(model, text):
  tag_sims = list()
  similars = model.get_similar(text, 5)
  for doc in similars(text):
    for tag in doc.tags:
      tag_sims += [tag, similarity(doc.text, text)]
  return tag_sims.sort()
# [['tag1',0.63], ['tag2',0.23], ['tag3',0.12]]


def predict_kind(tag_sims):
  # kind = unique tags
  kinds = [['1-1','1-2','1-3'],['2-1','2-2'],['3-1','3-2','3-3']]
  kinded = list()
  for i in range(len(kinds)):
    kind_tags = kinds[i]
    hits = list()
    for tag_sim in tag_sims:
      tag = tag_sim[0]
# should be one line
      if tag in kind_tags:
        hits.append(tag_sim)
    kinded.append(hits)
  return kinded
# [[['tag1',0.63], ['tag3',0.24]],[['tag2',0.32],['tag4',0.12]]]
# whereas tag1 and tag3, tag2 and tag4 is exclusive




















