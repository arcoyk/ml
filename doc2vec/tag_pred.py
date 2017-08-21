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
  tag_sims = hash()
  similars = model.get_similar(text, 5)
  for doc in similars(text):
    for tag in doc.tags:
      tag_sims[tag] += similarity(doc.text, text)
  return tag_sims.sort()
"""{'tag1':0.63, 'tag2':0.23, 'tag3':0.12}"""


def predict_kind(tag_sims):
  # kind = unique tags
  kinds = [['1-1','1-2','1-3'],['2-1','2-2'],['3-1','3-2','3-3']]
  kinded = list()
  for i in range(len(kinds)):
    kind_tags = kinds[i]
    tmp = list()
    for tag_sim in tag_sims:
      if tag_sim.key in kind_tags:
        tmp.append(tag_sim)
    kinded.append(tmp)
  return kinded



