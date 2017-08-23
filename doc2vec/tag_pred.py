import os
ROOT = '書類'
paths = list()

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

paths = search(paths, ROOT)
paths_tags = get_paths_tags(paths)
