import random
import numpy as np
import matplotlib.pyplot as plt 
def csv_read(fname):
    data = []
    with open(fname) as f:
        lines = f.read()
        for line in lines.split('\n'):
            if len(line) <= 1:
                continue
            line = [int(x) for x in line.split(',')]
            data.append(line)
    return data

data = []
csv_file = "cloud_hard.csv"
points = csv_read(csv_file)
points.show(ps, '#ff0000', '#0000ff')
