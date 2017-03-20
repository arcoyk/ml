import random
import numpy as np
import matplotlib.pyplot as plt 
import point

data = []
csv_file = "cloud_hard.csv"
ps = point.csv_read(csv_file)
point.show(ps, '#ff0000', '#0000ff')
