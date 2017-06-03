import random

def X_y(fname, shuffled=False):
    X = []
    y = []
    with open(fname) as f:
        lines = f.read().split('\n')
        if (shuffled):
            random.shuffle(lines)
        for line in lines:
            if len(line) <= 1:
                continue
            line = [int(x) for x in line.split(',')]
            X.append(line[:-1])
            y.append(line[-1])
    return X, y
def test():
    X, y = X_y("cloud.csv")
    print(X[0])

