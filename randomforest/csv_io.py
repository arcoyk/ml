def X_y(fname):
    X = []
    y = []
    with open(fname) as f:
        lines = f.read()
        for line in lines.split('\n'):
            if len(line) <= 1:
                continue
            line = [int(x) for x in line.split(',')]
            X.append(line[:-1])
            y.append(line[-1])
    return X, y
def test():
    X, y = X_y("cloud.csv")
    print(X[0])

