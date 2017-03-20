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

def show(points, colors):
    for point in points:
        x = point[0]
        y = point[1]
        c = colors[point[2] % len(colors)]
        plt.plot(x, y, '.', color=c)
    plt.show()

def plot(points, colors):
    for point in points:
        x = point[0]
        y = point[1]
        c = colors[point[2] % len(colors)]
        plt.plot(x, y, '.', color=c)
    return plt

