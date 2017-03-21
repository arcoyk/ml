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

