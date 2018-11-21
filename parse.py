def read():
    X = []
    Y = []
    with open('data/train.nmv.txt') as infile:
        line = infile.readline()
        while line:
            attributes = line.split()
            Y.append(attributes.pop())
            X.append(attributes)
            line = infile.readline()
    return X,Y 