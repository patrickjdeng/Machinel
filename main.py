import parse
import decisionTree
import predict
import neuralNet
import neighbors
import naiveBayes
import numpy as np
from collections import Counter

X,Y = parse.read()
print(len(X))
print(len(Y))

X = np.array(X)
X = X.astype(np.float64)

tree = decisionTree.trainTree(X,Y)
#nn = neuralNet.trainNet(X,Y)
neighbor = neighbors.trainNeighbors(X,Y)
bayes = naiveBayes.trainBayes(X,Y)

attr = []

with open('data/prelim-nmv-noclass.txt','r') as infile:
    for x in range (0,10):
        line = infile.readline()
        attrs = line.split()
        attrs.pop()
        attr.append(attrs)

attr = np.array(attr)
attr = attr.astype(np.float64)

treePredictions = predict.predictions(tree,attr)
neigborPredictions = predict.predictions(neighbor,attr)
bayesPredictions = predict.predictions(bayes,attr)

print(treePredictions)
print(neigborPredictions)
print(bayesPredictions)

rawPredictions = zip(treePredictions,neigborPredictions,bayesPredictions)

predictions = []
for p in rawPredictions:
    data = Counter(p)
    predictions.append(int(data.most_common(1)[0][0]))

with open('predictions.txt','w') as outfile:
    for p in predictions:
        print >> outfile, p