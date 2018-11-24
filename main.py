import sys
from collections import Counter
import parse
import decisionTree
import predict
import neuralNet
import neighbors
import naiveBayes
import numpy as np
import random

NAME = 0
VALUE_LIST = 1

def find_average(index, instance_set):
    '''given index, set, find average of value at index for each entry in set'''
    total = 0
    for row in instance_set:
        if row[index] != '?':
            total += float(row[index])
    return total/len(instance_set)

def find_most_common(attr_index, value_set, instance_set):
    '''given index, possible value set, training set, find most common of value at index for each entry in set'''
    counts = []
    for _ in range(len(value_set)):
        counts.append(0)
    for row in instance_set:
        if row[attr_index] != '?':
            #find index of value in value set and increment count of it
            value = row[attr_index]
            value_index = value_set.index(value)
            counts[value_index] += 1
    
    max_count = 0
    for count in counts:
        if count > max_count:
            max_count = count
    max_index = counts.index(max_count)
    return value_set[max_index]

def fill_in_missing_values(attr_types, set):
    #fill in missing values based on most common/average
    for i in range(len(attr_types) - 1):
        if attr_types[i][NAME][0] == 'C':
            attr_value = find_average(i,set) 
        else:
            attr_values_list = attr_types[i][VALUE_LIST]
            attr_value = find_most_common(i, attr_values_list, set)
        for j in range(len(set)):
            if set[j][i] == '?':
                set[j][i] = attr_value

def main():
    attribute_filename = sys.argv[1]
    training_filename = sys.argv[2]
    test_filename = sys.argv[3]   #parametrized name

    attr_types = parse.read_attribute_file(attribute_filename)

    X,Y = parse.read_training_file(training_filename)

    print(len(X))
    print(len(Y))


    fill_in_missing_values(attr_types, X)
    X = np.array(X)
    X = X.astype(np.float64)


    tree = decisionTree.trainTree(X,Y)
    #nn = neuralNet.trainNet(X,Y)
    neighbor = neighbors.trainNeighbors(X,Y)
    bayes = naiveBayes.trainBayes(X,Y)

    attr = []


    with open(test_filename,'r') as infile:
        for x in range (0,10):
            line = infile.readline()
            attrs = line.split()
            attrs.pop()
            attr.append(attrs)


    fill_in_missing_values(attr_types, attr)
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


main()