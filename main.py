import sys
from collections import Counter
import parse
import decisionTree
import predict
import neuralNet
import neighbors
import naiveBayes
import svc
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
        attr_name = attr_types[i][NAME]
        if attr_name[0] == 'C':
            attr_value = find_average(i,set) 
        elif attr_name[0] == 'D':
            attr_values_list = attr_types[i][VALUE_LIST]
            attr_value = find_most_common(i, attr_values_list, set)
        for j in range(len(set)):
            if set[j][i] == '?':
                set[j][i] = attr_value

def train_and_predict(X,Y,test_vals):
    tree = decisionTree.trainTree(X, Y)
    # nn = neuralNet.trainNet(X,Y)
    neighbor = neighbors.trainNeighbors(X,Y)
    bayes = naiveBayes.trainBayes(X,Y)
    sv = svc.trainSVM(X, Y)

    treePredictions = predict.predictions(tree,test_vals)
    svPredictions = predict.predictions(sv, test_vals)
    neighborPredictions = predict.predictions(neighbor,test_vals)
    bayesPredictions = predict.predictions(bayes,test_vals)

    # neural net method
    treeTrainPred = predict.predictions(tree, X)
    neighborTrainPred = predict.predictions(neighbor, X)
    bayesTrainPred = predict.predictions(bayes, X)
    svTrainPred = predict.predictions(sv, X)

    neuralX = []
    for i in range(len(treeTrainPred)):
        #make list of 1800 rows of 4 values (prediction of eachk)
        neuralX.append([treeTrainPred[i], neighborTrainPred[i], bayesTrainPred[i], svTrainPred[i]])
    neuralX = np.array(neuralX)
    neuralX = neuralX.astype(np.float64)
    print(len(neuralX))
    nn = neuralNet.trainNet(neuralX,Y)

    neuralTestVals = []
    for i in range(len(treePredictions)):
        #make list of 1800 rows of 4 vbalues (prediction of eachk)
        neuralTestVals.append([treePredictions[i], neighborPredictions[i], bayesPredictions[i], svPredictions[i]])
    neuralTestVals = np.array(neuralTestVals)
    neuralTestVals = neuralTestVals.astype(np.float64)
    predictions = predict.predictions(nn, neuralTestVals)
    
    '''    
    rawPredictions = zip(treePredictions, svPredictions, bayesPredictions, neighborPredictions)
    predictions = []
    for p in rawPredictions:
        data = Counter(p)
        predictions.append(int(data.most_common(1)[0][0]))
    '''
    return predictions

def main():
    attribute_filename = sys.argv[1]
    training_filename = sys.argv[2]
    test_filename = sys.argv[3]   #parametrized name

    test_vals = []
    with open(test_filename,'r') as infile:
        for x in range (4000):
            line = infile.readline()
            tvs = line.split()
            tvs.pop()
            test_vals.append(tvs)

    attr_types = parse.read_attribute_file(attribute_filename)
    fill_in_missing_values(attr_types, test_vals)
    test_vals = np.array(test_vals)
    test_vals = test_vals.astype(np.float64)

    X,Y = parse.read_training_file(training_filename)
    print(len(X))
    print(len(Y))
    fill_in_missing_values(attr_types, X)
    X = np.array(X)
    X = X.astype(np.float64)



    predictions_array = []
    for i in range(1000):
        subsetX = []
        subsetY = []
        possible_indices = range(len(X))
        #pick 100 random training sets, predict based on them
        for j in range(len(X)/10):
            random_num = random.randrange(0,len(possible_indices),1)
            index = possible_indices[random_num]
            subsetX.append(X[index])
            subsetY.append(Y[index])
            possible_indices.remove(index)
        predictions_array.append(train_and_predict(subsetX,subsetY,test_vals))
    predictions = []
    for i in range (4000):
        predictions.append(find_most_common(i, ['0','1'], predictions_array))
    print(len(predictions_array))
    print(len(predictions_array[0]))

    with open('predictions.txt','w') as outfile:
        for p in predictions[:-1]:
            outfile.write(str(p) + '\n')
        outfile.write(str(predictions[-1]))


main()