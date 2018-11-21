from sklearn.neural_network import MLPClassifier

def trainNet(X,Y):
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
    return clf.fit(X,Y)