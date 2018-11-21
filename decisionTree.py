from sklearn import tree

def trainTree(X,Y):
    clf = tree.DecisionTreeClassifier()
    return clf.fit(X, Y)