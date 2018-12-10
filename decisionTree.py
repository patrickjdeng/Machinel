from sklearn import tree
def trainTree(X, Y):
    clf = tree.DecisionTreeClassifier(max_depth=50)
    return clf.fit(X, Y)