from sklearn.neighbors import KNeighborsClassifier

def trainNeighbors(X,Y):
    clf = KNeighborsClassifier()
    return clf.fit(X,Y)