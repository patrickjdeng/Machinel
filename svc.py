from sklearn.svm import SVC
def trainSVM(X, Y):
    clf = SVC()
    return clf.fit(X, Y)