from sklearn.naive_bayes import GaussianNB

def trainBayes(X,Y):
    clf = GaussianNB()
    return clf.fit(X,Y)