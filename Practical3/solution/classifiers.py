import numpy as np
from sklearn import linear_model
from sklearn import svm
from sklearn import cross_validation

import util

class Classifier(object):
    def name(self):
        return 'unknown'

    def save(self, file):
        pass

    def load(self, file):
        pass

    def fit(self, X, y, feat_dict):
        pass

    def score(self, X, y):
        total = len(y)
        c = 0
        for i, j in zip(X,y):
            if i == j: c += 1
        return float(c) / float(total)

    def predict(self, X):
        return np.ones(X.shape[0], np.int64) * 8 # hardcoded value for No virus


class RandomClassifier(Classifier):
    def __init__(self):
        self.learned_W = None

    def name(self):
        return 'random'

    def fit(self, X, y, feat_dict):
        self.learned_W = np.random.random((len(feat_dict),len(util.malware_classes)))
        return self.learned_W

    def predict(self, X):
        return np.argmax(X.dot(self.learned_W),axis=1)

class MostFrequent(Classifier):
    def name(self):
        return 'baseline'

class LogisticRegression(Classifier):
    def __init__(self):
        self.logreg = linear_model.LogisticRegression(C=1e5)

    def name(self):
        return 'LogisticRegression'

    def fit(self, X, y, feat_dict):
        self.logreg.fit(X, y)

    def predict(self, X):
        return self.logreg.predict(X)

class SVM(Classifier):
    def __init__(self):
        self.svm = svm.SVC(kernel='linear', C=1.0)

    def get_params(self, deep=False, *args):
        return self.svm.get_params(*args)

    def name(self):
        return 'SVM'

    def fit(self, X, y, feat_dict):
        self.svm.fit(X, y)

    def predict(self, X):
        return self.svm.predict(X)

def getClassifiers():
    return [
            SVM, 
            LogisticRegression
           ]

