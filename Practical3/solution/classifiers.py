import numpy as np
from sklearn import linear_model

import util

class Classifier(object):
    def name(self):
        return 'unknown'

    def save(self, file):
        pass

    def load(self, file):
        pass

    def fit(self, X, y, feat_dict):
        raise NotImplementedError("base class")

    def predict(self, X):
        raise NotImplementedError("base class")

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

    def fit(self, X, y, feat_dict):
        pass

    def predict(self, X):
        return np.ones(X.shape[0], np.int64) * 8 # hardcoded value for No virus


def getClassifiers():
    return [
            RandomClassifier,
            MostFrequent
           ]

