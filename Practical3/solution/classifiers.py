import numpy as np

import util

def randomClassify(X = None,feat_dict = None):
    learned_W = np.random.random((len(feat_dict),len(util.malware_classes)))
    preds = np.argmax(X.dot(learned_W),axis=1)
    return preds

def expectNothing(X, feat_dict):
    preds = np.ones(X.shape[0], np.int64) * 8 # hardcoded value for No virus
    return preds

classifier_names = [
    'random', 
    'baseline'
]

classifiers = [
    randomClassify,
    expectNothing
]

def getClassiferNames():
    return classifier_names

def getClassifiers():
    return classifiers
