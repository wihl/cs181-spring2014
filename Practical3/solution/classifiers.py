import numpy as np

import util

def randomClassify(X = None,feat_dict = None):
    learned_W = np.random.random((len(feat_dict),len(util.malware_classes)))
    preds = np.argmax(X.dot(learned_W),axis=1)
    return preds

classifier_names = ['random']
classifiers = [
    randomClassify
]

def getClassiferNames():
    return classifier_names

def getClassifiers():
    return classifiers
