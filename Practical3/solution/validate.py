'''
validate.py - testing harness
'''
import os
from collections import Counter
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET

import numpy as np
from scipy import sparse
from scipy import io
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score
import yaml
import csv
from random import randrange
import datetime

import featurefunc
import classifiers as cl
import util

def validate(num_folds, clf, direc = 'mintrain'):
    assert clf != None
    ffs = featurefunc.getFeatures()
    fds = [] # list of feature dicts
    classes = {}
    ids = []

    for datafile in os.listdir(direc):
        # extract id and true class (if available) from filename
        id_str,clazz = datafile.split('.')[:2]
        ids.append(id_str)
        # add target class if this is training data
        if clazz != "X":
            classes[id_str] = util.malware_classes.index(clazz)

        featurefunc.extract_feats_by_file(ffs, fds, direc, datafile)

    assert num_folds < len(ids)
    accuracy = []
    X,feat_dict = featurefunc.make_design_mat(fds,None)

    y = [classes[item] for item in ids]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3)
    clf.fit(X_train,y_train,feat_dict)
    preds = clf.predict(X_test)
    accuracy = clf.score(preds,y_test)

    # TODO: temp fix so mean() and std() will work below.
    return np.array([accuracy, .1 , .2])

def main():
    num_folds = 10

    with open('error_log.txt', 'a') as errfile:
        wr = csv.writer(errfile, dialect = 'excel')
    
        for clf in cl.getClassifiers():
            c = clf()
            accuracy = validate(num_folds, c)
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            avgAcc = accuracy.mean()
            #accStd = accuracy.std()

            wr.writerow([timestamp, num_folds, c.name(), avgAcc, np.argmax(accuracy), np.argmin(accuracy)])

            #print c.name(),"score is ", avgAcc * 100.0, "%"
    
            print("%s accuracy: %0.2f (+/- %0.2f)" % (c.name(), accuracy.mean(), accuracy.std() * 2))
        
if __name__ == "__main__":
    main()
