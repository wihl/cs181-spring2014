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

import featurefunc as ff
import classifiers as cl
import util

def validate(num_iterations, clf, direc = 'mintrain'):
    assert clf != None
    accuracy = []
    ds = ff.Dataset()
    X, y, ids = ds.getDataset(direc)
    
    for size in [0.3, 0.2, 0.1]: # try 3 fold, 5 fold and 10 fold
        for i in range(num_iterations):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=size)
            clf.fit(X_train,y_train)
            preds = clf.predict(X_test)
            accuracy.append(clf.score(preds,y_test))

    return np.array(accuracy)

def main():
    num_folds = 5

    with open('error_log.txt', 'a') as errfile:
        wr = csv.writer(errfile, dialect = 'excel')
    
        for clf in cl.getClassifiers():
            c = clf()
            accuracy = validate(num_folds, c)
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            avgAcc = accuracy.mean()
            stdAcc = accuracy.std() * 2

            wr.writerow([timestamp, num_folds, c.name(), avgAcc, stdAcc, 
                         np.max(accuracy), np.min(accuracy)])

            #print c.name(),"score is ", avgAcc * 100.0, "%"
    
            print("%s accuracy: %0.2f (+/- %0.2f)" % (c.name(), avgAcc, stdAcc))
        
if __name__ == "__main__":
    main()
