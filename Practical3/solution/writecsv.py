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
from sklearn.cross_validation import train_test_split
import csv
import datetime
import argparse
import operator

import featurefunc as ff
import classifiers as cl
import util

def generate(wr, clf, direc, ds, ds2 = None):
    assert clf != None
    X, y, ids = ds.getDataset(direc)
    if ds2 is not None:
        X2, y2, ids2 = ds2.getDataset(direc)
    featureDict = ds.getFeatureDict()
    print "X len=",X.todense().shape, "Y len=", len(y)
    X_dense = np.asarray(X.todense())
    wr.writerow([x[0] for x in 
                 sorted(featureDict.iteritems(), key=operator.itemgetter(1))]
                 + ['y'])
    for i in range(len(y)):
        wr.writerow([x for x in X_dense[i]] + [y[i]])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--full',help='full or minimal training run (default minimal)',action='store_true')
    parser.add_argument('-t', '--thread',help='use thread metrics (default process)', action='store_true')
    parser.add_argument('classifier',nargs='?', help='Which classifer to use', 
                        default='LogisticRegression-1000')
    args = parser.parse_args()

    if args.full:
        direc = 'train'
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        print("%s Start of full run" % (timestamp))

    else:
        direc = 'mintrain'

    if args.classifier:
        classifier = args.classifier
    else:
        classifier = None

    if args.thread:
        ds = ff.Dataset(ff.MetricType().thread)
    else:
        ds = ff.Dataset()

    ds2 = None

    with open('output.csv', 'w') as outfile:
        wr = csv.writer(outfile)
    
        c = cl.LogisticRegression()

        generate(wr, c, direc, ds, ds2)

        #wr.writerow([timestamp, num_iter, direc, c.name(), avgAcc, stdAcc, 
        #             np.max(accuracy), np.min(accuracy)])

        #        wcsv.writerow(["Key","mean","std"])

if __name__ == "__main__":
    main()
