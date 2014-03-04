'''
dump.py - utility functions to dump different sparse matrices
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
import yaml
import csv
from random import randrange
import datetime

import featurefunc
import classifiers as cl
import util



def extract_feats(ffs, fds, direc, datafile):
    rowfd = {}
    # parse file as an xml document
    tree = ET.parse(os.path.join(direc,datafile))
    # accumulate features
    [rowfd.update(ff(tree)) for ff in ffs]
    fds.append(rowfd)        

    return fds

def train_pred(fds, test_fds):
    X,feat_dict = featurefunc.make_design_mat(fds,None)
    learned_W = np.random.random((len(feat_dict),len(util.malware_classes)))
    preds = np.argmax(X.dot(learned_W),axis=1)
    return preds

def calcAccuracy(preds, classes, ids):
    # calculate accuracies
    item_count = 0
    accuracyByClass = []

    for i in range(len(util.malware_classes)):
       accuracyByClass.append({'true_pos':0, 'cond_pos':0, 'outcome_pos': 0, 'false_pos':0, 'false_neg':0})

    for i, fileid in enumerate(ids):
        if preds[i] == np.int64(classes[fileid]):
            item_count += 1
            accuracyByClass[classes[fileid]]['true_pos'] += 1
        else:
            accuracyByClass[int(preds[i])]['false_pos'] += 1
            accuracyByClass[classes[fileid]]['false_neg'] += 1
        accuracyByClass[classes[fileid]]['cond_pos'] += 1
        accuracyByClass[int(preds[i])]['outcome_pos'] += 1
    
    accuracy = float(item_count) / float(len(ids)) 
    j = 0
    for i in accuracyByClass:
        tp = float(i['true_pos'])
        fp = float(i['false_pos'])
        ocp = float(i['outcome_pos'])
        try:
            precision = tp/(tp+fp)
        except:
            precision = 0.0
        cp = float(i['cond_pos'])
        try:
            sensitivity = tp/cp
        except:
            sensitivity = 0.0
        #print i
        #print j, "Precision:", precision, "Sensitivity:",sensitivity
        j += 1

    return accuracy


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

        extract_feats(ffs, fds, direc, datafile)

    assert num_folds < len(ids)
    accuracy = []
    for i in range(num_folds):
        test_ids = []
        test_fds = []

        # slice the data in folds
        sliceSize = len(ids) / num_folds
        for j in range(sliceSize):
            # find a random entry
            random_index = randrange(0,len(ids))
            test_ids.append(ids[random_index])
            del ids[random_index]
            test_fds.append(fds[random_index])
            del fds[random_index]

        # train and predict
        X,feat_dict = featurefunc.make_design_mat(fds,None)

        preds = clf(X,feat_dict)
        a = calcAccuracy(preds, classes, ids)
        accuracy.append(a)

    print "feat dict:", feat_dict
    return accuracy

def accuracyMetrics(accuracy):
    avgAcc = 0.0
    bestAcc = 0.0
    worstAcc = float("inf")

    for i in range(len(accuracy)):
        avgAcc += accuracy[i]
        if accuracy[i] < worstAcc: worstAcc = accuracy[i]
        if accuracy[i] > bestAcc:  bestAcc  = accuracy[i]
        
    avgAcc /= len(accuracy)
    return avgAcc, bestAcc, worstAcc

def main():
    num_folds = 10

    classifier_names = cl.getClassiferNames()
    classifiers      = cl.getClassifiers()
    with open('error_log.txt', 'a') as errfile:
        wr = csv.writer(errfile, dialect = 'excel')
    
        for name, clf in zip(classifier_names, classifiers):
            accuracy = validate(num_folds, clf)
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            avgAcc, bestAcc, worstAcc = accuracyMetrics(accuracy)

            wr.writerow([timestamp, num_folds, name, avgAcc, bestAcc, worstAcc])

            print name, "score is ", avgAcc * 100.0, "%"
        
if __name__ == "__main__":
    main()
