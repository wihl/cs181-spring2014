'''
dump.py - utility functions to dump different sparse matrices
'''
import numpy as np
from scipy import sparse
from scipy import io
import yaml
import csv

import featurefunc
import util


def dumpFeatureVec(filename, csrmat, keys, rows):
    '''
    dump n beginning rows from the feature matrix. If rows == 0, dump everything
    '''
    sort_keys = sorted(keys, key=keys.get)
    
    myfile = open(filename, 'wb')
    wr = csv.writer(myfile, dialect = 'excel')
    cx = sparse.coo_matrix(csrmat)
   

    print "dumping to ", filename
    wr.writerow(sort_keys)
    cx = sparse.coo_matrix(csrmat)

    xavg = {}
    xtot = {}
    for i in cx.row:
        for j in cx.col:
            for v in cx.data:
                xtot[j] = v.__repr__
        print xtot
        xtot = {}

                #for i,j,v in zip(cx.row, cx.col, cx.data):
                #print "(%d, %d), %s" % (i,j,v)
        


def main():
    print "dumping ..."
        # TODO put the names of the feature functions you've defined above in this list
    ffs = featurefunc.getFeatures()
    
    # load vectorized data
    X_train = np.load('data.npy')

    with open('features.yaml', 'r') as f:
        global_feat_dict = yaml.load(f)

    t_train = np.load('features.npy')

    with open('ids.yaml','r') as f:
        train_ids = yaml.load(f)

    learned_W = np.load('learned_w.npy')

    dumpFeatureVec('x_train.txt',X_train,global_feat_dict, 10)

if __name__ == "__main__":
    main()
