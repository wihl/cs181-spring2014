
import os
from collections import Counter
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
import numpy as np
from scipy import sparse
import yaml

import featurefunc
import util


## The following function does the feature extraction, learning, and prediction
def main():
    train_dir = "train"
    test_dir = "test"
    outputfile = "mypredictions-vec.csv"  # feel free to change this or take it as an argument

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
    
    # get rid of training data and load test data
    del X_train
    del t_train
    del train_ids
    print "extracting test features..."
    X_test,_,t_ignore,test_ids = featurefunc.extract_feats(ffs, test_dir,
                                                           global_feat_dict=global_feat_dict)
    print "done extracting test features"
    print
    
    # TODO make predictions on text data and write them out
    print "making predictions..."
    preds = np.argmax(X_test.dot(learned_W),axis=1)
    print "done making predictions"
    print
    
    print "writing predictions..."
    util.write_predictions(preds, test_ids, outputfile)
    print "done!"

if __name__ == "__main__":
    main()
    
