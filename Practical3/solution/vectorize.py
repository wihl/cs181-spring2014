'''
  vectorize
  Take one or more source XML files, extract features and write to a YAML file
'''

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
    outputfile = "mypredictions.csv"  # feel free to change this or take it as an argument
    
    # TODO put the names of the feature functions you've defined above in this list
    ffs = featurefunc.getFeatures()
    
    # extract features
    print "extracting training features..."
    X_train,global_feat_dict,t_train,train_ids = featurefunc.extract_feats(ffs, train_dir)
    print "done extracting training features"
    print
    
    # TODO train here, and learn your classification parameters
    print "learning..."
    learned_W = np.random.random((len(global_feat_dict),len(util.malware_classes)))
    print "done learning"
    print

    # write out vectorized result
    # see http://stackoverflow.com/questions/8955448/save-load-scipy-sparse-csr-matrix-in-portable-data-format
    print "serializing training features..."
    with open('data.npy', 'w') as outfile:
        np.save(outfile, X_train)

    with open('features.npy','w') as outfile:
        np.save(outfile, t_train)

    with open('learned_w.npy','w') as outfile:
        np.save(outfile, learned_W)

    with open('features.yaml', 'w') as outfile:
        outfile.write( yaml.dump(global_feat_dict, default_flow_style=True) )

    with open('ids.yaml', 'w') as outfile:
        outfile.write( yaml.dump(train_ids, default_flow_style=True) )
    print "done"



if __name__ == "__main__":
    main()
    
