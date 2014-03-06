
import numpy as np
from scipy import sparse

import featurefunc as ff
import classifiers as cl
import util


## The following function does the feature extraction, learning, and prediction
def main():
    train_dir = "mintrain"
    test_dir = "mintest"
    outputfile = "mypredictions-min2.csv"  # feel free to change this or take it as an argument

    lr = cl.LogisticRegression()
    
    ds = ff.Dataset()

    print "training..."

    X, y, ids = ds.getDataset(train_dir)
    lr.fit(X,y)
    print "training complete. Now preparing for submit"

    X, y, ids = ds.getDataset(test_dir)
    preds = lr.predict(X)
    
    print "writing predictions..."
    util.write_predictions(preds, ids, outputfile)
    print "done!"

if __name__ == "__main__":
    main()
    
