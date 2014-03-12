
import numpy as np
from scipy import sparse

import featurefunc as ff
import classifiers as cl
import util


## The following function does the feature extraction, learning, and prediction
def main():
    train_dir = "train"
    test_dir = "test"
    outputfile = "mypredictions.csv"  # feel free to change this or take it as an argument

    lr = cl.LogisticRegression()
    knn = cl.kNN()
    
    ds = ff.Dataset()

    print "training..."

    X, y, ids = ds.getDataset(train_dir)
    lr.fit(X,y)
    knn.fit(X,y)
    print "training complete. Now preparing for submit"

    X, y, ids = ds.getDataset(test_dir)
    
    predsLR = lr.predict(X)
    pbLR    = lr.classifier_().predict_proba(X)

    predskNN = knn.predict(X)
    pbkNN = knn.classifier_().predict_proba(X)

    finalpred = []
    for i in xrange(len(predsLR)):
        if np.max(pbkNN[i]) - np.max(pbLR[i]) > 0.4:
            choice = predskNN[i]
        else:
            choice = predsLR[i]
        finalpred.append(choice)
    
    print "writing predictions..."
    util.write_predictions(finalpred, ids, outputfile)
    print "done!"

if __name__ == "__main__":
    main()
    
