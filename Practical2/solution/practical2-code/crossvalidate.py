import numpy as np
from scipy import sparse
from scipy.sparse import csr_matrix
from scipy.sparse import linalg as splinalg

from sklearn import linear_model
import math
import stageWise as sw
import csv


NUM_FOLDS = 10

def calcError(trainx, testx, trainy, Y, start, f, meanY):
    # train
    w = f(trainx,trainy)[0]
    pred = testx.dot(w)

    error = 0.0

    #myfile = open('analysis.csv', 'wb')
    #wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)


    # test
    for i in range (len(pred)):
        if testx[i,1] > 100000000.0:
            pred[i] *= 1.75;
        error += abs(pred[i] - Y[start + i])
        #wr.writerow([i, testx[i,0], testx[i,1], pred[i], Y[start+i]])
        err_percent = int(math.floor(error / Y[start + i])) -1

    return error / float(len(pred))


def getScore(X,Y,f):
    '''
    X is input values
    Y is target values
    f is the function to generate predictions
    '''
    m = len(Y)
    meanY = np.mean(Y)
    print "meanY:",meanY
    print "X shape:", X.shape
    if m < NUM_FOLDS:
        num_folds = 2
        fold_size = m / 2
    else:
        num_folds = NUM_FOLDS
        fold_size = m / num_folds

    train_size = m - fold_size
    mean_error = []

    # start at the first fold
    trainx = X[fold_size:]
    testx  = X[0:fold_size]
    trainy = Y[fold_size:]
    mean_error.append(calcError(trainx, testx, trainy, Y, 0, f, meanY))

    '''
    clf = linear_model.BayesianRidge()

    y_pred_lasso = clf.fit(trainx.toarray(), trainy.toarray()).predict(testx)

    error = 0.0

    # test
    for i in range (len(y_pred_lasso)):
        error += abs(y_pred_lasso[i] - Y[i])
        #print i, y_pred_lasso[i], Y[i]
    
    print error / float(len(y_pred_lasso))
    '''
    # rerun at the last fold
    trainx = X[0:train_size]
    testx  = X[train_size:]
    trainy = Y[0:train_size]
    mean_error.append(calcError(trainx, testx, trainy, Y, train_size, f, meanY))

    '''
    pred = sw.stageWise(trainx, trainy)

    error = 0.0

    # test
    for i in range (len(pred)):
        error += abs(pred[i] - Y[train_size + i])
        print i, pred, Y[train_size + i]
    
    print error / float(len(pred))
    '''

    return mean_error


