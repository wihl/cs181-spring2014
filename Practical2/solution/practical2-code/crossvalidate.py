import numpy as np
from scipy import sparse
from scipy.sparse import linalg as splinalg

NUM_FOLDS = 10

def calcError(trainx, testx, trainy, Y, start, f):
    # train
    w = f(trainx,trainy)[0]
    pred = testx.dot(w)
    error = 0.0

    # test
    for i in range (len(pred)):
        error += abs(pred[i] - Y[start + i])
    
    return error / float(len(pred))


def getScore(X,Y,f):
    '''
    X is input values
    Y is target values
    f is the function to generate predictions
    '''
    m = len(Y)
    print X.shape, m
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
    mean_error.append(calcError(trainx, testx, trainy, Y, 0, f))

    # rerun at the last fold
    trainx = X[0:train_size]
    testx  = X[train_size:]
    trainy = Y[0:train_size]
    mean_error.append(calcError(trainx, testx, trainy, Y, train_size, f))

    return mean_error


