import numpy as np
from scipy import sparse
from scipy.sparse import linalg as splinalg

NUM_FOLDS = 10

def getScore(X,Y,f):
    '''
    X is input values
    Y is target values
    f is the function to generate predictions
    '''
    m = len(Y)
    if m < NUM_FOLDS:
        num_folds = 2
        fold_size = m / 2
    else:
        num_folds = NUM_FOLDS
        fold_size = m / num_folds

    train_size = m - fold_size

    trainx = X[0:train_size]
    testx  = X[train_size:]
    trainy = Y[0:train_size]

    # train
    w = f(trainx,trainy)[0]
    pred = trainx.dot(w)
    error = 0.0
    
    # test
    for i in range (fold_size):
        error += abs(pred[i] - Y[train_size + i])
        print i, pred[i], Y[train_size + i]

    mean_error = error / float(fold_size)
    return mean_error


