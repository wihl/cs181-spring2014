import numpy as np
from scipy import sparse
from scipy.sparse import linalg as splinalg

from sklearn.linear_model import Lasso



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
    mean_error.append(calcError(trainx, testx, trainy, Y, 0, f))

    alpha = 100.0
    lasso = Lasso(alpha = alpha, normalize=False, positive=True, precompute='auto', max_iter=5000)

    y_pred_lasso = lasso.fit(trainx, trainy).predict(testx)

    error = 0.0

    # test
    for i in range (len(y_pred_lasso)):
        error += abs(y_pred_lasso[i] - Y[i])
        #print i, y_pred_lasso[i], Y[i]
    
    print error / float(len(y_pred_lasso))



    # rerun at the last fold
    trainx = X[0:train_size]
    testx  = X[train_size:]
    trainy = Y[0:train_size]
    mean_error.append(calcError(trainx, testx, trainy, Y, train_size, f))

    lasso = Lasso(alpha = alpha, normalize=False, positive=True, precompute='auto', max_iter=1500)

    y_pred_lasso = lasso.fit(trainx, trainy).predict(testx)

    error = 0.0

    # test
    for i in range (len(y_pred_lasso)):
        error += abs(y_pred_lasso[i] - Y[train_size + i])
        #print i, y_pred_lasso[i], Y[train_size + i]
    
    print error / float(len(y_pred_lasso))


    return mean_error


