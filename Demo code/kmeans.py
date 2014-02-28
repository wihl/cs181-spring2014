## Sample implementations of kmeans. 
## The first implementation is fairly vectorized, but loops once.
## The second one doesn't loop at all, and is consequently a good bit faster.
##
## On my laptop, I get the following output when running demo():
## In [8]: demo()
## data has dimensions (10000, 3072)
## 47 iterations
## one_loop_time 191.426865816
## 47 iterations
## no_loop_time 18.5532119274
## assignments equal: True
## mu difference 0.0
##
## NB: The demo assumes you have unzipped the python version of CIFAR-10 and 
## have it in the same directory as the code

import numpy as np
import matplotlib.pyplot as plt

def kmeans_one_loop(X,K):
    """
    arguments:
        X is an NxD matrix of feature vectors;
        K is the desired number of classes
    returns:
        an N-length vector of cluster-assignments and a KxD matrix of cluster-centers
    """
    N,D = X.shape
    prev_assn = np.zeros(N, dtype=np.int) # initialize with garbage
    
    # uniformly pick random responsibilities; form NxK matrix
    R = np.random.multinomial(1, np.ones(K)/float(K), size=N)
    new_assn = np.argmax(R,axis=1)
    it = 0

    while not np.array_equal(prev_assn, new_assn):
        prev_assn = new_assn
        # calculate class counts
        N_k = R.sum(0)
        # form K x D matrix of initial mus by summing selected rows and dividing rows by N_k
        Mus = R.T.dot(X)/N_k[:,None]
        
        # calculate distances of each pt from each mean
        Dists = np.zeros((N,K))
        for k in xrange(K):
            Dists[:,k] = ((X-Mus[k])**2).sum(1)
        
        R[:,:] = 0 # clear responsibilities
        # set new responsibilities with fancy indexing
        R[np.arange(N), np.argmin(Dists,1)] = 1
        new_assn = np.argmax(R,axis=1)
        it += 1
    
    print it, "iterations"
    return new_assn, Mus

def kmeans_no_loop(X,K):
    """
    arguments:
        X is an NxD matrix of feature vectors;
        K is the desired number of classes
    returns:
        an N-length vector of cluster-assignments and a KxD matrix of cluster-centers
    """
    N,D = X.shape
    prev_assn = np.zeros(N, dtype=np.int) # initialize with garbage
    
    # uniformly pick random responsibilities; form NxK matrix
    R = np.random.multinomial(1, np.ones(K)/float(K), size=N)
    new_assn = np.argmax(R,axis=1)
    it = 0

    while not np.array_equal(prev_assn, new_assn):
        prev_assn = new_assn
        # calculate class counts
        N_k = R.sum(0)
        # form K x D matrix of initial mus by summing selected rows and dividing rows by N_k
        Mus = R.T.dot(X)/N_k[:,None]
        
        # calculate NxK matrix of distances of each pt from each mean
        # (using that (x-m)**2 = x**2 + m**2 - 2xm)
        Dists = (X**2).sum(1)[:,None] + (Mus**2).sum(1) - 2*X.dot(Mus.T)
        
        R[:,:] = 0 # clear responsibilities
        # set new responsibilities with fancy indexing
        R[np.arange(N), np.argmin(Dists,1)] = 1
        new_assn = np.argmax(R,axis=1)
        it += 1
    
    print it, "iterations"
    return new_assn, Mus

def demo():
    """
    runs and times two kmeans implementations on a subset of the CIFAR-10 data, 
    checks that they're equivalent, and displays some pictures from 3 clusters
    """
    import cPickle
    import time
    with open("cifar-10-batches-py/data_batch_1") as f:
        d = cPickle.load(f)
    X = d['data']
    X = X/255.0
    print "data has dimensions", X.shape
    np.random.seed(2)
    start = time.time()
    assn_one, Mus_one = kmeans_one_loop(X,10)
    end = time.time()
    print "one_loop_time", end-start
    np.random.seed(2)
    start = time.time()
    assn_none, Mus_none = kmeans_no_loop(X,10)
    end = time.time()
    print "no_loop_time", end-start
    # check if we got the same answers
    print "assignments equal:", np.array_equal(assn_one, assn_none)
    print "mu difference", np.sum(np.abs(Mus_one-Mus_none))
    # just pick first three clusters to show
    for i in xrange(3):
        clust = d['data'][assn_none==i]
        np.random.shuffle(clust)
        assert len(clust) > 4
        for j in xrange(5):
            ax = plt.subplot(3,5,i*5+j+1)
            ax.imshow(clust[j].reshape((3,1024)).T.reshape((32,32,3)))
    plt.show()
