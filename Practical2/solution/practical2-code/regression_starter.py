## This file provides starter code for extracting features from the xml files and
## for doing some learning.
##
## The basic set-up: 
## ----------------
## main() will run code to extract features, learn, and make predictions.
## 
## extract_feats() is called by main(), and it will iterate through the 
## train/test xml files and extract each instance into a util.MovieData object.
## It will then use a series of "feature-functions" that you will write/modify
## in order to extract dictionaries of features from each util.MovieData object.
## Finally, it will produce an N x D sparse design matrix containing the union
## of the features contained in the dictionaries produced by your "feature-functions."
## This matrix can then be plugged into your learning algorithm.
##
## The learning and prediction parts of main() are largely left to you, though
## it does contain code for naive linear regression and prediction so you
## have a sense of where/what to modify.
##
## Feature-functions:
## --------------------
## "feature-functions" are functions that take a util.MovieData object representing
## a single movie, and return a dictionary mapping feature names to their respective
## numeric values. 
## For instance, a simple feature-function might map a movie object to the
## dictionary {'company-Fox Searchlight Pictures': 1}. This is a boolean feature
## indicating whether the production company of this move is Fox Searchlight Pictures,
## but of course real-valued features can also be defined. Because this feature-function
## will be run over MovieData objects for each movie instance, we will have the (different)
## feature values of this feature for each movie, and these values will make up 
## one of the columns in our final design matrix.
## Of course, multiple features can be defined within a single dictionary, and in
## the end all the dictionaries returned by feature functions will be unioned
## so we can collect all the feature values associated with a particular instance.
##
## Two example feature-functions, metadata_feats() and unigram_feats() are defined
## below. These extract metadata and unigram text features, respectively.
##
## What you need to do:
## --------------------
## 1. Write new feature-functions (or modify the example feature-functions) to
## extract useful features for this prediction task.
## 2. Implement an algorithm to learn from the design matrix produced, and to
## make predictions on unseen data. Naive code for these two steps is provided
## below, and marked by TODOs.


from collections import Counter
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
import numpy as np
from scipy import sparse
from scipy.sparse import linalg as splinalg
import matplotlib.pyplot as plt
import argparse
from dateutil import parser
import operator
import math
import csv

import util
import crossvalidate

def extract_feats(ffs, datafile="train.xml", global_feat_dict=None):
    """
    arguments:
      ffs are a list of feature-functions.
      datafile is an xml file (expected to be train.xml or testcases.xml).
      global_feat_dict is a dictionary mapping feature_names to column-numbers; it
      should only be provided when extracting features from test data, so that 
      the columns of the test matrix align correctly.

    returns: 
      a sparse design matrix, a dict mapping features to column-numbers,
      a vector of target values, and a list of movie ids in order of their
      rows in the design matrix
    """
    fds = [] # list of feature dicts
    targets = []
    ids = [] 
    begin_tag = "<instance" # for finding instances in the xml file
    end_tag = "</instance>"
    in_instance = False
    curr_inst = [] # holds lines of file associated with current instance

    # start iterating thru file
    with open(datafile) as f:
        # get rid of first two lines
        _ = f.readline()
        _ = f.readline()
        row = 0
        for line in f:
            if begin_tag in line:
                if in_instance: 
                    assert False  # cannot have nested instances
                else:
                    curr_inst = [line]
                    in_instance = True
            elif end_tag in line:
                # we've read in an entire instance; we can extract features
                curr_inst.append(line)
                # concatenate the lines we've read and parse as an xml element
                movie_data = util.MovieData(ET.fromstring("".join(curr_inst)))
                rowfd = {}
                # union the output of all the feature functions over this instance
                [rowfd.update(ff(movie_data)) for ff in ffs]
                # add the final dictionary for this instance to our list
                fds.append(rowfd)
                # add target val
                targets.append(movie_data.target)
                # keep track of the movie id's for later
                ids.append(movie_data.id)
                #print row, movie_data.id
                row += 1
                # reset
                curr_inst = []
                in_instance = False
            elif in_instance:
                curr_inst.append(line)

    X,feat_dict = make_design_mat(fds,global_feat_dict)
    return X, feat_dict, np.array(targets), ids


def make_design_mat(fds, global_feat_dict=None):
    """
    arguments:
      fds is a list of feature dicts (one for each row).
      global_feat_dict is a dictionary mapping feature_names to column-numbers; it
      should only be provided when extracting features from test data, so that 
      the columns of the test matrix align correctly.
       
    returns: 
        a sparse NxD design matrix, where N == len(fds) and D is the number of
        the union of features defined in any of the fds 
    """
    if global_feat_dict is None:
        all_feats = set()
        [all_feats.update(fd.keys()) for fd in fds]
        feat_dict = dict([(feat, i) for i, feat in enumerate(sorted(all_feats))])
    else:
        feat_dict = global_feat_dict
        
    cols = []
    rows = []
    data = []        
    for i in xrange(len(fds)):
        temp_cols = []
        temp_data = []
        for feat,val in fds[i].iteritems():
            try:
                # update temp_cols iff update temp_data
                temp_cols.append(feat_dict[feat])
                temp_data.append(val)
            except KeyError as ex:
                if global_feat_dict is not None:
                    pass  # new feature in test data; nbd
                else:
                    raise ex

        # all fd's features in the same row
        k = len(temp_cols)
        cols.extend(temp_cols)
        data.extend(temp_data)
        rows.extend([i]*k)

    assert len(cols) == len(rows) and len(rows) == len(data)
   

    X = sparse.csr_matrix((np.array(data),
                   (np.array(rows), np.array(cols))),
                   shape=(len(fds), len(feat_dict)))
    return X, feat_dict
    

## Here are two example feature-functions. They each take in a util.MovieData
## object, and return a dictionary mapping feature-names to numeric values.
def metadata_feats(md):
    """
    arguments:
      md is a util.MovieData object
    returns:
      a dictionary containing a mapping from a subset of the possible metadata features
      to their values on this util.MovieData object
    """
    d = {}
    for k,v in md.__dict__.iteritems():
        if k == 'production_budget':
            d[k] = v
        if k == 'number_of_screens':
            d[k] = pow(v,4)
    return d

def unigram_feats(md):
    """
    arguments:
      md is a util.MovieData object
    returns:
      a dictionary containing a mapping from unigram features from the reviews
      to their values on this util.MovieData object
    """
    c = Counter()
    for rev in util.MovieData.reviewers:
        if hasattr(md,rev):
            # count occurrences of asciified, lowercase, non-numeric unigrams 
            # after removing punctuation
            c.update([token for token in 
                        util.punct_patt.sub("",
                         util.asciify(md.__dict__[rev].strip().lower())).split()
                          if util.non_numeric(token)])
    return c

##Created squared terms for numeric fields
def squared_terms(md):
    d = {}
    # for k,v in md.__dict__.iteritems():
    #     d.update([(k+"-"+val,1) for val in v])
    # return d
    #d = {}
    for k,v in md.__dict__.iteritems():
        if k == 'number_of_screens':
            if isinstance(v, float):
                d[k] = v**4.3
            elif isinstance(v, bool):
                d[k] = float(v**2)
            else:
                d[k]=0
        # if k == 'number_of_screens':
        #     if isinstance(v, float):
        #         d['num_scr_mod'] = v**2
        #     elif isinstance(v, bool):
        #         d['num_scr_mod'] = float(v**2)
        #     else:
        #         d['num_scr_mod']=0

    return d

def prod_company(md):
    d = {}
    # for k,v in md.__dict__.iteritems():
    #     d.update([(k+"-"+val,1) for val in v])
    # return d
    #d = {}
    for k,v in md.__dict__.iteritems():
        c = Counter()
        if k != 'company':
            d['company'] = 0
        elif k=='company':
            d['company'+str(v)]=1
        else:
            pass
    return d


##captures review data
def review_terms(md):
    d = {}
    rev_count = 0
    for rev in util.MovieData.reviewers:
        if hasattr(md,rev):
            d['Review-'+rev] = 1
    return d

# global_list=[]
# global_screens=[]

##captures if numbers are above a particular treshold (e.g. non linear 'jump' in values)
def threshold_terms(md):
    # global global_list
    # global global_screens
    d = {}
    for k,v in md.__dict__.iteritems():
        if k in ['production_budget']:
            if isinstance(v, float):
                d[k] = float(v>8000000)
                # global_list.append(v)
            else:
                d[k]=0
        if k in ['number_of_screens']:
            if isinstance(v, float):
                d[k] = float(v>300)
                # global_screens.append(v)
            else:
                d[k]=0
    return d

# print global_list
# print global_screens

# plt.hist(np.array(global_list))
# plt.show()

# plt.hist(np.array(global_screens))
# plt.show()


##captures review data
def review_score(md):
    d = {"review_pos":0, "review_neg":0}
    rev_count = 0
    for rev in util.MovieData.reviewers:
        if hasattr(md,rev):
            for review in md.__dict__[rev].strip().lower().split():
                #print review
                for word_pos in ["good", "great", "excellent", "seeing", "superb", "must-see", "fantastic", "credible", "best"]:
                    if review.find(word_pos)>-1:
                        d["review_pos"]=1
 
                for word_neg in ["bad", "slow", "boring", "horrible", "waste", "poor", "lazy","worst"]:
                    if review.find(word_neg)>-1:
                        d["review_neg"]=1
    return d


## The following function does the feature extraction, learning, and prediction
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cross_validate',help='Run cross-validation (instead of of output)', action='store_true')
    parser.add_argument('-t', '--train_file', nargs=1, help='Training file')
    args = parser.parse_args()

    if args.train_file:
        trainfile = args.train_file[0]
    else:
        trainfile = "train.xml"
    testfile = "testcases.xml"
    outputfile = "mypredictions2.csv"  # feel free to change this or take it as an argument
    
    # put the names of the feature functions you've defined above in this list
    ffs = [metadata_feats, squared_terms]#, prod_company, review_score] #, review_terms] #, unigram_feats, threshold_terms]

    
    # extract features
    print "extracting training features..."
    X_train,global_feat_dict,y_train,train_ids = extract_feats(ffs, trainfile)
    global_feat_dict_sorted = sorted(global_feat_dict.iteritems(), key=operator.itemgetter(1))
    print global_feat_dict_sorted
    #print X_train.sum(axis=0)
    #print "1:",X_train[0]
    #print "2:",X_train[1]
    #print "3:",X_train[2]

    print "done extracting training features"
    print
    

    if args.cross_validate:
        print "running cross-validation tests..."
        score = crossvalidate.getScore(X_train,y_train, splinalg.lsqr)
        print "MAE cross validation score:",score
        print "done cross-validation"
    else:

        # write out predictions on test data

        # train here, and return regression parameters
        print "learning..."
        learned_w = splinalg.lsqr(X_train,y_train)[0]
        print '\n'.join(['%i: %8.8f %s' % 
                         (n, learned_w[n], global_feat_dict_sorted[n][0]) for n in xrange(len(learned_w))])
        '''
        preds = np.absolute(X_train.dot(learned_w))
        myfile = open('bb.txt','wb')
        wr = csv.writer(myfile, dialect='excel')
        for i in range(len(preds)):
            wr.writerow([i,X_train[i,0],X_train[i,1], y_train[i], preds[i]])
        '''
        print "done learning"
        print

        # get rid of training data and load test data
        del X_train
        del y_train
        del train_ids
        print "extracting test features..."
        X_test,_,y_ignore,test_ids = extract_feats(ffs, testfile, global_feat_dict=global_feat_dict)
        print "done extracting test features"
        print
        
        # make predictions on text data and write them out
        print "making predictions..."
        preds = np.absolute(X_test.dot(learned_w))
        # blockbuster correction factor
        for i in range(len(preds)):
            if X_test[i,1] > 50000000.0:
                preds[i] *= 0.85
        print "done making predictions"
        print
    
        print "writing predictions..."
        util.write_predictions(preds, test_ids, outputfile)
        print "done!"

if __name__ == "__main__":
    main()
    
