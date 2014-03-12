import numpy as np
from scipy import stats
from sklearn import linear_model
from sklearn import svm
from sklearn import cross_validation
from sklearn.neighbors import NearestNeighbors

import featurefunc as ff
import util

class Classifier(object):
    def name(self):
        return 'unknown'

    def save(self, file):
        pass

    def load(self, file):
        pass

    def fit(self, X, y):
        pass

    def decision_function(self, X):
        return None

    def score(self, X, y):
        total = len(y)
        c = 0
        for i, j in zip(X,y):
            if i == j: c += 1
        return float(c) / float(total)
        '''
        # calculate accuracies
        item_count = 0
        accuracyByClass = []

        for i in range(len(util.malware_classes)):
            accuracyByClass.append({'true_pos':0, 'cond_pos':0, 'outcome_pos': 0, 'false_pos':0, 'false_neg':0})

        for i, fileid in enumerate(ids):
            if preds[i] == np.int64(classes[fileid]):
                item_count += 1
                accuracyByClass[classes[fileid]]['true_pos'] += 1
            else:
                accuracyByClass[int(preds[i])]['false_pos'] += 1
                accuracyByClass[classes[fileid]]['false_neg'] += 1
            accuracyByClass[classes[fileid]]['cond_pos'] += 1
            accuracyByClass[int(preds[i])]['outcome_pos'] += 1
    
        accuracy = float(item_count) / float(len(ids)) 
        j = 0
        for i in accuracyByClass:
            tp = float(i['true_pos'])
            fp = float(i['false_pos'])
            ocp = float(i['outcome_pos'])
            try:
                precision = tp/(tp+fp)
            except:
                precision = 0.0
            cp = float(i['cond_pos'])
            try:
                sensitivity = tp/cp
            except:
                sensitivity = 0.0
        #print i
        #print j, "Precision:", precision, "Sensitivity:",sensitivity
            j += 1

        return accuracy
        '''

    def predict(self, X):
        return np.ones(X.shape[0], np.int64) * 8 # hardcoded value for No virus

    def weights(self):
        return []

class RandomClassifier(Classifier):
    def __init__(self):
        self.dist = [0.0369, 0.0162, 0.012, 0.0103, 0.0133, 0.0126, 0.0172, 0.0133, 0.5214, 0.0068, 0.1756, 0.0104, 0.1218, 0.0191, 0.0130] 
        self.xk = np.arange(len(self.dist))
        self.custm = stats.rv_discrete(name="custm", values=(self.xk,self.dist))
        #self.learned_W = None

    def name(self):
        return 'random'

    def fit(self, X, y):
        self.learned_W = np.random.random((X.shape[1],len(util.malware_classes)))
        return self.learned_W

    def predictOne(self):
        return self.custm.rvs()

    def predict(self, X):
        y = [self.custm.rvs() for x in xrange(X.shape[0])]
        return y #np.argmax(X.dot(self.learned_W),axis=1)

class MostFrequent(Classifier):
    def name(self):
        return 'baseline'

class LogisticRegression(Classifier):
    def __init__(self):
        self.C = 1000
        self.logreg = linear_model.LogisticRegression(C=self.C)

    def name(self):
        return 'LogisticRegression-'+str(self.C)

    def fit(self, X, y):
        self.logreg.fit(X, y)

    def predict(self, X):
        return self.logreg.predict(X)

    def weights(self):
        return self.logreg.coef_

    def decision_function(self, X):
        return self.logreg.decision_function(X)

    def classifier_(self):
        return self.logreg

class SVM(Classifier):
    def __init__(self):
        self.svm = svm.SVC(kernel='linear', max_iter=100000, C=0.03)

    def get_params(self, deep=False, *args):
        return self.svm.get_params(*args)

    def name(self):
        return 'SVM'

    def fit(self, X, y):
        self.svm.fit(X, y)

    def predict(self, X):
        return self.svm.predict(X)

    def weights(self):
        return self.svm.coef_


class kNN(Classifier):
    def __init__(self):
        self.knn = NearestNeighbors(n_neighbors=len(util.malware_classes), algorithm='brute')

    def name(self):
        return 'kNN'

    def fit(self, X, y):
        self.knn.fit(X, y)

    def predict(self, X):
        dist, ind = self.knn.kneighbors(X)
        print "dist:", dist
        print "ind:", ind
        return np.ones(X.shape[0], np.int64) * 8 # hardcoded value for No virus


class Combined(Classifier):
    def __init__(self):
        self.lrProcess = LogisticRegression()
        self.lrThread  = LogisticRegression()

    def name(self):
        return 'Combined'

    def fit(self, X, y, X2=None, y2=None):
        self.lrProcess.fit(X, y)
        if X2 is not None:
            self.lrThread.fit(X2,y2)

    def predict(self, X, X2=None):
        predProcess = self.lrProcess.predict(X)
        if X2 is not None:
            predThread  = self.lrThread.predict(X2)
        # TODO combine the two
        
        return predProcess

    def weights(self):
        return None
        #return [self.lrProcess.coef_, self.lrThread.coef_]

    def classifier_(self):
        return None

    def predict_proba(self,X, X2 = None):
        # see http://stackoverflow.com/questions/10104245/python-numpy-combine-array
        return np.vstack((self.lrProcess.classifier_().predict_proba(X),
                   self.lrThread.classifier_().predict_proba(X2))).T



def getClassifiers():
    return [ SVM,
            LogisticRegression,
             Combined,
             RandomClassifier
           ]

