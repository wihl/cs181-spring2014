import csv
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn import linear_model
import matplotlib.pyplot as plt
import pickle 


with open('ghost_train.csv', 'Ur') as f:
    data = list(tuple(rec) for rec in csv.reader(f, delimiter=' '))


ghostarray=np.array(data)
ghostarray=ghostarray.astype(np.float)
#print type(ghostarray[2][2])


X=ghostarray[:,[3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,15]]
Y=np.ravel(ghostarray[:,[1]])

#print X
print Y

#fit classification model
clf = SGDClassifier(loss="hinge", penalty="l1")
clf.fit(X, Y)
SGDClassifier(alpha=0.00001, class_weight=None, epsilon=0.1, eta0=0.0,
       fit_intercept=True, l1_ratio=0.15, learning_rate='optimal',
       loss='hinge', n_iter=100, n_jobs=100, penalty='l1', power_t=0.5,
       random_state=False, rho=None, shuffle=False, verbose=0,
       warm_start=False)

pickle.dump( clf, open( "ghostpredict.p", "wb" ) )
predicts=clf.predict(X)


#fit logistic regression classifier
logreg = linear_model.LogisticRegression(C=1e5)
logreg.fit(X, Y)
log_predicts=logreg.predict(X)

#print predicts
actuals=Y
plt.scatter(predicts, actuals)
plt.xlabel("prediction")
plt.ylabel("actual")
#	plt.show()

counter=0
ttl=0
for i in xrange(1, len(actuals)):
	if actuals[i]==5:
		ttl+=1
	if actuals[i]==5 and predicts[i]==5:
		counter+=1
print 1.0*counter/ttl



print "logistic regression:"
counter=0
ttl=0
for i in xrange(1, len(actuals)):
	if actuals[i]==5:
		ttl+=1
	if actuals[i]==5 and log_predicts[i]==5:
		counter+=1
print 1.0*counter/ttl


#print np.average(predicts)
#print np.average(actuals)

# plt.scatter(ghostarray[:,1],ghostarray[:,2])
# plt.xlabel("class")
# plt.ylabel("score")
# plt.show()
