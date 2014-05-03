import csv
import numpy as np
from sklearn.linear_model import SGDClassifier
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


clf = SGDClassifier(loss="hinge", penalty="l1")
clf.fit(X, Y)
SGDClassifier(alpha=0.0001, class_weight=None, epsilon=0.1, eta0=0.0,
       fit_intercept=True, l1_ratio=0.15, learning_rate='optimal',
       loss='hinge', n_iter=100, n_jobs=10, penalty='l1', power_t=0.5,
       random_state=True, rho=None, shuffle=False, verbose=0,
       warm_start=False)

pickle.dump( clf, open( "ghostpredict.p", "wb" ) )

predicts=clf.predict(X)
#print predicts
actuals=Y
plt.scatter(predicts, actuals)
plt.xlabel("prediction")
plt.ylabel("actual")
#plt.show()

#print np.average(predicts)
#print np.average(actuals)

# plt.scatter(ghostarray[:,1],ghostarray[:,2])
# plt.xlabel("class")
# plt.ylabel("score")
# plt.show()
