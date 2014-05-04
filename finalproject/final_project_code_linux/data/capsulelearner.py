import csv
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn import linear_model
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle 
from sklearn.cross_validation import StratifiedKFold
from sklearn.externals.six.moves import xrange
from sklearn.mixture import GMM
import sklearn as sklearn


with open('capsule_train.csv', 'Ur') as f:
    data = list(tuple(rec) for rec in csv.reader(f, delimiter=' '))

capsulearray=np.array(data)
capsulearray=capsulearray.astype(np.float)
#print capsulearray


kmeans=sklearn.cluster.KMeans(n_clusters=3, init='k-means++', n_init=10, max_iter=300, tol=0.0001, precompute_distances=True, verbose=0, random_state=None, copy_x=True, n_jobs=1)

kmeans.fit(capsulearray)

scorelist=[]
for i in xrange(1, len(capsulearray)):
	scorelist.append(kmeans.score(capsulearray[i,:]))
#print max(scorelist)
#print min(scorelist)

#print kmeans.score(np.array([30,-10, 35]))

#resulting score range on training data was from 0 to -110.15. so anything outside this is probably not a 'good' capsule

#save kmeans results
pickle.dump( kmeans, open( "capsulepredict.p", "wb" ) )


#plot all the positive capsule training points
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#ax.scatter(capsulearray[:,0], capsulearray[:,1], capsulearray[:,2])
#plt.show()
