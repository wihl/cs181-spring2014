#CS 181 PRACTICAL 3 WARM-UP
#Zachary Hendlin and David Wihl
#fmin_bfgs(f, x0[, fprime, args, gtol, norm, ...]) 	Minimize a function using the BFGS algorithm.

#Import relevant libraries
import scipy
from scipy import optimize
import numpy as np
import matplotlib as plot
import pylab as pl


from numpy import genfromtxt
my_data = genfromtxt('fruit.csv', delimiter=',')



#print my_data[:,1]


#plot basic data
#pl.figure(1)
#pl.scatter(my_data[:,1], my_data[:,2],_data = genfromtxt('fruit.csv', delimiter=',') c=my_data[:,0])
#pl.show()

#three class generalization of logistic regression 


#tmatrix represents the 'ideal assigment of points and is a n*k column vector'
tmatrix=np.zeros( (len(my_data[:,1]), 3))
for i in xrange(1,len(my_data)):
	tmatrix[i,my_data[i,0]-1]=1

tmatrix=np.delete(tmatrix, (0),0)
#print tmatrix
#print my_data

#intialize weights matrix
#weights=np.matrix([[1,1,1],[1,1,1], [1,1,1]])
#print weights

def f(weights):
	weights=np.reshape(weights,(3,3))
#	print weights
	sumval=0.
	for n in xrange(1,len(tmatrix[:,0]+1)):
		for k in xrange(0, 3):
			#print "w0   ",weights[k,0], "\t     w1",weights[k,1], "\t w2",weights[k,2]
			#print "MD0   ",my_data[n,1], "\t     w1",my_data[n,2]

			a=(weights[k,0]+my_data[n,1]*weights[k,1]+my_data[n,2]*weights[k,2])
			a_sum=0.
			#print "a=", a
			for j in xrange(0,3):
				a_sum+=np.exp(weights[j,0]+my_data[n,1]*weights[j,1]+my_data[n,2]*weights[j,2])
				#print a_sum
			sumval+= tmatrix[n,k]*(a -np.log(a_sum))
			#print sumval

	return -sumval

initweight=np.asarray([1.,0.,0., 1.,0.,0., 1.,0.,0.])
#print np.reshape(initweight, (3,3))
res= optimize.fmin_bfgs(f, initweight, full_output=1)
res= res[0]

weights=np.reshape(np.ravel(res), (3,3))
print res


ASSIGNED_VALUES=[]
#assign points to a cluster
for n in xrange(0,len(tmatrix[:,0]+1)):
	checker=[]
	for k in xrange(0, 3):
		a=(weights[k,0]+my_data[n,1]*weights[k,1]+my_data[n,2]*weights[k,2])
		a_sum=0.
		for j in xrange(0,3):
			a_sum+=np.exp(weights[j,0]+my_data[n,1]*weights[j,1]+my_data[n,2]*weights[j,2])
		#	print a_sum
		checker.append(a -np.log(a_sum))
#		print checker
	ASSIGNED_VALUES.append(checker.index(max(checker)))

print ASSIGNED_VALUES

#Graphing parameters
Y=my_data[1:,0]
X=my_data[1:,1:3]
h=.02

# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, m_max]x[y_min, y_max].
x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

#print xx
#print len(ASSIGNED_VALUES)
newvar=np.array(ASSIGNED_VALUES)

#Z=newvar.reshape(xx.shape)
#print newvar

pl.figure(1, figsize=(4, 3))
#pl.pcolormesh(xx, yy, Z, cmap=pl.cm.Paired)
pl.scatter(X[:, 0], X[:, 1], c=Y, edgecolors='k', cmap=pl.cm.Paired)

pl.xlim(xx.min(), xx.max())
pl.ylim(yy.min(), yy.max())
pl.xticks(())
pl.yticks(())

pl.show()