#CS 181 PRACTICAL 3 WARM-UP
#Zachary Hendlin and David Wihl

#Import relevant libraries
import scipy
from scipy import optimize
import numpy as np
import matplotlib as plot
import pylab as pl
from numpy import genfromtxt

my_data = genfromtxt('fruit.csv', delimiter=',')

####################################################
#three class generalization of logistic regression 
####################################################

#tmatrix represents the 'ideal assigment of points and is a n*k column vector'
tmatrix=np.zeros( (len(my_data[:,1]), 3))
for i in xrange(1,len(my_data)):
	tmatrix[i,my_data[i,0]-1]=1

tmatrix=np.delete(tmatrix, (0),0)
#print tmatrix
#print my_data

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

#assign points to a cluster to check assignments
ASSIGNED_VALUES=[]
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

#print ASSIGNED_VALUES

#Graphing parameters
Y=my_data[1:,0]
X=my_data[1:,1:3]
h=.02

# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, m_max]x[y_min, y_max].
x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))


xx_ravel=xx.ravel()
yy_ravel=yy.ravel()

Z_vals=[]
for n in xrange(0,len(xx_ravel)):
	checker=[]
	for k in xrange(0, 3):
		a=(weights[k,0]+xx_ravel[n]*weights[k,1]+yy_ravel[n]*weights[k,2])
		a_sum=0.
		for j in xrange(0,3):
			a_sum+=np.exp(weights[j,0]+xx_ravel[n]*weights[j,1]+yy_ravel[n]*weights[j,2])
		#	print a_sum
		checker.append(a -np.log(a_sum))
#		print checker
	Z_vals.append(checker.index(max(checker)))


Z=np.array(Z_vals).reshape(xx.shape)

##make first contour plot for logistic classifer
pl.figure(1, figsize=(8, 6))
pl.pcolormesh(xx, yy, Z, cmap=pl.cm.Paired)
pl.scatter(X[:, 0], X[:, 1], c=Y, edgecolors='k', cmap=pl.cm.Paired)

pl.xlim(xx.min(), xx.max())
pl.ylim(yy.min(), yy.max())
pl.show()

####################################################
#Generative model
####################################################

#set up variables for the three classes
d1x=[]
d2x=[]
d3x=[]
d1y=[]
d2y=[]
d3y=[]

#create seperate datasets for each group
for datum in my_data:
	if datum[0]==1:
		d1x.append(datum[1])
		d1y.append(datum[2])
	if datum[0]==2:
		d2x.append(datum[1])
		d2y.append(datum[2])
	if datum[0]==3:
		d3x.append(datum[1])
		d3y.append(datum[2])

#compute means
d1x_m=np.mean(d1x)
d1y_m=np.mean(d1y)
d2x_m=np.mean(d2x)
d2y_m=np.mean(d2y)
d3x_m=np.mean(d3x)
d3y_m=np.mean(d3y)

#create arrays for each class to calculate Gaussians
d1=np.transpose(np.asarray([d1x,d1y]))
d2=np.transpose(np.asarray([d2x,d2y]))
d3=np.transpose(np.asarray([d3x,d3y]))

d1cov=np.cov(d1, rowvar=0)
d2cov=np.cov(d2, rowvar=0)
d3cov=np.cov(d3, rowvar=0)

#calc prior probabilities P(C=k)
c1_prior=1.0*len(d1x)/len(my_data)
c2_prior=1.0*len(d2x)/len(my_data)
c3_prior=1.0*len(d3x)/len(my_data)

print c1_prior
print c2_prior
print c3_prior

def multivar_norm(value, means, covariance):
	k=len(value)
	#print k
	return (1/np.sqrt(np.power((2*np.pi),k)*np.linalg.det(covariance)) )* np.exp(-.5*np.dot(np.dot(np.transpose(value - means),np.linalg.inv(covariance)), value - means))

#print multivar_norm(np.array([5,5]), np.array([d1x_m, d1y_m]), d1cov)

generative_z=[]
for n in xrange(0,len(xx_ravel)):
	probs=[
	multivar_norm(np.array([xx_ravel[n], yy_ravel[n]]), np.array([d1x_m, d1y_m]), d1cov)*c1_prior,
	multivar_norm(np.array([xx_ravel[n], yy_ravel[n]]), np.array([d2x_m, d2y_m]), d2cov)*c2_prior,
	multivar_norm(np.array([xx_ravel[n], yy_ravel[n]]), np.array([d3x_m, d3y_m]), d3cov)*c3_prior
	]
	generative_z.append(probs.index(max(probs)))

generative_z=np.array(generative_z).reshape(xx.shape)
# apply softmax to assign point to most likely class, ignoring denominator since it is the same for all classes

#make 2nd plot
pl.figure(2, figsize=(8, 6))
pl.pcolormesh(xx, yy, generative_z, cmap=pl.cm.Paired)
pl.scatter(X[:, 0], X[:, 1], c=Y, edgecolors='k', cmap=pl.cm.Paired)

pl.xlim(xx.min(), xx.max())
pl.ylim(yy.min(), yy.max())
pl.show()


#make basic plot 
pl.figure(3, figsize=(8, 6))
pl.scatter(X[:, 0], X[:, 1], c=Y, edgecolors='k', cmap=pl.cm.Paired)

pl.xlim(xx.min(), xx.max())
pl.ylim(yy.min(), yy.max())
pl.show()
