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

#print my_data[:,0]
#print my_data[:,1]
#print my_data[:,1]


#plot basic data
#pl.figure(1)
#pl.scatter(my_data[:,1], my_data[:,2], c=my_data[:,0])
#pl.show()

#three class generalization of logistic regression 


#tmatrix represents the 'ideal assigment of points and is a n*k column vector'
tmatrix=np.zeros( (len(my_data[:,1]), 3))
for i in xrange(1,len(my_data)):
	tmatrix[i,my_data[i,0]-1]=1

#intialize weights matrix
weights=np.matrix([[1,1,1],[1,1,1], [1,1,1]])
#print weights

def f(weights):
	weights=np.reshape(weights,(3,3))
	sumval=0
	for n in xrange(1,len(tmatrix[:,0])):
		for k in xrange(0, 3):
			a=np.exp(weights[k,0]+my_data[n,1]*weights[k,1]+my_data[n,2]*weights[k,2])
			a_sum=0
			#print a
			for j in xrange(0,3):
				a_sum+=np.exp(weights[j,0]+my_data[n,1]*weights[j,1]+my_data[n,2]*weights[j,2])
			#	print a_sum
			sumval+= tmatrix[n,k]*np.log( np.exp(a/a_sum))
			#print sumval

	return -sumval

initweight=np.asarray([0,0,0, 0,0,0, 0,0,0])
#print initweight
res= optimize.minimize(f, initweight)
print res.x
weights=np.reshape(res.x, (3,3))

for n in xrange(1,len(tmatrix[:,0])):
	checker=[]
	
	for k in xrange(0, 3):
		a=np.exp(weights[k,0]+my_data[n,1]*weights[k,1]+my_data[n,2]*weights[k,2])
		a_sum=0
		for j in xrange(0,3):
			a_sum+=np.exp(weights[j,0]+my_data[n,1]*weights[j,1]+my_data[n,2]*weights[j,2])
		#	print a_sum
		checker.append(np.exp(a/a_sum))
		print checker
	print checker.index(max(checker))



#def gradf(x, *args):
	# u, v = x
 # 	a, b, c, d, e, f = args
 # 	w1 = 2*a*u + b*v + d     # u-component of the gradient
 # 	w2 = b*u + 2*c*v + e     # v-component of the gradient
 # 	w3 = 
 # 	return np.asarray((w1, w2, w3))

#print f(weights)
#res1 = optimize.fmin_cg(f, weights)#, fprime=gradf, args=args)
#print 'res1 = ', res1




def sigmoid(input):
	return 1.0/(1+ numpy.exp(-input))

# def f(x, weights)

# 	return 



#scipy.optimize.fmin_bfgs(f, x0, fprime=None, args=(), gtol=1e-05, norm=inf, epsilon=1.4901161193847656e-08, maxiter=None, full_output=0, disp=1, retall=0, callback=None)[source]