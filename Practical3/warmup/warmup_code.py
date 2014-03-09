#CS 181 PRACTICAL 3 WARM-UP
#Zachary Hendlin and David Wihl
#fmin_bfgs(f, x0[, fprime, args, gtol, norm, ...]) 	Minimize a function using the BFGS algorithm.

#Import relevant libraries
import scipy
import numpy as np
import matplotlib as plot


from numpy import genfromtxt
my_data = genfromtxt('fruit.csv', delimiter=',')


