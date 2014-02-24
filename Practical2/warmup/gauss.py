import reg
import matplotlib.pyplot as plt

from numpy import *
xArr,yArr=reg.loadDataSet('motorcycle.csv')
'''
ws = reg.standRegres(xArr,yArr)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(xMat[:,1].flatten().A[0], yMat.T[:,0].flatten().A[0])
xCopy = xMat.copy()
yHat = xCopy*ws
ax.plot(xCopy[:,1],yHat)
#plt.show()
'''
yHat = reg.lwlrTest(xArr, xArr, yArr, 1.0)
xMat = mat(xArr)
strInd = xMat[:,1].argsort(0)
xSort = xMat[strInd][:,0,:]
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(xSort[:,1],yHat[strInd])
ax.scatter(xMat[:,1].flatten().A[0], mat(yArr).T.flatten().A[0], s=2, c='red')
plt.show()


