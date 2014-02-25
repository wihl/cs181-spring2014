'''
CSCI-E181 Practical 2

 Locally weighted Linear Regression.

Modified from "Machine Learning in Action", Peter Harrington, ISBN 9781617290183
'''
from numpy import *
import matplotlib.pyplot as plt


def loadDataSet(fileName): 
    '''
    general function to parse comma-delimited floats
    '''
    dataMat = []; labelMat = []
    fr = open(fileName)
    numFeat = len(fr.readline().split(',')) - 1 #get number of fields from header row
    for line in fr.readlines():
        lineArr =[]
        curLine = line.strip().split(',')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append([1] + lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat,labelMat

def lwlr(testPoint,xArr,yArr,k=1.0):
    '''
    locally weight linear regression implementation
    '''
    xMat = mat(xArr); yMat = mat(yArr).T
    m = shape(xMat)[0]
    weights = mat(eye((m)))
    for j in range(m):                      #next 2 lines create weights matrix
        diffMat = testPoint - xMat[j,:]     #
        weights[j,j] = exp(diffMat*diffMat.T/(-2.0*k**2))
    xTx = xMat.T * (weights * xMat)
    if linalg.det(xTx) == 0.0:
        print "This matrix is singular, cannot do inverse"
        return
    ws = xTx.I * (xMat.T * (weights * yMat))
    return testPoint * ws

def lwlrTest(testArr,xArr,yArr,k=1.0):  #loops over all the data points and applies lwlr to each one
    m = shape(testArr)[0]
    yHat = zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArr[i],xArr,yArr,k)
    return yHat

def lwlrTestPlot(xArr,yArr,k=1.0):  #same thing as lwlrTest except it sorts X first
    yHat = zeros(shape(yArr))       #easier for plotting
    xCopy = mat(xArr)
    xCopy.sort(0)
    for i in range(shape(xArr)[0]):
        yHat[i] = lwlr(xCopy[i],xArr,yArr,k)
    return yHat,xCopy

def computeCost(yArr,yHatArr): #yArr and yHatArr both need to be arrays
    m = float(len(yArr))
    return (1.0/(2.0*m)) * ((yArr-yHatArr)**2).sum()

def main():
    xArr,yArr=loadDataSet('motorcycle.csv')
    yHat = lwlrTest(xArr, xArr, yArr, 1.0)
    xMat = mat(xArr)
    strInd = xMat[:,1].argsort(0)
    xSort = xMat[strInd][:,0,:]
    print "Final cost of LWLR is ",computeCost(yArr,yHat) 
    fig = plt.figure()
    yErr = yArr - yHat
    #plt.errorbar(xSort[:,1], yHat[strInd], yErr )
    ax = fig.add_subplot(111)
    ax.plot(xSort[:,1],yHat[strInd])
    ax.scatter(xMat[:,1].flatten().A[0], mat(yArr).T.flatten().A[0], s=2, c='red')
    plt.show()
    return

if __name__ == "__main__":
    main()


