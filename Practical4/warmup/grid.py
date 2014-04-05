'''
grid.py - the main game grid
'''

class Grid(object):
    def __init__(self):
        self.pointGrid = [
            [0, 0, 0, 0, 0, 0],
            [0, 7,12, 1,14, 0],
            [0, 2,13, 8,11, 0],
            [0,16, 3,10, 5, 0],
            [0, 9, 6,15, 4, 0],
            [0, 0, 0, 0, 0, 0]]
        self.numRows = len(self.pointGrid)
        self.numCols = len(self.pointGrid[0])
        self.targetScore = 101

    def getNumRows(self):
        return self.numRows

    def getNumCols(self):
        return self.numCols


    def reward(self,points):
        if points < self.targetScore: return 0
        if points == self.targetScore: return 1
        if points > self.targetScore: return -1
        return 0 #defensive


    def throw(self,x,y):
        ''' 
        given a row x, and a column y, return the points at that location
        '''
        assert x >= 0
        assert x < self.numRows
        assert y >= 0
        assert y < self.numCols
        return self.pointGrid[x][y]

    def printGrid(self):
        for i in range(1, self.numRows - 1):
            for j in range(1, self.numCols - 1):
                print '\t{0} '.format(self.pointGrid[i][j]),
            print

