'''
grid.py - the main game grid
'''

import random 

class Grid(object):
    def __init__(self):
        self.pointGrid = [
            [0, 0, 0, 0, 0, 0],
            [0, 7,12, 1,14, 0],
            [0, 2,13, 8,11, 0],
            [0,16, 3,10, 5, 0],
            [0, 9, 6,15, 4, 0],
            [0, 0, 0, 0, 0, 0]]
        self.numRows = len(self.pointGrid) - 2
        self.numCols = len(self.pointGrid[0]) - 2
        self.maxscore = 117
        self.targetScore = 101
        '''
        introduce randomness to the throw, such that:
            60% of throws are on target
            10% are one of north(-1,0), south(1,0), 
                east (0,1) or west (-1,0)
        '''
        self.noisePattern = [(0,0)] * 6 +    \
                       [(-1,0)] + [(1,0)] +  \
                       [(0,1)]  + [(0,-1)]

    def pbDist(self, action, state):
        # returns the reward distribution for taking a given action at a specific state
        x = action[0]
        y = action[1]
        sum = 0.0
        sum += 0.6 * self.reward(self.throw(x, y) + state)
        sum += 0.1 * self.reward(self.throw(x+1, y) + state)
        sum += 0.1 * self.reward(self.throw(x-1, y) + state)
        sum += 0.1 * self.reward(self.throw(x, y+1) + state)
        sum += 0.1 * self.reward(self.throw(x, y-1) + state)
        print "state", state, "action", action, "sum",sum
        return sum

    def getTargetScore(self):
        return self.targetScore

    def getGrid(self):
        return self.pointGrid

    def getMaxScore(self):
        return self.maxscore

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
        # given a row x, and a column y, 
        # return the points at that location
        return self.pointGrid[x][y]

    def noisyThrow(self,x,y):
        noise = random.choice(self.noisePattern)
        return self.throw(x + noise[0], y + noise[1])        

    def printGrid(self):
        for i in range(1, self.numRows + 1):
            for j in range(1, self.numCols + 1):
                print '\t{0} '.format(self.pointGrid[i][j]),
            print

