import grid as g
import random

class Learner(object):
    def __init__(self,rows,cols,grid,targetScore,maxscore):
        self.states = range(maxscore + 1)
        self.rows = rows
        self.cols = cols
        self.targetScore = targetScore
        self.grid = grid
        self.maxthrow = max(max(grid))
        self.valueMatrix = [(1,1)] * (maxscore + 1)
        self.actions = []
        for x in range(1,self.rows + 1):
            for y in range(1,self.cols + 1):
                self.actions.append((x,y))

    def learn(self, grid):
        #self.valueMatrix = self.buildValueMatrix(grid.getGrid(), self.states)
        #return
        iterations = 1
        for i in xrange(iterations):
            # keep refining value matrix
            state = 0
            for state in self.states:
                bestValue = grid.throw(self.actions[0][0], self.actions[0][1])
                self.valueMatrix[state] = self.actions[0]
                # go through all possible actions at this state
                for action in self.actions:
                    value = grid.throw(action[0], action[1])
                    reward = grid.reward(state + value)
                    if reward == 1:
                        # is there are reward? if so, put in that action
                        self.valueMatrix[state] = action
                        break
                    elif reward == 0:
                        # if there isn't a reward, choose the action with the best value
                        if value > bestValue:
                            bestValue = value
                            self.valueMatrix[state] = action
                    else : #reward == -1:
                        self.valueMatrix[state] = (0,0)
        print self.valueMatrix
                

    def buildValueMatrix(self,grid,states):
        # for each state, create the best throw
        x, y = self.findLocation(self.maxthrow)
        valueMatrix = []
        for state in self.states:
            # when we aren't close, aim for the largest number
            if state < (self.targetScore - self.maxthrow):
                valueMatrix.append((x,y))
            elif state < self.targetScore:
            # as we get closer, aim for the desired number
                x, y = self.findLocation(self.targetScore - state)
                valueMatrix.append((x,y))
            else:
            # we're over - fill in the rest with zeroes
                valueMatrix.append((0,0))
        return valueMatrix
            

    def findLocation(self,desiredScore):
        for x in range(1,self.rows+1):
            for y in range(1,self.cols+1):
                if self.grid[x][y] == desiredScore:
                    return x, y
        return 1,1 # defensive

    def action(self, state):
        return self.valueMatrix[state]

def playgame():
    grid = g.Grid()
    score = 0
    learner = Learner(grid.getNumRows(),
                      grid.getNumCols(),
                      grid.getGrid(),
                      grid.getTargetScore(),
                      grid.getMaxScore())
    learner.learn(grid)
    while grid.reward(score) == 0:
        grid.printGrid()

        x, y = learner.action(score)
        print "aiming for", x, y
        round = grid.noisyThrow(x, y)
        score += round
        print "You got ",round," Current score:", score
    if grid.reward(score) == -1:
        print "You lose."
        return
    if grid.reward(score) == 1:
        print "You win!"
        return

def main():
    playgame()

if __name__ == '__main__':
    main()
