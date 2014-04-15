import grid as g
import random

class Learner(object):
    def __init__(self,rows,cols,grid,targetScore,maxscore):
        self.reward = 0
        self.states = range(maxscore + 1)
        self.rows = rows
        self.cols = cols
        self.currentState = 0
        self.reward = 0
        self.targetScore = targetScore
        self.grid = grid
        self.maxthrow = max(max(grid))

    def findLocation(self,desiredScore):
        print "desired",desiredScore
        for x in range(1,self.rows+1):
            for y in range(1,self.cols+1):
                if self.grid[x][y] == desiredScore:
                    print "trying ",x,y
                    return x, y
        print "did not find it"
        return 1,1

    def getPolicy(self):
        # based on the current state, find the best throw
        desiredScore = self.targetScore - self.currentState
        x = random.randint(1,self.rows)
        y = random.randint(1,self.cols)
        if desiredScore < self.maxthrow:
            # we're close - aim for the closest
            print "getting close..."
            x, y = self.findLocation(desiredScore)
        return x, y

    def action(self, grid):
        self.grid = grid
        x, y = self.getPolicy()
        print x, y
        return x, y

    def setState(self,newState):
        self.currentState = newState

    def setReward(self,newReward):
        self.reward = newReward

def playgame():
    grid = g.Grid()
    x = 0
    score = 0
    learner = Learner(grid.getNumRows(),
                      grid.getNumCols(),
                      grid.getGrid(),
                      grid.getTargetScore(),
                      grid.getMaxScore())
    while grid.reward(score) == 0:
        grid.printGrid()

        x, y = learner.action(grid.getGrid())
        round = grid.noisyThrow(x, y)
        score += round
        learner.setState(score)
        learner.setReward(grid.reward(score))
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
