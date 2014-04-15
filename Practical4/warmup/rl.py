import grid as g
import random

class Learner(object):
    def __init__(self,rows,cols,maxscore):
        self.reward = 0
        self.states = range(maxscore + 1)
        self.rows = rows
        self.cols = cols
        self.currentState = 0
        self.reward = 0

    def action(self):
        x = int(random.random()) * self.rows + 1
        y = int(random.random()) * self.cols + 1
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
                      grid.getMaxScore())
    while grid.reward(score) == 0:
        grid.printGrid()

        x, y = learner.action()
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
