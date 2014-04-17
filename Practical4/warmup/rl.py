import grid as g
import random

class Learner(object):
    def __init__(self,rows,cols,maxscore):
        self.states = range(maxscore + 1)
        self.valueMatrix = [(1,1)] * (maxscore + 1)
        self.actions = []
        for x in range(1,rows + 1):
            for y in range(1,cols + 1):
                self.actions.append((x,y))

    def learn(self, grid):
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

    def action(self, state):
        return self.valueMatrix[state]

def playgame():
    grid = g.Grid()
    score = 0
    learner = Learner(grid.getNumRows(),
                      grid.getNumCols(),
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
