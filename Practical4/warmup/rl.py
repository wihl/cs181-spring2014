import grid as g
import random
import pylab

class Learner(object):
    def __init__(self,rows,cols,maxscore):
        self.states   = range(maxscore + 1)
        self.values   = [0.0] * len(self.states)
        self.policies = [0]   * len(self.states)
        self.valueMatrix = [(1,1)] * (maxscore + 1)
        # actions is a list of (row, column) tuples denoting the target position on the grid 
        self.actions = []
        for x in range(1,rows + 1):
            for y in range(1,cols + 1):
                self.actions.append((x,y))

    def vDistance(self,V1, V2):
        diff = 0.0
        for v1, v2 in zip(V1, V2):
            diff += abs(v1 - v2)
        return diff

    def valueIteration(self, gamma):
        vOld = [random.random()*1000] * len(self.values)
        epsilon = 0.01
        count = 0
        # find action with highest value in case Q is 0.0
        bestScore = 0
        for action in xrange(len(self.actions)):
            x = self.actions[action][0]
            y = self.actions[action][1]
            if self.grid.throw(x,y) > bestScore:
                bestScore = self.grid.throw(x,y)
                bestAction = action

        while self.vDistance(vOld, self.values) >= epsilon:
            count += 1
            vOld = self.values[:]

            for state in xrange(len(self.states)-1,-1,-1):
                Q = [0.0] * len(self.actions)
                for action in xrange(len(self.actions)):
                    Q[action] = self.grid.reward(state) + gamma * self.grid.pbDist(self.actions[action], state)
                if max(Q) != 0.0:
                    self.policies[state] = Q.index(max(Q)) #equivalent of argmax
                else:
                    self.policies[state] = bestAction
                self.values[state] = max(Q)


    def learn(self, grid):
        self.grid = grid
        self.valueIteration(0.9)
        return self.values
        ''' NOT USED 
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
        END NOT USED
        '''

    def action(self, state):
        #return self.valueMatrix[state]
        return self.actions[self.policies[state]]

def playgame():
    grid = g.Grid()
    score = 0
    learner = Learner(grid.getNumRows(),
                      grid.getNumCols(),
                      grid.getMaxScore())
    values = learner.learn(grid)
    print "Values:", values
    while grid.reward(score) == 0:
        grid.printGrid()

        x, y = learner.action(score)
        print "aiming for", x, y
        round = grid.noisyThrow(x, y)
        score += round
        print "You got ",round," Current score:", score

    pylab.plot(values)
    pylab.axhline(y=0.54, xmin=0, xmax=grid.getMaxScore(), color="r")
    pylab.title("Value by State")
    pylab.xlabel("State")
    pylab.ylabel("Value")
    pylab.savefig('values.pdf', format='PDF')
    #pylab.show()

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
