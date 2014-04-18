import grid as g
import random

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

    def pbSum(self, startState):
        if startState > 101: 
            return 0.0
        if self.grid.reward(startState) != 0:
            return 0.0
        bestValue = float("-inf")
        for action in self.actions:
            vTemp = self.grid.pbDist(action, startState)
            if vTemp > bestValue:
                bestValue = vTemp
        return bestValue



    def valueIteration(self, gamma):
        vOld = [random.random()*1000] * len(self.values)
        epsilon = 0.01
        count = 0
        while self.vDistance(vOld, self.values) >= epsilon:
            print "count",count
            count += 1
            vOld = self.values[:]

            for state in self.states: # TODO - make this for all items
                Q = [0.0] * len(self.actions)
                for action in xrange(len(self.actions)):
                    Q[action] = self.grid.reward(state) + gamma * self.grid.pbDist(self.actions[action], state)
                print "state", state, "Q:",Q
                self.policies[state] = Q.index(max(Q)) #equivalent of argmax
                self.values[state] = max(Q)


    def learn(self, grid):
        # TODO: 
        # value iteration + graph
        # policy iteration
        # use total discounted award utility (sum over inifinity of discounted reward multiplied by gamma-t)
        self.grid = grid
        self.valueIteration(0.9)
        print "values:", self.values
        for x in self.states:
            print "state",x, "action", self.actions[self.policies[x]]
        return
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
    return
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
