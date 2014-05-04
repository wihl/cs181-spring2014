from distanceCalculator import Distancer
from game import Actions
from game import Directions
from observedState import ObservedState
import numpy as np
import csv
import random
import pickle

class BaseStudentAgent(object):
    """Superclass of agents students will write"""

    def registerInitialState(self, gameState):
        """Initializes some helper modules"""
        import __main__
        self.display = __main__._display
        self.distancer = Distancer(gameState.data.layout, False)
        self.firstMove = True

    def observationFunction(self, gameState):
        """ maps true state to observed state """
        return ObservedState(gameState)

    def getAction(self, observedState):
        """ returns action chosen by agent"""
        return self.chooseAction(observedState)

    def chooseAction(self, observedState):
        "By default, a BustersAgent just stops.  This should be overridden."
        return Directions.STOP


## Below is the class students need to rename and modify

class RLatedAgent(BaseStudentAgent):
    def __init__(self, *args, **kwargs):
        self.prevTarget = None
        self.bigDistance = 1000
        self.monsterScaredTimeBuffer = 10
        self.minCapsuleThreshold = -175

    def registerInitialState(self, gameState):
        super(RLatedAgent, self).registerInitialState(gameState)
        # TODO fix the paths so they are taken from the directory where this code resides.
        self.ghostPredictor = pickle.load( open( "ghostpredict.p", "rb" ) )
        self.capsulePredictor = pickle.load( open( "capsulepredict.p", "rb" ) )
    
    def updatedTarget(self):
        # update the target based on where it was previously and whether we
        # want to towards (attract) or away (repel) from it
        if self.prevTarget is None:
            self.prevTarget = self.target
            return
        # which direction did the target move?
        deltaX = self.target[0] - self.prevTarget[0]
        deltaY = self.target[1] - self.prevTarget[1]
        self.prevTarget = self.target
        if abs(deltaX) > 1 or abs(deltaY) > 1:
            # our target has changed
            return
        # assume it will keep going in that direction
        newX = min( max(self.target[0] + self.attract * deltaX,1), self.width)
        newY = min( max(self.target[1] + self.attract * deltaY,1), self.height)
        if not self.observedState.hasWall(newX, newY):
            self.target = (newX, newY)
                
    def bestAction(self):
        actionChoices = []
        distances     = []
        for la in self.observedState.getLegalPacmanActions():
            if la == Directions.STOP:
                continue

            successorPos = Actions.getSuccessor(self.pacmanPosition,la)                    
            newDist = self.distancer.getDistance(successorPos,self.target)
            actionChoices.append(la)
            distances.append(newDist)

        if self.attract > 0:
            return actionChoices[distances.index(min(distances))]
        else:
            return actionChoices[distances.index(max(distances))]

    def chooseAction(self, observedState):
        # setup
        self.height = observedState.layout.height
        self.width  = observedState.layout.width
        self.observedState = observedState
        self.pacmanPosition = observedState.getPacmanPosition()

        ghost_states   = observedState.getGhostStates() 
        ghost_dists    = np.array([self.distancer.getDistance(self.pacmanPosition,
                                                              gs.getPosition()) 
                                   for gs in ghost_states])
        isScared       = observedState.scaredGhostPresent()

        # find the closest ghost by sorting the distances
        sorted_ghosts = sorted(zip(range(len(ghost_states)),ghost_dists), key=lambda t: t[1])
        closest_idx = sorted_ghosts[0][0]

        # find the predicted monster
        # There are brief times when there is no monster on the board so choose defaults
        monsterDist = self.bigDistance
        monsterPos = (1,1)
        scaredTimer = 0
        for gs in ghost_states:
            if self.ghostPredictor.predict(gs.getFeatures()) == 5:
                scaredTimer = gs.scaredTimer
                monsterPos = gs.getPosition()
                monsterDist = self.distancer.getDistance(self.pacmanPosition, gs.getPosition())

	# model the capsules
        bestcapsulePos = None
        bestcapsuleDist = self.bigDistance
        for element in observedState.getCapsuleData():
            if self.capsulePredictor.score(element[1]) < self.minCapsuleThreshold:
                continue
            # find the good capsules we can get to without being eaten
            dp = self.distancer.getDistance(self.pacmanPosition, element[0])
            dm = self.distancer.getDistance(monsterPos, element[0])
            if dp < dm:
                # find the closest one
                if dp < bestcapsuleDist:
                    bestcapsuleDist = dp
                    bestcapsulePos = element[0]

        # assume we want to go towards our target
        self.attract = +1

        # choose our target
        self.target = ghost_states[closest_idx].getPosition() # default
        if isScared:
            # can we get to the monster before it reverts? 
            if monsterDist < (scaredTimer + self.monsterScaredTimeBuffer):
                self.target = monsterPos
        else:
            if bestcapsulePos is not None:
                self.target = bestcapsulePos

        if self.target == monsterPos and not isScared:
            # running away from monster
            self.attract = -1

        # "Skate where the puck will be" - The Great One
        self.updatedTarget()

        bestAction = self.bestAction()
        #print "chosen action",bestAction
        #raw_input("Press enter to continue:")
        return bestAction


