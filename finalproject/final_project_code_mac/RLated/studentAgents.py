from distanceCalculator import Distancer
from game import Actions
from game import Directions
from observedState import ObservedState
import numpy as np
import csv
#import pickle object storing the classificaton model for ghosts
import pickle
from qlearningAgents import PacmanQAgent

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
    """
    An example TeamAgent. After renaming this agent so it is called <YourTeamName>Agent,
    (and also renaming it in registerInitialState() below), modify the behavior
    of this class so it does well in the pacman game!
    """
    
    def __init__(self, *args, **kwargs):
        """
        arguments given with the -a command line option will be passed here
        """



    def registerInitialState(self, gameState):
        """
        Do any necessary initialization
        """
        # Here, you must replace "ExampleTeamAgent" with "<YourTeamName>Agent"
        super(RLatedAgent, self).registerInitialState(gameState)
        
        # Here, you may do any necessary initialization, e.g., import some
        # parameters you've learned, as in the following commented out lines
        # learned_params = cPickle.load("myparams.pkl")
        # learned_params = np.load("myparams.npy")        
        
        self.clf = pickle.load( open( "ghostpredict.p", "rb" ) )

    
    def chooseAction(self, observedState):
	#print observedState
        """
        This  pacman agent will towards the ghost that it is closest to.
        """
        pacmanPosition = observedState.getPacmanPosition()
        ghost_states = observedState.getGhostStates() # states have getPosition() and getFeatures() methods
        legalActs = [a for a in observedState.getLegalPacmanActions()]
        ghost_dists = np.array([self.distancer.getDistance(pacmanPosition,gs.getPosition()) 
                              for gs in ghost_states])

        # find the closest ghost by sorting the distances
        closest_idx = sorted(zip(range(len(ghost_states)),ghost_dists), key=lambda t: t[1])[0][0]


	# store the ghost states AND previous score to the ghost_list

        # take the action that minimizes distance to the current closest ghost
        best_action = Directions.STOP
        best_dist = -np.inf
        for la in legalActs:
            if la == Directions.STOP:
                continue
            successor_pos = Actions.getSuccessor(pacmanPosition,la)
            new_dist = self.distancer.getDistance(successor_pos,ghost_states[closest_idx].getPosition())

	    closestGhostClass = self.clf.predict(ghost_states[closest_idx].getFeatures())
		
            if new_dist > best_dist and closestGhostClass == 5:
                best_action = la
                best_dist = new_dist
	    else:
                best_action = la
                best_dist = new_dist
	   		
        return best_action


class HardCodedAgent(BaseStudentAgent):
    def __init__(self, *args, **kwargs):
        pass

    def registerInitialState(self, gameState):
        super(HardCodedAgent, self).registerInitialState(gameState)
        self.clf = pickle.load( open( "ghostpredict.p", "rb" ) )
    
    def chooseAction(self, observedState):
        pacmanPosition = observedState.getPacmanPosition()
        ghost_states = observedState.getGhostStates() # states have getPosition() and getFeatures() methods
        legalActs = [a for a in observedState.getLegalPacmanActions()]
        ghost_dists = np.array([self.distancer.getDistance(pacmanPosition,gs.getPosition()) 
                              for gs in ghost_states])

        # find the closest ghost by sorting the distances
        sorted_ghosts = sorted(zip(range(len(ghost_states)),ghost_dists), key=lambda t: t[1])
        closest_idx = sorted_ghosts[0][0]
        print closest_idx, sorted_ghosts

        # take the action that minimizes distance to the current closest ghost
        best_action = Directions.STOP
        best_dist = -np.inf
        for la in legalActs:
            if la == Directions.STOP:
                continue
            successor_pos = Actions.getSuccessor(pacmanPosition,la)
            new_dist = self.distancer.getDistance(successor_pos,ghost_states[closest_idx].getPosition())

	    closestGhostClass = self.clf.predict(ghost_states[closest_idx].getFeatures())
		
            if new_dist > best_dist and closestGhostClass == 5:
                best_action = la
                best_dist = new_dist
	    else:
                best_action = la
                best_dist = new_dist
	   		
        return best_action


class QAgent(BaseStudentAgent):
    def __init__(self, *args, **kwargs):
        self.q = PacmanQAgent()

    def registerInitialState(self, gameState):
        super(QAgent, self).registerInitialState(gameState)
        self.clf = pickle.load( open( "ghostpredict.p", "rb" ) )
        self.legal_actions = gameState.getLegalActions(0)
        print self.legal_actions
    
    def chooseAction(self, observedState):

        return self.q.getAction(observedState)
        pacmanPosition = observedState.getPacmanPosition()
        ghost_states = observedState.getGhostStates() # states have getPosition() and getFeatures() methods
        legalActs = [a for a in observedState.getLegalPacmanActions()]
        ghost_dists = np.array([self.distancer.getDistance(pacmanPosition,gs.getPosition()) 
                              for gs in ghost_states])

        # find the closest ghost by sorting the distances
        closest_idx = sorted(zip(range(len(ghost_states)),ghost_dists), key=lambda t: t[1])[0][0]


        # take the action that minimizes distance to the current closest ghost
        best_action = Directions.STOP
        best_dist = -np.inf
        for la in legalActs:
            if la == Directions.STOP:
                continue
            successor_pos = Actions.getSuccessor(pacmanPosition,la)
            new_dist = self.distancer.getDistance(successor_pos,ghost_states[closest_idx].getPosition())

	    closestGhostClass = self.clf.predict(ghost_states[closest_idx].getFeatures())
		
            if new_dist > best_dist and closestGhostClass == 5:
                best_action = la
                best_dist = new_dist
	    else:
                best_action = la
                best_dist = new_dist
	   		
        return best_action

