from distanceCalculator import Distancer
from game import Actions
from game import Directions
from observedState import ObservedState
import numpy as np
import csv
#import pickle object storing the classificaton model for ghosts
import pickle
import random
clf = pickle.load( open( "ghostpredict.p", "rb" ) )
kmeans = pickle.load( open( "capsulepredict.p", "rb" ) )

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
learning_rate_start=0.1
learning_rate=learning_rate_start
discount_factor=0.85
state_space_dict=defaultdict(float)
prob_tryrandom=.001
last_state=None

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
        pass # you probably won't need this, but just in case
    
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

    	#fine the best capsule based on the highest score (e.g. maximum of loss function, using kmeans)
        capsuledata=observedState.getCapsuleData()
        bestcapsule=capsuledata[0][0]
        capsulescores=[]
        for element in capsuledata:
            #print element[1]
            capsulescores.append(kmeans.score(element[1]))
        bestcapsule=capsulescores.index(max(capsulescores))
        #this is the coordinates of the best capsule (e.g. most likely to be good)
        bestcapsulecoords= capsuledata[bestcapsule][0]

        # find the closest ghost by sorting the distances
        closest_idx = sorted(zip(range(len(ghost_states)),ghost_dists), key=lambda t: t[1])[0][0]
	
	ghost_info=[]
	for gs in ghost_states:
	    ghost_info.append(clf.predict(gs.getFeatures()) ==5)
	    ghost_info.append(gs.getPosition())


	state=(pacmanPosition,legalActs, bestcapsulecoords, ghost_info)

        # take the action that minimizes distance to the current closest ghost
        best_action = Directions.STOP
        best_dist = np.inf
        for la in legalActs:
            if la == Directions.STOP:
                continue
            successor_pos = Actions.getSuccessor(pacmanPosition,la)
            new_dist = self.distancer.getDistance(successor_pos,ghost_states[closest_idx].getPosition())
	    

	if self.last_state== None:
	    qval=0
	    legalActs = [a for a in observedState.getLegalPacmanActions()]
	    new_action=random.choice(legalActs)
	    last_action=new_action

	else:
	    #implement Q learning
            qval= state_space_dict[tuple(self.last_state, last action)] +  learning_rate*(self.last_reward + discount_factor*max(state_space_dict.get(tuple([learner.gen_bins(state, nbins), 0]), 0), state_space_dict.get(tuple([learner.gen_bins(state, nbins), 1]), 0)) - state_space_dict.get(tuple([learner.gen_bins(self.last_state, nbins), self.last_action])))
            
            #assign qval to a previous state, action pair 
            state_space_dict[state]=qval

            # You might do some learning here based on the current state and the last state.

            # You'll need to take an action, too, and return it.
            # Return 0 to swing and 1 to jump.
            
            # consider two actions
            multiple_states=(state_space_dict.get(tuple(state)) 
            #select the action that maximized the qvalue
            new_action = multiple_states.index(max(multiple_states))
            #select the 'non optimal' action tryrandom percent of the time to ensure that you exploring enough of the state space
            if (npr.rand()<prob_tryrandom):
                legalActs = [a for a in observedState.getLegalPacmanActions()]
	        new_action=random.choice(legalActs)
                last_action=new_action
        
        new_state  = state
        self.last_action = new_action
        self.last_state  = new_state
	   		
        return best_action

class SampleAgent(BaseStudentAgent):
    """
    An example TeamAgent. After renaming this agent so it is called <YourTeamName>Agent,
    (and also renaming it in registerInitialState() below), modify the behavior
    of this class so it does well in the pacman game!
    """

    def __init__(self, *args, **kwargs):
        """
        arguments given with the -a command line option will be passed here
        """
        pass # you probably won't need this, but just in case
    
    def registerInitialState(self, gameState):
        """
        Do any necessary initialization
        """
        # Here, you must replace "ExampleTeamAgent" with "<YourTeamName>Agent"
        super(SampleAgent, self).registerInitialState(gameState)
        
        # Here, you may do any necessary initialization, e.g., import some
        # parameters you've learned, as in the following commented out lines
        # learned_params = cPickle.load("myparams.pkl")
        # learned_params = np.load("myparams.npy")        
    
    def chooseAction(self, observedState):
	#print observedState
        """
        Here, choose pacman's next action based on the current state of the game.
        This is where all the action happens.
        
        This silly pacman agent will move away from the ghost that it is closest
        to. This is not a very good strategy, and completely ignores the features of
        the ghosts and the capsules; it is just designed to give you an example.
        """
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
            if new_dist > best_dist:
                best_action = la
                best_dist = new_dist
        return best_action
