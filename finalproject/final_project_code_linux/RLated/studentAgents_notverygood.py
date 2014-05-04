from distanceCalculator import Distancer
from game import Actions
from game import Directions
from observedState import ObservedState
import numpy as np
import numpy.random as npr
import csv
#import pickle object storing the classificaton model for ghosts
import pickle
import random
from collections import defaultdict
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
learning_rate_start=0.4
learning_rate=learning_rate_start
discount_factor=0.85
state_space_dict=defaultdict(float)
prob_tryrandom=.001


class RLatedAgent(BaseStudentAgent):
    """
    An example TeamAgent. After renaming this agent so it is called <YourTeamName>Agent,
    (and also renaming it in registerInitialState() below), modify the behavior
    of this class so it does well in the pacman game!
    """
    
    def __init__(self, *args, **kwargs):
        self.last_state=None
        self.last_action=None
        self.last_score=0
        self.new_score=0
        self.turns_since_capsule=100
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
        """
        This  pacman agent will towards the ghost that it is closest to.
        """
        self.new_score=observedState.getScore()
        self.last_reward=self.new_score - self.last_score

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
        #closest_idx = sorted(zip(range(len(ghost_states)),ghost_dists), key=lambda t: t[1])[0][0]
	
    	ghost_info=[]
    	for gs in ghost_states:
            if clf.predict(gs.getFeatures()) ==5:
                ghost_info.append(1)
            else:
                ghost_info.append(0)
    	    ghost_info.append(gs.getPosition())


    	state=(tuple(pacmanPosition), tuple(bestcapsulecoords), tuple(ghost_info))
        #print state

        # take the action that minimizes distance to the current closest ghost
        best_action = Directions.STOP

    	if self.last_state== None:
    	    qval=0
    	    legalActs = [a for a in observedState.getLegalPacmanActions()]
    	    best_action=random.choice(legalActs)
    	    self.last_action=best_action
    	    self.last_state=state
            newlist=tuple([state, self.last_action])
            #print newlist
            #print type(newlist)
            state_space_dict[newlist]=qval
            #print best_action
            return best_action

    	else:
    	    #implement Q learning
    	    state_action_list=[]
            best_val=-1
            for act in legalActs:
                state_action_list.append(state_space_dict.get(tuple([state,act])))
                if state_space_dict.get(tuple([state,act])) > best_val:
                    bestval=state_space_dict.get(tuple([state,act]))
                    savestate=act
    	    

            if best_val==-1:
                state_action_list=[0,0]
            #print state_action_list
            qval= state_space_dict[tuple([self.last_state, self.last_action])]+ learning_rate*(self.last_reward + discount_factor*best_val- state_space_dict.get(tuple([self.last_state,self.last_action])))
            state_space_dict[tuple([state, self.last_action])]=qval
            multiple_states=state_action_list
            #print state_space_dict
                #select the action that maximized the qvalue
            #print "hit here"
            if best_val==-1:
                #print "1"
                legalActs = [a for a in observedState.getLegalPacmanActions()]
                best_action=random.choice(legalActs)
                #select the 'non optimal' action tryrandom percent of the time to ensure that you exploring enough of the state space
            elif (npr.rand()<prob_tryrandom):
                legalActs = [a for a in observedState.getLegalPacmanActions()]
    	        best_action=random.choice(legalActs)
                self.last_action=best_action
            else:
                print "--------------------------"
                best_action = act

            
            self.last_action = best_action
            self.last_state  = state
    	    #print best_action
            return best_action



class SampleAgent(BaseStudentAgent):
    """
    An example TeamAgent. After renaming this agent so it is called <YourTeamName>Agent,
    (and also renaming it in registerInitialState() below), modify the behavior
    of this class so it does well in the pacman game!
    """

    def __init__(self, *args, **kwargs):
        self.turns_since_capsule=100
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

        capsuledata=observedState.getCapsuleData()
        bestcapsule=capsuledata[0][0]
        capsulescores=[]
        for element in capsuledata:
            #print element[1]
            capsulescores.append(kmeans.score(element[1]))
        bestcapsule=capsulescores.index(max(capsulescores))
        #this is the coordinates of the best capsule (e.g. most likely to be good)
        bestcapsulecoords= capsuledata[bestcapsule][0]

        # take the action that minimizes distance to the current closest ghost
        best_action = Directions.STOP
        best_dist = 100
        new_dist=100
        for la in legalActs:
                if la == Directions.STOP:
                    pass
                else:
                    successor_cap_pos = Actions.getSuccessor(pacmanPosition,la)
                    curr_cap_dist= self.distancer.getDistance(pacmanPosition,bestcapsulecoords)
                    new_cap_dist = self.distancer.getDistance(successor_cap_pos,bestcapsulecoords)

                    for ghost in ghost_states:
                        successor_pos = Actions.getSuccessor(pacmanPosition,la)
                        curr_dist= self.distancer.getDistance(pacmanPosition,ghost.getPosition())
                        new_dist = self.distancer.getDistance(successor_pos,ghost.getPosition())

                        if clf.predict(ghost.getFeatures())==5 and curr_dist<3 and self.turns_since_capsule>=15 and observedState.hasWall(int(successor_pos[0]), int(successor_pos[1])) ==False:
                            if new_dist>curr_dist:
                                self.turns_since_capsule+=1
                                return la

                        elif self.turns_since_capsule<15 and clf.predict(ghost.getFeatures())==5 and curr_dist<15 and observedState.hasWall(int(successor_pos[0]), int(successor_pos[1])) ==False:
                            if new_dist<=curr_dist:
                                self.turns_since_capsule+=1
                                return la
                        else:
                            if new_cap_dist<curr_cap_dist-.9 and observedState.hasWall(int(successor_pos[0]), int(successor_pos[1])) ==False:
                                    if new_cap_dist==0:
                                        self.turns_since_capsule=0
                                    return la

                            # else:
                            #     if new_dist<best_dist and observedState.hasWall(int(successor_pos[0]), int(successor_pos[1])) ==False:
                            #         best_action=la
                            #         self.turns_since_capsule+=1
        return best_action
