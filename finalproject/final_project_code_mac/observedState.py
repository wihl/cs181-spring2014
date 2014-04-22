import numpy as np

from util import random, nearestPoint, manhattanDistance
from game import Actions
from game_rules import GhostRules

## This class is pacman's interface into the state of the game. The methods defined
## on the ObservedState object (below) are what you should use in implementing
## pacman's strategy.
class ObservedState(object):
    """
    This class encapsulates what pacman knows about the state of the game.
    """
    def __init__(self, gameState=None):
        """
        creates an observed state from a pacman.GameState object by corrupting
        some of the data.
        """
        if gameState is not None:
            self.legal_actions = gameState.getLegalActions(0)
            state_copies = [state.copy() if state is not None else None
                              for state in gameState.data.agentStates]
            self.pacman_state = state_copies[0]
            self.score = gameState.data.score
            self.good_capsule_examps = np.copy(gameState.data.capsule_examples)
            self.layout = gameState.data.layout.deepCopy()
            self.midpoint = ((self.layout.width-2)/2, (self.layout.height-2)/2)
            self.ghost_states = state_copies[1:gameState.getNumAgents()]
            self.scared_ghost_present = self.ghost_states[-1].scaredTimer > 1

            # set scary ghost's timer to 0
            self.ghost_states[-1].scaredTimer = 0

            # save the moves
            self.maxMoves = gameState.maxMoves
            self.numMoves = gameState.numMoves

            # shuffle ghost states
            random.shuffle(self.ghost_states)

            # shuffle capsule data
            z = zip(gameState.data.capsules, [np.copy(feats) for feats in gameState.data.capsule_feats])
            random.shuffle(z)
            self.capsule_pos, self.capsule_feats = zip(*z)

    def copy(self):
        os = ObservedState(None)
        os.legal_actions = list(self.legal_actions)
        os.pacman_state = self.pacman_state.copy()
        os.score = self.score
        os.good_capsule_examps = np.copy(self.good_capsule_examps)
        os.layout = self.layout.deepCopy()
        os.ghost_states = [state.copy() if state is not None else None for state in self.ghost_states]
        os.scared_ghost_present = self.scared_ghost_present
        os.capsule_pos = list(self.capsule_pos)
        os.capsule_feats = [np.copy(feats) for feats in self.capsule_feats]

    def getNumMovesLeft( self ):
        """
            returns: the number of moves remaining in the game
        """
        return self.maxMoves - self.numMoves

    def getGhostQuadrant( self, ghostState ):
        """
            input: ghostState (game.AgentState object)

            returns quadrant 0, 1, 2, 3
            i.e.:
                0 2
                1 3
        """

        x, y = ghostState.getPosition()

        if x <= self.midpoint[0] and y <= self.midpoint[1]:
            return 0
        elif x <= self.midpoint[0] and y > self.midpoint[1]:
            return 1
        elif x > self.midpoint[0] and y <= self.midpoint[1]:
            return 2
        else:
            return 3

    def getLegalPacmanActions( self ):
        """
        returns:
            actions that pacman can take from the current state.
            the returned actions are of type game.Direction.
        """
        return self.legal_actions

    def pacmanFuturePosition(self, actions):
        """
        arguments:
            actions is a list of game.Directions.
        returns:
            the x,y position of pacman after making the sequence of moves
            in the actions variable from the current state.

        if actions is empty, None is returned.
        """
        next_pos = None
        for action in actions:
            vector = Actions.directionToVector( action, 1)
            new_conf = self.pacman_state.configuration.generateSuccessor(vector)
            next_pos = new_conf.getPosition()
        return next_pos

    def ghostFuturePosition(self, ghost_idx, actions):
        """
        arguments:
            ghost_idx assumed to be a valid index into list returned by self.getGhostStates()
            actions is a list of game.Directions
        returns:
            the x,y position of the ghost indexed by ghost_idx after making the sequence
            of moves in the actions variable from the current state.

        if actions is empty, None is returned
        """
        next_pos = None
        for action in actions:
            vector = Actions.directionToVector( action, 1 )
            new_conf = self.ghost_states[ghost_idx].configuration.generateSuccessor(vector)
            next_pos = new_conf.getPosition()
        return next_pos

    def capsuleEdible(self, pacmanPos, capsulePos):
        """
        arguments:
            pacmanPos is an x,y position on the board (representing pacman's location)
            capsulePos is an x,y position on the board (representing a capsule's location)
        returns:
            True if pacman can eat the capsule at capsulePos from pacmanPos, else False
        """
        nearest = nearestPoint(pacmanPos)
        return nearest == capsulePos and manhattanDistance(nearest, pacmanPos) <= 0.5

    def pacmanCollision(self, pacmanPos, ghostPos):
        """
        arguments:
            pacmanPos is an x,y position on the board (representing pacman's location)
            ghostPos is an x,y position on the board (representing a ghost's location)
        returns:
            True if pacman can kill or be killed by the ghost at ghostPos, else False
        """
        return GhostRules.canKill(pacmanPos, ghostPos)

    def getPacmanState( self ):
        """
        returns:
            a game.AgentState object for pacman.
            game.AgentState.getPosition() gives pacman's current x,y position;
            game.AgentState.getDirection() gives the direction pacman was
            most recently taking.
        """
        return self.pacman_state

    def getPacmanPosition( self ):
        """
        returns:
            pacman's current x,y position
        """
        return self.pacman_state.getPosition()

    def getGhostStates(self):
        """
        returns:
            a list of game.AgentState objects for each ghost on the board.
            game.AgentState.getPosition() gives the ghost's current x,y position;
            game.AgentState.getDirection() gives the direction the ghost was
            most recently taking;
            most importantly, game.AgentState.getFeatures() returns a ghost's
            feature vector.
        """
        return self.ghost_states

    def scaredGhostPresent(self):
        """
        returns:
            True if the attacking ghost is currently scared, else False
        """
        return self.scared_ghost_present

    def getScore( self ):
        """
        returns:
            pacman's current score in the game
        """
        return self.score

    def getCapsuleData(self):
        """
        returns:
            a list of ((x,y),feature-vector) tuples, corresponding to the locations
            and feature-vectors (resp.) of the capsules on the board.
        """
        return zip(self.capsule_pos,self.capsule_feats)

    def getGoodCapsuleExamples(self):
        """
        returns:
            a list of feature-vectors from capsules that are known to be good capsules
            (i.e., that make the attacking ghost scared).
            these good capsule examples can be used to figure out which of the capsules
            on the board will make the attacking ghost scared.
        """
        return self.good_capsule_examps

    def hasWall(self, x, y):
        """
        returns:
            True if there is a wall at position x,y, else False
        """
        return self.layout.walls[x][y]
