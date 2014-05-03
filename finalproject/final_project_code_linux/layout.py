# layout.py
# ---------
# Licensing Information:  You are free to use or extend these projects for 
# educational purposes provided that (1) you do not distribute or publish 
# solutions, (2) you retain this notice, and (3) you provide clear 
# attribution to UC Berkeley, including a link to 
# http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero 
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and 
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance, nprand, random
from game import Grid
import os
from collections import deque

VISIBILITY_MATRIX_CACHE = {}

class Layout:
    """
    A Layout manages the static information about the game board.
    """
    def __init__(self, width, height, walls, food, capsules, agentPositions, numGhosts):
        """
        walls and food are Grid objects
        """
        self.width = width
        self.height= height
        self.walls = walls
        self.food = food
        self.capsules = capsules
        self.agentPositions = agentPositions
        self.numGhosts = numGhosts

    def getNumGhosts(self):
        return self.numGhosts

    def initializeVisibilityMatrix(self):
        global VISIBILITY_MATRIX_CACHE
        if reduce(str.__add__, self.layoutText) not in VISIBILITY_MATRIX_CACHE:
            from game import Directions
            vecs = [(-0.5,0), (0.5,0),(0,-0.5),(0,0.5)]
            dirs = [Directions.NORTH, Directions.SOUTH, Directions.WEST, Directions.EAST]
            vis = Grid(self.width, self.height, {Directions.NORTH:set(), Directions.SOUTH:set(), Directions.EAST:set(), Directions.WEST:set(), Directions.STOP:set()})
            for x in range(self.width):
                for y in range(self.height):
                    if self.walls[x][y] == False:
                        for vec, direction in zip(vecs, dirs):
                            dx, dy = vec
                            nextx, nexty = x + dx, y + dy
                            while (nextx + nexty) != int(nextx) + int(nexty) or not self.walls[int(nextx)][int(nexty)] :
                                vis[x][y][direction].add((nextx, nexty))
                                nextx, nexty = x + dx, y + dy
            self.visibility = vis
            VISIBILITY_MATRIX_CACHE[reduce(str.__add__, self.layoutText)] = vis
        else:
            self.visibility = VISIBILITY_MATRIX_CACHE[reduce(str.__add__, self.layoutText)]

    def isWall(self, pos):
        x, col = pos
        return self.walls[x][col]

    def getRandomLegalPosition(self):
        x = random.choice(range(self.width))
        y = random.choice(range(self.height))
        while self.isWall( (x, y) ):
            x = random.choice(range(self.width))
            y = random.choice(range(self.height))
        return (x,y)

    def getRandomDistinctLegalPositions(self, k):
        """
        k is the number of distinct positions; useful if you need to place
        things that can't overlap (e.g., capsules)
        returns a set of positions
        """
        chosen = set()
        xs = range(self.width)
        ys = range(self.height)
        while len(chosen) < k:
            x = random.choice(xs)
            y = random.choice(ys)
            if not self.isWall((x,y)):
                chosen.add((x,y))
        return chosen

    def getRandomCorner(self):
        poses = [(1,1), (1, self.height - 2), (self.width - 2, 1), (self.width - 2, self.height - 2)]
        return random.choice(poses)

    def getFurthestCorner(self, pacPos):
        poses = [(1,1), (1, self.height - 2), (self.width - 2, 1), (self.width - 2, self.height - 2)]
        dist, pos = max([(manhattanDistance(p, pacPos), p) for p in poses])
        return pos

    def isVisibleFrom(self, ghostPos, pacPos, pacDirection):
        row, col = [int(x) for x in pacPos]
        return ghostPos in self.visibility[row][col][pacDirection]

    def deepCopy(self):
        """
        everything that isn't a grid is a list of tuples, so this should be fine
        (tuples are immutable anyway...)
        """
        return Layout(self.width, self.height, self.walls.deepCopy(),
                     self.food.deepCopy(), list(self.capsules), list(self.agentPositions),
                     self.numGhosts)

    def __str__(self):
        return "\n".join(["".join(["=" if self.walls.data[j][i] else " " 
                                            for j in xrange(self.width)]) 
                                               for i in xrange(self.height-1,-1,-1)])
    def __repr__(self):
        return "\n".join(["".join(["=" if self.walls.data[j][i] else " " 
                                            for j in xrange(self.width)]) 
                                               for i in xrange(self.height-1,-1,-1)]) 
                                               
class TextLayout(Layout):

    def __init__(self, layoutText):
        width = len(layoutText[0])
        height= len(layoutText)
        walls = Grid(width, height, False)
        food = Grid(width, height, False)
        capsules = []
        agentPositions = []
        numGhosts = 0
        Layout.__init__(self, width, height, walls, food, capsules, agentPositions, numGhosts)
        self.processLayoutText(layoutText)
        self.layoutText = layoutText
        # self.initializeVisibilityMatrix()

    def processLayoutText(self, layoutText):
        """
        Coordinates are flipped from the input format to the (x,y) convention here

        The shape of the maze.  Each character
        represents a different type of object.
         % - Wall
         . - Food
         o - Capsule
         G - Ghost
         P - Pacman
        Other characters are ignored.
        """
        maxY = self.height - 1
        for y in range(self.height):
            for x in range(self.width):
                layoutChar = layoutText[maxY - y][x]
                self.processLayoutChar(x, y, layoutChar)
        self.agentPositions.sort()
        self.agentPositions = [ ( i == 0, pos) for i, pos in self.agentPositions]

    def processLayoutChar(self, x, y, layoutChar):
        if layoutChar == '%':
            self.walls[x][y] = True
        elif layoutChar == '.':
            self.food[x][y] = True
        elif layoutChar == 'o':
            self.capsules.append((x, y))
        elif layoutChar == 'P':
            self.agentPositions.append( (0, (x, y) ) )
        elif layoutChar in ['G']:
            self.agentPositions.append( (1, (x, y) ) )
            self.numGhosts += 1
        elif layoutChar in  ['1', '2', '3', '4']:
            self.agentPositions.append( (int(layoutChar), (x,y)))
            self.numGhosts += 1

    def __str__(self):
        return "\n".join(self.layoutText)

    def deepCopy(self):
        return TextLayout(self.layoutText[:])

def getLayout(name, back = 2):
    if name.endswith('.lay'):
        layout = tryToLoad('layouts/' + name)
        if layout == None: layout = tryToLoad(name)
    else:
        layout = tryToLoad('layouts/' + name + '.lay')
        if layout == None: layout = tryToLoad(name + '.lay')
    if layout == None and back >= 0:
        curdir = os.path.abspath('.')
        os.chdir('..')
        layout = getLayout(name, back -1)
        os.chdir(curdir)
    return layout

def tryToLoad(fullname):
    if(not os.path.exists(fullname)): return None
    f = open(fullname)
    try: return TextLayout([line.strip() for line in f])
    finally: f.close()


class RandomLayout(Layout):
    """
    should we set height and width randomly too? eh? sure why not
    """
    expected_width = 19
    expected_height = 19
    min_width = 5
    min_height = 5
    wall_prior = 0.1
    neighbor_bonus = 0.055  # add to prior for num walled neighbors (at most 2)

    def __init__(self):
        # set height and width randomly
        width = nprand.poisson(self.expected_width)
        height = nprand.poisson(self.expected_height)
        # make sure the board is big enough
        while width < self.min_width or height < self.min_height:
            width = nprand.poisson(self.expected_width)
            height = nprand.poisson(self.expected_height)
                  
        walls = None # we'll set walls randomly
        food = Grid(width, height, False)
        capsules = []
        agentPositions = []
        numGhosts = 0
        Layout.__init__(self, width, height, walls, food, capsules, agentPositions, numGhosts)
        self.generate()
    
    def generate(self):
        """
        randomly generates a board and ensures it's connected
        a few constraints: walls surround the board, and the corners inside the
        wall are always not walled
        """
        # by convention, the origin is the lower left-hand corner
        # also by convention, the first coordinate is the column, not the row
        valid = False
        while not valid:
            #print "wee"
            walls = Grid(self.width, self.height, False)
            for y in xrange(self.height-1,-1,-1): # we generate top-down
                for x in xrange(self.width):
                    # edges are always walls
                    if y == self.height-1 or y == 0 or x == self.width-1 or x == 0:
                        walls[x][y] = True
                    elif ((y==1 and x==1) or (y==self.height-2 and x==1) 
                                        or (y==self.height-2 and x==self.width-2)
                                        or (y==1 and x==self.width-2)):
                        pass # no walls allowed hur
                    else:
                        # the following will always be defined since we pad with walls
                        left_bonus = self.neighbor_bonus*walls[x-1][y]
                        up_bonus = self.neighbor_bonus*walls[x][y+1]
                        walls[x][y] = bool(nprand.binomial(1,
                                            self.wall_prior+left_bonus+up_bonus))
            # get rid of unit-walls
            for y in xrange(self.height-2,0,-1):
                for x in xrange(1,self.width-1):
                    if walls[x][y] and len(self._neighbors(x,y,walls)) == 4:
                        walls[x][y] = False
            # check that open tiles are connected
            valid = self.valid_board(walls)
        # we found a valid board
        self.walls = walls
        # randomly place pacman
        self.agentPositions.append((0,self.getRandomLegalPosition()))
        
        
    def valid_board(self, walls):
        """
        ensures all non-walled tiles are reachable by doing a bfs search 
        """
        #print "height", self.height, "width", self.width
        total_walled = sum([sum(row) for row in walls.data])
        seen = set([(1,1)])
        q = deque([(1,1)])  # guaranteed to not be a wall
        while len(q) > 0:
            x,y = q.popleft()
            neighbs = [n for n in self._neighbors(x,y,walls) if not n in seen]
            seen.update(neighbs)
            q.extend(neighbs)
        # true if we've seen all the non-walled tiles
        return len(seen) == self.width*self.height - total_walled

    def _neighbors(self,x,y,walls):
        return ([(x,y+i) for i in [-1,1] if not walls[x][y+i] and not y+i > self.height-2 and not y+i < 1]
            + [(x+i,y) for i in [-1,1] if not walls[x+i][y] and not x+i > self.width-2 and not x+i < 1])

