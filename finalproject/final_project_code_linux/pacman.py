# busters.py
# ----------
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
#
# This file (and others) extensively modified (and renamed) for Harvard CS181
#

import sys
import os
import importlib
import util


#############################
# FRAMEWORK TO START A GAME #
#############################

def default(str):
    return str + ' [Default: %default]'

def parseAgentArgs(str):
    if str == None: return {}
    pieces = str.split(',')
    opts = {}
    for p in pieces:
        if '=' in p:
            key, val = p.split('=')
        else:
            key,val = p, 1
        opts[key] = val
    return opts

def readCommand( argv ):
    """
    Processes the command used to run pacman from the command line.
    """
    from optparse import OptionParser
    usageStr = """
    USAGE:      python pacman.py <options>
    """
    parser = OptionParser(usageStr)

    parser.add_option('-T', '--teamName', dest='teamName',
                    help="Enter your team's name", default=None)
    parser.add_option('-p', '--agentName', dest='agentName',
                    help="Enter your agents's name (for testing purposes only)",
                    default=None)
    parser.add_option('-n', '--numGames', dest='numGames', type='int',
                      help=default('the number of GAMES to play'), default=1)
    parser.add_option('-d', '--dataCollectionMode', action='store_true',
        dest='dataCollectionMode', help='Enter data-collection mode', default=False)
    parser.add_option('-a','--agentArgs',dest='agentArgs',
                      help='Comma separated values sent to agent. e.g. "opt1=val1,opt2,opt3=val3"')
    parser.add_option('-q', '--quietTextGraphics', action='store_true', dest='quietGraphics',
                      help='Generate minimal output and no graphics', default=False)
    parser.add_option('-m', '--maxMoves', dest='maxMoves', type='int',
                      help=default('the maximum number of moves in a game'), default=-1)
    parser.add_option('-z', '--zoom', type='float', dest='zoom',
                      help=default('Zoom the size of the graphics window'), default=1.0)
    parser.add_option('-t', '--frameTime', dest='frameTime', type='float',
                      help=default('Time to delay between frames; Must be > 0'), default=0.1)
    parser.add_option('-s', '--seed', dest='seed', type='int',
                      help=default('random seed'), default=3)


    options, otherjunk = parser.parse_args()
    if len(otherjunk) != 0:
        raise Exception('Command line input not understood: ' + otherjunk)

    if options.frameTime <= 0:
        raise Exception("the frameTime option must be greater than 0")

    # set the seed before we do anything else
    util.set_seeds(options.seed)

    # add team directory to python path
    #sys.path.append(os.path.join(os.path.dirname(os.path.abspath("pacman.py")), options.teamDirectory))
    sys.path.append(os.path.dirname(os.path.abspath("pacman.py")))
    args = dict()

    # generate a layout
    import layout
    args['layout'] = layout.RandomLayout()

    # make sure we have a team name, not a directory
    if options.teamName is not None and options.teamName.endswith("/"):
        options.teamName = options.teamName[:-1]

    # Choose a Pacman agent
    pacmanType = loadAgent(options.teamName, options.agentName)
    agentOpts = parseAgentArgs(options.agentArgs)
    pacman = pacmanType(**agentOpts) # Instantiate Pacman with agentArgs
    args['pacman'] = pacman

    import graphicsDisplay
    if options.quietGraphics:
        args['display'] = graphicsDisplay.QuietGraphics()
    else:
        args['display'] = graphicsDisplay.FirstPersonPacmanGraphics(options.zoom, \
                                                                True, \
                                                            frameTime = options.frameTime)

    args['numGames'] = options.numGames
    args['maxMoves'] = options.maxMoves
    args['dataCollectionMode'] = options.dataCollectionMode

    return args

def loadAgent(teamName, agentName):
    """
    if teamDir is None we return the KeyboardBustersAgent; else we return teamName.studentAgents.teamNameAgent
    """
    # students can either use the keyboard inference module or one they've
    # written themselves
    module_name = "keyboardAgents" if teamName is None else teamName+".studentAgents"
    if teamName:
        agent = teamName+"Agent" if agentName is None else agentName
    pacman = "KeyboardAgent" if teamName is None else agent
    try:
        module = importlib.import_module(module_name)
    except ImportError as ex:
        print "Could not import the module", module_name, "; Make sure your agent in your team-name directory"
        raise ex
    if pacman in dir(module):
        return getattr(module, pacman)
    raise Exception("The agent " + pacman+ " does not seem to be specified in " + module_name)


def runGames( layout, pacman, display, numGames, dataCollectionMode, maxMoves):

    # Hack for agents writing to the display
    import __main__
    __main__.__dict__['_display'] = display

    from game_rules import GameRules
    rules = GameRules()
    games = []

    if dataCollectionMode:
        print 'Currently in data collection mode!'

    for i in range( numGames ):
        # params below is a global variable defined above; should presumably move it
        game = rules.newGame(layout, pacman, display, maxMoves,dataCollectionMode )
        game.run()
        games.append(game)

    if numGames > 1:
        scores = [game.state.getScore() for game in games]
        print 'Average Score:', sum(scores) / float(len(scores))
        print 'Scores:       ', ', '.join([str(score) for score in scores])
    else:
        # just print the score
        print "Score:", game.state.getScore()

    return games

if __name__ == '__main__':
    """
    The main function called when pacman.py is run
    from the command line:

    > python pacman.py

    See the usage string for more details.

    > python pacman.py --help
    """
    args = readCommand( sys.argv[1:] ) # Get game components based on input
    runGames( **args )
