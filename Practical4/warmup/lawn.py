pointGrid = [
    [0, 0, 0, 0, 0, 0],
    [0, 7,12, 1,14, 0],
    [0, 2,13, 8,11, 0],
    [0,16, 3,10, 5, 0],
    [0, 9, 6,15, 4, 0],
    [0, 0, 0, 0, 0, 0]]

TARGETSCORE = 101

def reward(points):
    if points < 101: return 0
    if points == 101: return 1
    if points > 101: return -1
    return 0 #defensive


def throw(x,y):
    ''' 
    given a row x, and a column y, return the points at that location
    '''
    global pointGrid
    assert x >= 0
    assert x < len(pointGrid)
    assert y >= 0
    assert y < len(pointGrid[0])
    return pointGrid[x][y]

def printGrid():
    global pointGrid
    for i in range(1, len(pointGrid) - 1):
        for j in range(1, len(pointGrid[0]) - 1):
            print '\t{0} '.format(pointGrid[i][j]),
        print

#throw(-1,-1)
#print pointGrid
