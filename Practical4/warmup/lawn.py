pointGrid = [
[7,12,1,14],
[2,13,8,11],
[16,3,10,5],
[9,6,15,4]]

TARGETSCORE = 101

def reward(points):
    if points < 101: return 0
    if points == 101: return 1
    if points > 101: return -1
    return 0 #defensive




#print pointGrid
