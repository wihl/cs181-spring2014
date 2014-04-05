import grid as g

def playgame():
    grid = g.Grid()
    x = 0
    score = 0
    while x != 'x' and grid.reward(score) == 0:
        grid.printGrid()
        x, y = raw_input("Enter row, column: ").split()
        if x == 'x': break
        x = int(x)
        y = int(y)
        if x < 0 or x >= grid.getNumRows():
            print "Invalid row entry"
            continue
        if y < 0 or y >= grid.getNumCols():
            print "Invalid column entry"
            continue
        round = grid.throw(x, y)
        score += round
        print "You got ",round," Current score:", score
    if x == 'x': return
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
