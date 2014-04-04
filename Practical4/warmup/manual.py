import lawn

def playgame():
    lawn.printGrid()
    x, y = raw_input("Enter row, column: ").split()
    print "You got ",lawn.throw(int(x),int(y))

def main():
    playgame()

if __name__ == '__main__':
    main()
