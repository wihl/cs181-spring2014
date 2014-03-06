'''
sanity check of prediction file before submission
'''

import argparse
import csv

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('file',nargs='?',default = "mypredictions.csv",
                        help='Number of cross validation iterations to run')
    args = parser.parse_args()

    counts = [0] * 15
    rows   = 0

    print "opening ", args.file
    with open(args.file, 'rb') as f:
        mycsv = csv.reader(f) # read and discard the first row
        mycsv.next()
        for row in mycsv:
            rows += 1
            pred = int(row[-1])
            assert pred >= 0
            assert pred < 15
            counts[pred] += 1

    print "total predictions", sum(counts), "rows",rows
    for i, val in enumerate(counts):
        print i,'\t',float(val)/ float(rows) * 100.0

if __name__ == "__main__":
    main()
