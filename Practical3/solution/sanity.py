'''
sanity check of prediction file before submission
'''

import argparse
import csv
import numpy as np
import matplotlib.pyplot as plt
import pylab as pl

origDistributions = [3.69, 1.62, 1.2, 1.03, 1.33, 1.26, 1.72, 1.33, 52.14, 0.68, 17.56, 1.04, 12.18, 1.91, 1.30]
virusNames = ['Agent','AutoRun','FraudLoad','FraudPack','Hupigon','Krap','Lipler','Magania','None','Poison','Swizzor','Tdss','VB','Virut','Zbot']

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

    newDistributions = [0] * len(origDistributions)
    print "total predictions", sum(counts), "rows",rows
    for i, val in enumerate(counts):
        newDistributions[i] = float(val)/ float(rows) * 100.0

    width = 0.35
    ind = np.arange(len(origDistributions))
    fig, ax = pl.subplots()
    rects1 = ax.bar(ind, origDistributions, width, color = 'b')
    rects2 = ax.bar(ind+width, newDistributions, width, color = 'r')
    ax.legend( (rects1[0], rects2[0]), ('Original','Predicted') )
    ax.set_xticks(ind + width/2)
    ax.set_xticklabels(virusNames, rotation=90)
    ax.set_title('Prior Day vs Predicted Distribution')
    ax.set_ylabel('Percent')
    pl.show()

if __name__ == "__main__":
    main()
