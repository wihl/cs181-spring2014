'''
    main.py
    David Wihl, CSCI-E181, Spring 2014

    Usage:
        python main.py
'''
import argparse
import numpy as np
import util
import cofi as cofi


pred_filename  = 'simple-pred.csv'
train_filename = 'ratings-train.csv'
test_filename  = 'ratings-test.csv'
user_filename  = 'users.csv'

def loadData():
    training_data  = util.load_train(train_filename)
    test_queries   = util.load_test(test_filename)
    user_list      = util.load_users(user_filename)
    return training_data, test_queries, user_list


def runPearson(training_data):
    print "Pearson picked"
    return 1.0

def runCoFi(training_data, user_list):
    print "CoFi picked"
    #ratings, rating_exists = cofi.buildRatingMatrix(training_data)
    #Theta = cofi.buildTheta(user_list)

    return 1.0

#ratings, rating_exists = cofi.buildRatingMatrix(training_data)
#Theta = cofi.buildTheta(user_list)

def main():
    F1 = 0.0
    print "Data loading...",
    training_data, test_queries, user_list = loadData()
    print "complete"

    while True:
        print "Please choose which algorithm to run:"
        print " "
        print "1: Pearson Correlation"
        print "2: Collaborative Filtering"
        print "x: Exit"
        print " "
        choice = raw_input("Please choose: ")
        if choice == '1':
            F1 = runPearson(training_data)
        elif choice == '2':
            F1 = runCoFi(training_data, user_list)
        elif choice.lower() == 'x':
            break
        print "Resulting F1 quality is",F1
        
    print "Thanks and good-bye!"
            

if __name__ == "__main__":
    main()
