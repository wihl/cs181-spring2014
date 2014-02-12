'''
    main.py
    David Wihl, CSCI-E181, Spring 2014

    Usage:
        python main.py
'''
import argparse
import numpy as np
import util
import pearson as pearson
import cofi as cofi



#pred_filename  = 'simple-pred.csv'
#train_filename = 'ratings-train.csv'
#test_filename  = 'ratings-test.csv'
#user_filename  = 'users.csv'

pred_filename  = 'small-pred.csv'
train_filename = 'r-train100.csv'
test_filename  = 'r-test100.csv'
user_filename  = 'u100.csv'


def loadData():
    training_data  = util.load_train(train_filename)
    test_queries   = util.load_test(test_filename)
    user_list      = util.load_users(user_filename)
    return training_data, test_queries, user_list


def runPearson(training_data, test_queries):
    print "Pearson picked"
    users = {}
    for rating in training_data:
        user_id = rating['user']
        isbn    = rating['isbn']
        if not user_id in users: users[user_id] = {}
        users[user_id][isbn] =  rating['rating']

    pred_rating = {}

    for user in users:
        match = pearson.topMatches(users,user)
        print user, match
        #        pred_rating[user] = pearson.getRecommendations(users,user)
    '''
    for query in test_queries:
        user_id = query['user']
        isbn    = query['isbn']
        if user_id in pred_rating:
            query['rating'] = pred_rating[user_id].get(isbn,4)
        else:
            print user_id, "not found - defaulting"
            query['rating'] = 4

    # Write the prediction file.
    util.write_predictions(test_queries, pred_filename)
    '''
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
            F1 = runPearson(training_data, test_queries)
        elif choice == '2':
            F1 = runCoFi(training_data, user_list)
        elif choice.lower() == 'x':
            break
        print "Resulting F1 quality is",F1
        
    print "Thanks and good-bye!"
            

if __name__ == "__main__":
    main()
