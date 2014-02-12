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
import cosine as cosine
import cofi as cofi



pred_filename  = 'pred-full.csv'
train_filename = 'ratings-train.csv'
test_filename  = 'ratings-test.csv'
user_filename  = 'users.csv'

#pred_filename  = 'simple-pred.csv'
#train_filename = 'r-train100.csv'
#test_filename  = 'r-test100.csv'
#user_filename  = 'u100.csv'


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

def runCosine(training_set,user_list, validation_set, test_queries):
    print "Cosine picked"
    users = {}
    for row in training_set:
        user_id = row['user']
        isbn    = row['isbn']
        if not user_id in users: 
            users[user_id] = {}
            users[user_id]['ratings'] = {}
        users[user_id]['ratings'][isbn] =  row['rating']

    # calculate cosine distance and find closest match
    cosine.topMatch(users)

    # find mean rating per book
    books, global_mean = cosine.meanPerItem(users)

    total_error = 0.0
    sample_count  = 0
    # print "prediction, actual"
    '''
    for row in validation_set:
        user_id = row['user']
        isbn    = row['isbn']
        
        prediction = cosine.predict(users,user_id,books,isbn,global_mean)
        #print prediction, row['rating']
        total_error += abs(prediction - row['rating'])
        sample_count += 1
    return total_error / sample_count
    '''
    
    for query in test_queries:
        user_id = query['user']
        isbn    = query['isbn']
        query['rating'] = cosine.predict(users,user_id,books,isbn,global_mean)

    # Write the prediction file.
    util.write_predictions(test_queries, pred_filename)



#ratings, rating_exists = cofi.buildRatingMatrix(training_data)
#Theta = cofi.buildTheta(user_list)

def main():
    error = 0.0
    print "Data loading...",
    training_data, test_queries, user_list = loadData()
    # split training_data into 80% training and 20% validation
    split = int(len(training_data) * 0.8)
    training_set   = training_data[:split]
    validation_set = training_data[split:]
    print "complete"


    print "Please choose which algorithm to run:"
    print " "
    print "1: Pearson Correlation"
    print "2: Cosine"
    print "x: Exit"
    print " "
    choice = raw_input("Please choose: ")
    if choice == '1':
        error = runPearson(training_set, test_queries)
    elif choice == '2':
        error = runCosine(training_data, user_list, validation_set, test_queries)
    elif choice.lower() == 'x':
        return
    print "Resulting average error is",error

if __name__ == "__main__":
    main()
