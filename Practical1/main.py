'''
    main.py
    David Wihl, CSCI-E181, Spring 2014

    Usage:
        python main.py
'''

import numpy as np
import util
import pearson as pearson
import cosine as cosine

pred_filename  = 'pred-full.csv'

def loadData(dataChoice):
    if dataChoice == 'sim':
        train_filename = 'r-train100.csv'
        test_filename  = 'r-test100.csv'
        user_filename  = 'u100.csv'
    else:
        # dataChoice == 'validate' or dataChoice == 'full'
        train_filename = 'ratings-train.csv'
        test_filename  = 'ratings-test.csv'
        user_filename  = 'users.csv'

    training_data  = util.load_train(train_filename)
    test_queries   = util.load_test(test_filename)
    user_list      = util.load_users(user_filename)

    validation_set = {}

    if dataChoice != 'full':
        # split training_data into 80% training and 20% validation
        split = int(len(training_data) * 0.8)
        validation_set = training_data[split:]
        training_data   = training_data[:split]

    return training_data, test_queries, user_list, validation_set

def runCosine(training_set,user_list, validation_set, test_queries):
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
    print "user\tprediction\tactual"

    for row in validation_set:
        user = row['user']
        isbn  = row['isbn']
        
        prediction = cosine.predict(users,user,books,isbn,global_mean)
        print user,"\t",prediction,"\t\t",row['rating']
        total_error += abs(prediction - row['rating'])
        sample_count += 1
    return total_error / sample_count
    '''
    # TODO: parameterize validation vs. full
    for query in test_queries:
        user_id = query['user']
        isbn    = query['isbn']
        query['rating'] = cosine.predict(users,user_id,books,isbn,global_mean)

    # Write the prediction file.
    util.write_predictions(test_queries, pred_filename)
    '''

def main():
    error = 0.0

    print "Please choose:"
    print " "
    print "1: Use simulated data"
    print "2: Use validation data"
    print "3: Use full data"
    print "x: Exit"
    print " "
    choice = raw_input("Please choose: ")
    if choice == '1':
        training_data, test_queries, user_list, validation_set = loadData('sim')
    elif choice == '2':
        print "Please wait ... could take up to 20 minutes"
        training_data, test_queries, user_list, validation_set = loadData('validate')
    elif choice == '3':
        print "Please wait ... could take up to 20 minutes"
        training_data, test_queries, user_list, validation_set = loadData('full')
    elif choice.lower() == 'x':
        return

    error = runCosine(training_data, user_list, validation_set, test_queries)
    if error is not None: print "Resulting average error is",error

if __name__ == "__main__":
    main()
