import numpy as np
import util

# This makes predictions based on the mean rating for each user in the
# training data.  When there are no training data for a user, it
# defaults to the global mean.

pred_filename  = 'pred-user-mean.csv'
train_filename = 'ratings-train.csv'
test_filename  = 'ratings-test.csv'
user_filename  = 'users.csv'

training_data  = util.load_train(train_filename)
test_queries   = util.load_test(test_filename)
user_list      = util.load_users(user_filename)

# Compute the global mean rating for a fallback.
num_train = len(training_data)
mean_rating = float(sum(map(lambda x: x['rating'], training_data)))/num_train
print "The global mean rating is %0.3f." % (mean_rating)

# Turn the list of users into a dictionary.
# Store data for each user to keep track of the per-user average.
users = {}
for user in user_list:
    users[user['user']] = { 'total': 0, # For storing the total of ratings.
                            'count': 0, # For storing the number of ratings.
                            }
    
# Iterate over the training data to compute means.
for rating in training_data:
    user_id = rating['user']
    users[user_id]['total'] += rating['rating']
    users[user_id]['count'] += 1

# Make predictions for each test query.
for query in test_queries:

    user = users[query['user']]

    if user['count'] == 0:
        # Perhaps we did not having any ratings in the training set.
        # In this case, make a global mean prediction.
        query['rating'] = mean_rating

    else:
        # Predict the average for this user.
        query['rating'] = float(user['total']) / user['count']

# Write the prediction file.
util.write_predictions(test_queries, pred_filename)
