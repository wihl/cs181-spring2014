import numpy as np
import util
import recommendations as r

pred_filename  = 'simple-pred.csv'
train_filename = 'ratings-train.csv'
test_filename  = 'ratings-test50.csv'
user_filename  = 'users.csv'

training_data  = util.load_train(train_filename)
test_queries   = util.load_test(test_filename)
user_list      = util.load_users(user_filename)

users = {}

# TODO: account for user demographic
 
for rating in training_data:
    user_id = rating['user']
    isbn    = rating['isbn']
    if not user_id in users: users[user_id] = {}
    users[user_id][isbn] =  rating['rating']

pred_rating = {}
c = 0
for user in users:
    c += 1
    if c%50 == 0: print "%d / %d" % (c,len(users))
    pred_rating[user] = r.getRecommendations(users,user)


# TODO: handle predictions based on missing values

for query in test_queries:
    user_id = query['user']
    isbn    = query['isbn']
    if user_id in pred_rating:
        query['rating'] = pred_rating[user_id].get(isbn,4)
    else:
        query['rating'] = 4

# Write the prediction file.
util.write_predictions(test_queries, pred_filename)

