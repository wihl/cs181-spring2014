import numpy as np
import util

# This is just about the dumbest possible predictor, but it shows the
# really basic things you need to know to read in the training data
# and write a valid prediction file.

pred_filename  = 'pred-global-mean.csv'
train_filename = 'ratings-train.csv'
test_filename  = 'ratings-test.csv'

training_data  = util.load_train(train_filename)
test_queries   = util.load_test(test_filename)

# Compute the mean rating.
num_train = len(training_data)
mean_rating = float(sum(map(lambda x: x['rating'], training_data)))/num_train
print "The mean rating is %0.3f." % (mean_rating)

# Use the global mean to make predictions.
# Iterate over the test set and add a 'rating' dictionary element.
for query in test_queries:
    query['rating'] = mean_rating

# Write the prediction file.
util.write_predictions(test_queries, pred_filename)



