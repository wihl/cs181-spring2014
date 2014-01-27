import numpy as np
import util

# This makes predictions based on the mean rating for each book in the
# training data.  When there are no training data for a book, it
# defaults to the global mean.

pred_filename  = 'pred-book-mean.csv'
train_filename = 'ratings-train.csv'
test_filename  = 'ratings-test.csv'
book_filename  = 'books.csv'

training_data  = util.load_train(train_filename)
test_queries   = util.load_test(test_filename)
book_list      = util.load_books(book_filename)

# Compute the global mean rating for a fallback.
num_train = len(training_data)
mean_rating = float(sum(map(lambda x: x['rating'], training_data)))/num_train
print "The global mean rating is %0.3f." % (mean_rating)

# Turn the list of books into an ISBN-keyed dictionary.
# Store data for each book to keep track of the per-book average.
books = {}
for book in book_list:
    books[book['isbn']] = { 'total': 0, # For storing the total of ratings.
                            'count': 0, # For storing the number of ratings.
                            }
    
# Iterate over the training data to compute means.
for rating in training_data:
    books[rating['isbn']]['total'] += rating['rating']
    books[rating['isbn']]['count'] += 1

# Make predictions for each test query.
for query in test_queries:

    book = books[query['isbn']]

    if book['count'] == 0:
        # Perhaps we did not having any ratings in the training set.
        # In this case, make a global mean prediction.
        query['rating'] = mean_rating

    else:
        # Predict the average for this book.
        query['rating'] = float(book['total']) / book['count']

# Write the prediction file.
util.write_predictions(test_queries, pred_filename)
