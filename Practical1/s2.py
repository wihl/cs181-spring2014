import numpy as np
import util
import cofi as cofi

training_data = util.load_train('ratings-train.csv')
user_list      = util.load_users('users.csv')

users = {}

ratings, rating_exists = cofi.buildRatingMatrix(training_data)
Theta = cofi.buildTheta(user_list)
