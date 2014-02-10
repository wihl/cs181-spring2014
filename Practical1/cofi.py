import numpy as np

cofiUsers = {}

def buildRatingMatrix(training_data):
    # create arrays of items and users
    items = {}
    c_items = 0
    users = {}
    c_users = 0
    for rating in training_data:
        item = rating['isbn']
        user = rating['user']
        if not item in items:
            items[item] = c_items
            c_items += 1
        if not user in users:
            users[user] = c_users
            c_users += 1
        # map userId to appropriate row in rating matrix
        cofiUsers[user] = c_users

    ratings = np.zeros((c_items,c_users))
    rating_exists = np.zeros((c_items, c_users))

    for row in training_data:
        item = row['isbn']        
        user = row['user']
        rating = row['rating']
        #print item, user, rating
        #print items[item], users[user]
        rating_exists[items[item],users[user]] = 1
        ratings[items[item],users[user]] = rating

    return ratings, rating_exists

def buildTheta(user_list):
    users = {}
    c_users = 0

    countries = {}
    c_countries = 0

    Theta = np.zeros((len(user_list),2)) # location code, age

    for row in user_list:
        user = row['user']
        # quantify location
        # TODO: take into account region, not just country
        location = row['location']
        country = location.split()[-1]
        if not country in countries:
            countries[country] = c_countries
            c_countries += 1
        age  = row['age']

        Theta[c_users,0] = c_countries
        Theta[c_users,1] = age
        c_users += 1
        # TODO: what do I do with the userId?

    return Theta
