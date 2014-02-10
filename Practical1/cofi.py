from numpy import *

def buildRatingMatrix(training_data):
    #a = arange(130000 * 3000).reshape(130000,3000)
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

    ratings = arange(c_items * c_users).reshape(c_items,c_users)

    
    return
