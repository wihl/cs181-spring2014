import numpy as py
from math import sqrt 


def vectorlen(user):
    total = 0.0
    total = sum(pow(user['ratings'][rating],2) for rating in user['ratings'])
    user['vectorlen'] = sqrt(total)

def topMatch(users):
    # populate vector length
    for user in users:
        vectorlen(users[user])

    for user1 in users:
        # compute the user's mean rating
        total = 0.0
        count = 0
        for r1 in users[user1]['ratings']:
            count += 1
            total += users[user1]['ratings'][r1]

        if count > 0:
            users[user1]['mean'] = float(total) / count
        else:
            users[user1]['mean'] = 0
 
        # print "user mean", user1, users[user1]['mean']

        # find closet other user
        users[user1]['closest_user'] = 0
        users[user1]['cosine'] = -1.0

        for user2 in users:
            dotproduct = 0.0
            if user2 == user1: continue

            for r1 in users[user1]['ratings']:
                if r1 in users[user2]['ratings']:
                    dotproduct += users[user1]['ratings'][r1] * users[user2]['ratings'][r1]
            if users[user1]['vectorlen'] > 0.000001 and users[user2]['vectorlen'] > 0.000001:
                cosine = dotproduct / (users[user1]['vectorlen'] * users[user2]['vectorlen'])
                if cosine > users[user1]['cosine']:
                    users[user1]['cosine'] = cosine
                    users[user1]['closest_user'] = user2

    '''
    print "user, rating count, vectorlen, closest user, cosine value"
    for user in users:
        print user, len(users[user]['ratings']), users[user]['vectorlen'], \
              users[user]['closest_user'], \
              users[user]['cosine']
    '''


def predict(users,user,items,item,global_mean):
    # has the user already rated this item?

    if user in users:
        if item in users[user]['ratings']:
            return users[user]['ratings'][item]

        # has his closest buddy rated this?
        closest = users[user]['closest_user']
        if item in users[closest]['ratings']:
            return users[closest]['ratings'][item]

    # does the user have a mean rating? If so, use that
    


    # then return the book's mean if it is non-zero
    if item in items:
        if 'mean' in item:
            item_mean = items.get(item['mean'],0)
            if item_mean != 0:
                return item_mean

    return global_mean

def meanPerItem(users):
    items = {}
    for user in users:
        for item in users[user]['ratings']:
            if not item in items:
                items[item] = { 'total': users[user]['ratings'][item], 'count':1 }
            else:
                items[item]['total'] += users[user]['ratings'][item]
                items[item]['count'] += 1

    global_total = 0.0
    global_count = 0
    for item in items:
        items[item]['mean'] = float(items[item]['total']) / items[item]['count']
        #print global_total, items[item]['mean']
        global_total += items[item]['mean']
        global_count += 1

    if global_count > 0:
        global_mean = float(global_total) / global_count
    else:
        global_mean = 0.0

    print "global mean: ", global_mean

    return items, global_mean
