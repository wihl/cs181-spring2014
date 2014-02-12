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

    print "user, rating count, vectorlen, closest user, cosine value"
    for user in users:
        print user, len(users[user]['ratings']), users[user]['vectorlen'], \
              users[user]['closest_user'], \
              users[user]['cosine']

def predict(user,item):
    return 4
