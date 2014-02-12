#
# David Wihl CSCI-E181, Spring 2014
#
# sample recommender modified heavily from 
# Programming Collective Intelligence by Toby Segaran. 
# Copyright 2007 Toby Segaran, 978-0-596-52932-1

from math import sqrt


def sim_pearson(prefs,p1,p2):
  '''
  Returns the Pearson correlation coefficient for p1 and p2
  '''
  # Get the list of mutually rated items
  si={}
  for item in prefs[p1]: 
    if item in prefs[p2]: si[item]=1

  # if they are no ratings in common, return 0
  if len(si)==0: return 0

  # Sum calculations
  n=len(si)
  print "n:", n
  
  # Sums of all the preferences
  sum1=sum([prefs[p1][it] for it in si])
  sum2=sum([prefs[p2][it] for it in si])
  print "sum1, sum2",sum1,sum2
  # Sums of the squares
  sum1Sq=sum([pow(prefs[p1][it],2) for it in si])
  sum2Sq=sum([pow(prefs[p2][it],2) for it in si])	
  print "sum1sq, sum2sq",sum1Sq,sum2Sq  
  # Sum of the products
  pSum=sum([prefs[p1][it]*prefs[p2][it] for it in si])
  print "psum", pSum
  # Calculate r (Pearson score)
  num=pSum-(sum1*sum2/n)
  print "num:", num
  den=sqrt((sum1Sq-pow(sum1,2)/n)*(sum2Sq-pow(sum2,2)/n))
  print "den:", den
  if den==0: return 0

  r=num/den

  return r


#Returns the best matches for person from the prefs dictionary
#Number of the results and similiraty function are optional params.
def topMatches(prefs,person,n=5,similarity=sim_pearson):
    scores = [(similarity(prefs,person,other),other)
                                for other in prefs if other != person]
    scores.sort()
    scores.reverse()
    return scores[0:n]



def getRecommendations(prefs,person,similarity=sim_pearson):
  '''
  Gets recommendations for a person by using a weighted average
  of every other user's rankings
  '''
  totals={}
  simSums={}
  for other in prefs:
    # don't compare me to myself
    if other==person: continue
    sim=similarity(prefs,person,other)

    # ignore scores of zero or lower
    if sim<=0: continue
  
    for item in prefs[other]:
      # only score movies I haven't seen yet
      if item not in prefs[person] or prefs[person][item]==0:
        # Similarity * Score
        totals.setdefault(item,0)
        totals[item]+=prefs[other][item]*sim
        # Sum of similarities
        simSums.setdefault(item,0)
        simSums[item]+=sim

  # Create the normalized list
  rankings = {}
  for item, total in totals.items():
      rankings[item] = total/simSums[item]
  #rankings=[(total/simSums[item],item) for item,total in totals.items()]

  # Return the sorted list
#  rankings.sort()
#  rankings.reverse()
  return rankings
