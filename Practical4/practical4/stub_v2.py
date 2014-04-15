import numpy.random as npr
import sys
import numpy

from SwingyMonkey import SwingyMonkey


#calculated mins and maxes values for 5000 iterations. will be used to define bins
seq_tree_min={'bot': 11, 'top': 211, 'dist': -115}
seq_tree_max={'bot': 140, 'top': 340, 'dist': 310}
seq_monkey_min={'vel': -47, 'bot': -44, 'top': 12}
seq_monkey_max={'vel': 18, 'bot': 364, 'top': 420}

learning_rate=0.1
discount_factor=0.2
state_space_dict={}
nbins=10

class Learner:

    def __init__(self):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None

    def reset(self):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None



    def gen_bins(self, state, numbins):
         tree_bot=numpy.linspace(seq_tree_min['bot'], seq_tree_max['bot'], numbins)
         tree_top=numpy.linspace(seq_tree_min['top'], seq_tree_max['top'],numbins)
         tree_dist=numpy.linspace(seq_tree_min['dist'], seq_tree_max['dist'], numbins)
         mon_vel=numpy.linspace(seq_monkey_min['vel'], seq_monkey_max['vel'], numbins)
         mon_bot=numpy.linspace(seq_monkey_min['bot'], seq_monkey_max['bot'], numbins)
         mon_top=numpy.linspace(seq_monkey_min['top'], seq_monkey_max['top'], numbins)


        #find indices for each variable for a given state
         tree_bot_bin=numpy.digitize([state['tree']['bot']], tree_bot)
         tree_top_bin=numpy.digitize([state['tree']['top']], tree_top)
         tree_dist_bin=numpy.digitize([state['tree']['dist']], tree_dist)

         mon_vel_bin=numpy.digitize([state['monkey']['vel']], mon_vel)
         mon_bot_bin=numpy.digitize([state['monkey']['bot']], mon_bot)
         mon_top_bin=numpy.digitize([state['monkey']['top']], mon_top)

         return tuple([int(tree_bot_bin), int(tree_top_bin), int(tree_dist_bin), int(mon_vel_bin), int(mon_bot_bin), int(mon_top_bin)])
    

    def action_callback(self, state):
        '''Implement this function to learn things and take actions.
        Return 0 if you don't want to jump and 1 if you do.'''

        #keep track of all visited states and resultant outcomes

        #if this is the first run
        if self.last_state== None:
        # print state
        # print self.last_action
        # print tuple([learner.gen_bins(self.last_state, nbins), self.last_action])
            qval=0
            # qval= state_space_dict.get(tuple([learner.gen_bins(self.last_state, nbins), self.last_action]),0) +  self.last_reward + learning_rate*( discount_factor*max(state_space_dict.get(tuple([gen_bins(self.last_state, nbins), 0]), 0), state_space_dict.get(tuple([gen_bins(self.last_state, nbins), 1]), 0)) - state_space_dict[[gen_bins(self.last_state, nbins), self.last_action]])
            new_action=npr.rand() < 0.1

        else:
            #implement Q learning
            qval= state_space_dict.get(tuple([learner.gen_bins(self.last_state, nbins), self.last_action]),0) +  self.last_reward + learning_rate*( discount_factor*max(state_space_dict.get(tuple([learner.gen_bins(self.last_state, nbins), 0]), 0), state_space_dict.get(tuple([learner.gen_bins(self.last_state, nbins), 1]), 0)) - state_space_dict.get(tuple([learner.gen_bins(self.last_state, nbins), self.last_action])))
            #print [learner.gen_bins(self.last_state, 10), self.last_action]
            state_space_dict[tuple([learner.gen_bins(self.last_state, nbins), self.last_action])]=qval

    #        print state

            # You might do some learning here based on the current state and the last state.

            # You'll need to take an action, too, and return it.
            # Return 0 to swing and 1 to jump.
            two_states=(state_space_dict.get(tuple([gen_bins(self.last_state, nbins), 0]), 0), state_space_dict.get(tuple([gen_bins(self.last_state, nbins), 1]), 0))
            new_action = states.index(max(two_states))
        new_state  = state

        self.last_action = new_action
        self.last_state  = new_state

        return self.last_action

    def reward_callback(self, reward):
        '''This gets called so you can see what reward you get.'''

        self.last_reward = reward


iters = 100
learner = Learner()
scorelist=[]

for ii in xrange(iters):

    # Make a new monkey object.
    swing = SwingyMonkey(sound=False,            # Don't play sounds.
                         text="Epoch %d" % (ii), # Display the epoch on screen.
                         tick_length=1,          # Make game ticks super fast.
                         action_callback=learner.action_callback,
                         reward_callback=learner.reward_callback)

    # Loop until you hit something.
    while swing.game_loop():
        pass

    #store all values for mins and max  calcs -- only need to run once to get values for the find_state_bounds function which saves these values    
    # scorelist.append(swing.get_state())

    #print swing.get_state()
    # Reset the state of the learner.
    learner.reset()

#calculate avg score for this approach
#print numpy.average(scorelist)




#def find_state_bounds():

    # print scorelist
    # seq_tree = [x['tree'] for x in scorelist]
    # seq_tree_min= min(seq_tree)
    # seq_tree_max= max(seq_tree)

    # seq_monkey = [x['monkey'] for x in scorelist]
    # seq_monkey_min= min(seq_monkey)
    # seq_monkey_max= max(seq_monkey)

    # print seq_tree_min
    # print seq_tree_max
    # print seq_monkey_min
    # print seq_monkey_max

#    return seq_tree_min, seq_tree_max, seq_monkey_min, seq_monkey_max

