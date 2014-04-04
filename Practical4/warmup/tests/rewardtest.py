'''
  Use this as a template to create future unit tests

  Feel free to make as many classes as necessary. All classes and methods will be invoked.
'''

# the following is required because Skimbox's modules are in a parent folder (currently) and
# will therefore not appear in the PYTHONPATH
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 

import unittest
import lawn as lawn

class RewardTests(unittest.TestCase):

    def testRewards(self):
        self.assertEqual(lawn.reward(0), 0, 'reward 0 should be 0')
        self.assertEqual(lawn.reward(101), 1, 'reward 101 should be 1')
        self.assertEqual(lawn.reward(102), -1, 'reward 102 should be -1')

def main():
    unittest.main()

if __name__ == '__main__':
    main()

