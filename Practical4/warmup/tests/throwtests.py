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

class ThrowTests(unittest.TestCase):

    def testThrows(self):
        with self.assertRaises(AssertionError):
            lawn.throw(-1,-1)
            lawn.throw(100,100)
        self.assertEqual(lawn.throw(0,0), 0, '0,0 should be 0')
        self.assertEqual(lawn.throw(2,3), 8, '2,3 should be 8')
        self.assertEqual(lawn.throw(3,1), 16,'3,1 should be 16')
        self.assertEqual(lawn.throw(5,5), 0, '5,5 should be 0')

def main():
    unittest.main()

if __name__ == '__main__':
    main()

