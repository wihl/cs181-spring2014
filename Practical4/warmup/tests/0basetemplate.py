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

class FooTests(unittest.TestCase):

    def testFoo(self):
        self.failUnless(False)

def main():
    unittest.main()

if __name__ == '__main__':
    main()

