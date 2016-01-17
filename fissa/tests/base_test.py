'''
Provides a general testing class which inherits from unittest.TestCase
and also provides the numpy testing functions.
'''

import unittest
import os.path
from inspect import getsourcefile

import numpy as np
from numpy.testing import (assert_almost_equal,
                           assert_array_equal,
                           assert_allclose,
                           assert_equal)


class BaseTestCase(unittest.TestCase):
    '''
    Superclass for all the FISSA test cases
    '''

    test_directory = os.path.dirname(os.path.abspath(getsourcefile(lambda: 0)))

    def __init__(self, *args, **kw):
        '''Add test for numpy type'''
        # super(self).__init__(*args, **kw)  # Only works on Python3
        super(BaseTestCase, self).__init__(*args, **kw)  # Works on Python2
        self.addTypeEqualityFunc(np.ndarray, self.assert_allclose)

    def assert_almost_equal(self, *args, **kwargs):
        return assert_almost_equal(*args, **kwargs)

    def assert_array_equal(self, *args, **kwargs):
        return assert_array_equal(*args, **kwargs)

    def assert_allclose(self, *args, **kwargs):
        return assert_allclose(*args, **kwargs)

    def assert_equal(self, *args, **kwargs):
        return assert_equal(*args, **kwargs)
