'''
Provides a general testing class which inherits from unittest.TestCase
and also provides the numpy testing functions.
'''

import contextlib
import unittest
import os.path
from inspect import getsourcefile

import numpy as np
from numpy.testing import (assert_almost_equal,
                           assert_array_equal,
                           assert_allclose,
                           assert_equal)


# Check where the test directory is located, to be used when fetching
# test resource files
TEST_DIRECTORY = os.path.dirname(os.path.abspath(getsourcefile(lambda: 0)))


class BaseTestCase(unittest.TestCase):

    '''
    Superclass for all the FISSA test cases
    '''

    # Have the test directory as an attribute to the class as well as
    # a top-level variable
    test_directory = TEST_DIRECTORY

    def __init__(self, *args, **kw):
        '''Add test for numpy type'''
        # super(self).__init__(*args, **kw)  # Only works on Python3
        super(BaseTestCase, self).__init__(*args, **kw)  # Works on Python2
        self.addTypeEqualityFunc(np.ndarray, self.assert_allclose)

    @contextlib.contextmanager
    def subTest(self, *args, **kwargs):
        # For backwards compatability with Python < 3.4
        if hasattr(super(BaseTestCase, self), 'subTest'):
            yield super(BaseTestCase, self).subTest(*args, **kwargs)
        else:
            yield None

    def assert_almost_equal(self, *args, **kwargs):
        return assert_almost_equal(*args, **kwargs)

    def assert_array_equal(self, *args, **kwargs):
        return assert_array_equal(*args, **kwargs)

    def assert_allclose(self, *args, **kwargs):
        return assert_allclose(*args, **kwargs)

    def assert_equal(self, *args, **kwargs):
        return assert_equal(*args, **kwargs)

    def assert_equal_list_of_array_perm_inv(self, desired, actual):
        self.assert_equal(len(desired), len(actual))
        for desired_i in desired:
            n_matches = 0
            for actual_j in actual:
                if np.equal(actual_j, desired_i).all():
                    n_matches += 1
            self.assertTrue(n_matches >= 0)
