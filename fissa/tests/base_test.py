"""
Objects shared by all the test cases
"""
import unittest

import numpy as np

from numpy.testing import assert_almost_equal
from numpy.testing import assert_array_equal
from numpy.testing import assert_allclose

class BaseTestCase(unittest.TestCase):
    """
    Superclass for all the FISSA test cases
    """
    def __init__(self, *args, **kw):
        """Add test for numpy type"""
        # super(self).__init__(*args, **kw) # Only works on Python3
        super(BaseTestCase, self).__init__(*args, **kw) # Works on Python2
        self.addTypeEqualityFunc(np.ndarray, self.assert_allclose)

    def assert_almost_equal(self, *args, **kwargs):
        return assert_almost_equal(*args, **kwargs)

    def assert_array_equal(self, *args, **kwargs):
        return assert_array_equal(*args, **kwargs)

    def assert_allclose(self, *args, **kwargs):
        return assert_allclose(*args, **kwargs)
