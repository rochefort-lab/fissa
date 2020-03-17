'''
Tests for datahandler.py
'''
import os

import numpy as np
import tifffile

from .base_test import BaseTestCase
from .. import datahandler


class TestImage2Array(BaseTestCase):
    ''' Tests for image2array.'''
    def setup_class(self):
        self.expected = np.array([[1, 2, 3], [5, 6, 7], [8, 9, 10]])

    def test_actual_tiff(self):
        # make tif
        tifffile.imsave('test.tif', self.expected)

        # load from tif
        actual = datahandler.image2array('test.tif')

        # remove tif
        os.remove('test.tif')

        # assert equality
        self.assert_equal(actual, self.expected)

    def test_array(self):
        # load from array
        actual = datahandler.image2array(self.expected)
        # assert equality
        self.assert_equal(actual, self.expected)
