'''
Tests for datahandler_framebyframe.py
'''
import os

import numpy as np
import imageio
from PIL import Image

from .base_test import BaseTestCase
from .. import datahandler_framebyframe as datahandler


class TestImage2Array(BaseTestCase):
    ''' Tests for image2array.'''
    def setup_class(self):
        self.expected = np.array([[1, 2, 3], [5, 6, 7], [8, 9, 10]],
                                 dtype=np.uint8)
        # make tif
        imageio.imwrite('test.tif', self.expected)

    def test_actual_tiff(self):
        # load from tif
        actual = datahandler.image2array('test.tif')

        # assert equality
        self.assert_equal(np.asarray(actual), self.expected)

    def teardown_class(self):
        # remove tif
        os.remove('test.tif')
