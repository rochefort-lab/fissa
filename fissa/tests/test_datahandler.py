'''
Tests for datahandler.py
'''
import numpy as np
import tifffile
import os

from .base_test import BaseTestCase
from .. import datahandler
from .. import roitools


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


class TestRois2Masks(BaseTestCase):
    ''' Tests for rois2array
    '''
    def setup_class(self):
        self.polys = [np.array([[39., 62.], [60., 45.], [48., 71.]]),
                      np.array([[72., 107.], [78., 130.], [100., 110.]])]
        self.expected = roitools.getmasks(self.polys, (176, 156))
        self.data = np.zeros((1, 176, 156))

    def test_imagej_zip(self):
        # load zip of rois
        ROI_loc = 'fissa/tests/resources/RoiSet.zip'
        actual = datahandler.rois2masks(ROI_loc, self.data)

        # assert equality
        self.assert_equal(actual, self.expected)

    def test_arrays(self):
        # load from array
        actual = datahandler.rois2masks(self.polys, self.data)
        # assert equality
        self.assert_equal(actual, self.expected)

    def test_masks(self):
        # load from masks
        actual = datahandler.rois2masks(self.expected, self.data)

        # assert equality
        self.assert_equal(actual, self.expected)
