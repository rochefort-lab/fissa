'''
Tests for datahandler.py
'''
import os

import numpy as np
import tifffile
import imageio
from PIL import Image

from .base_test import BaseTestCase
from ..extraction import DataHandlerTifffile,  DataHandlerPillow
from .. import roitools

class TestImage2ArrayTifffile(BaseTestCase):
    ''' Tests for image2array.'''
    def setup_class(self):
        self.expected = np.array([[1, 2, 3], [5, 6, 7], [8, 9, 10]])
        self.datahandler = DataHandlerTifffile()

    def test_actual_tiff(self):
        # make tif
        tifffile.imsave('test.tif', self.expected)

        # load from tif
        actual = self.datahandler.image2array('test.tif')

        # remove tif
        os.remove('test.tif')

        # assert equality
        self.assert_equal(actual, self.expected)

    def test_array(self):
        # load from array
        actual = self.datahandler.image2array(self.expected)
        # assert equality
        self.assert_equal(actual, self.expected)


class TestRois2MasksTifffile(BaseTestCase):
    '''Tests for rois2masks.'''
    def setup_class(self):
        self.polys = [np.array([[39., 62.], [60., 45.], [48., 71.]]),
                      np.array([[72., 107.], [78., 130.], [100., 110.]])]
        self.expected = roitools.getmasks(self.polys, (176, 156))
        self.data = np.zeros((1, 176, 156))
        self.datahandler = DataHandlerTifffile()

    def test_imagej_zip(self):
        # load zip of rois
        ROI_loc = os.path.join(self.test_directory, 'resources', 'RoiSet.zip')
        actual = self.datahandler.rois2masks(ROI_loc, self.data)

        # assert equality
        self.assert_equal(actual, self.expected)

    def test_arrays(self):
        # load from array
        actual = self.datahandler.rois2masks(self.polys, self.data)
        # assert equality
        self.assert_equal(actual, self.expected)

    def test_transposed_polys(self):
        # load from array
        actual = self.datahandler.rois2masks([x.T for x in self.polys], self.data)
        # assert equality
        self.assert_equal(actual, self.expected)

    def test_masks(self):
        # load from masks
        actual = self.datahandler.rois2masks(self.expected, self.data)

        # assert equality
        self.assert_equal(actual, self.expected)

    def test_rois_not_list(self):
        # check that rois2masks fails when the rois are not a list
        with self.assertRaises(TypeError):
            self.datahandler.rois2masks({}, self.data)
        with self.assertRaises(TypeError):
            self.datahandler.rois2masks(self.polys[0], self.data)

    def test_polys_1d(self):
        # check that rois2masks fails when the polys are not 2d
        polys1d = [
            np.array([[39.,]]),
            np.array([[72.,]]),
        ]
        with self.assertRaises(ValueError):
            self.datahandler.rois2masks(polys1d, self.data)

    def test_polys_3d(self):
        # check that rois2masks fails when the polys are not 2d
        polys3d = [
            np.array([[39., 62., 0.], [60., 45., 0.], [48., 71., 0.]]),
            np.array([[72., 107., 0.], [78., 130., 0.], [100., 110., 0.]]),
        ]
        with self.assertRaises(ValueError):
            self.datahandler.rois2masks(polys3d, self.data)


class TestImage2ArrayPillow(BaseTestCase):
    ''' Tests for image2array.'''
    def setup_class(self):
        self.expected = np.array([[1, 2, 3], [5, 6, 7], [8, 9, 10]],
                                 dtype=np.uint8)
        # make tif
        imageio.imwrite('test.tif', self.expected)
        self.datahandler = DataHandlerPillow()

    def test_actual_tiff(self):
        # load from tif
        actual = self.datahandler.image2array('test.tif')

        # assert equality
        self.assert_equal(np.asarray(actual), self.expected)

    def teardown_class(self):
        # remove tif
        os.remove('test.tif')


class TestRois2MasksPillow(BaseTestCase):
    '''Tests for rois2masks.'''
    def setup_class(self):
        self.polys = [np.array([[39., 62.], [60., 45.], [48., 71.]]),
                      np.array([[72., 107.], [78., 130.], [100., 110.]])]
        self.expected = roitools.getmasks(self.polys, (176, 156))
        self.data = Image.fromarray(np.zeros((176, 156), dtype=np.uint8))
        self.datahandler = DataHandlerPillow()

    def test_imagej_zip(self):
        # load zip of rois
        ROI_loc = os.path.join(self.test_directory, 'resources', 'RoiSet.zip')
        actual = self.datahandler.rois2masks(ROI_loc, self.data)

        # assert equality
        self.assert_equal(actual, self.expected)

    def test_arrays(self):
        # load from array
        actual = self.datahandler.rois2masks(self.polys, self.data)
        # assert equality
        self.assert_equal(actual, self.expected)

    def test_transposed_polys(self):
        # load from array
        actual = self.datahandler.rois2masks([x.T for x in self.polys], self.data)
        # assert equality
        self.assert_equal(actual, self.expected)

    def test_masks(self):
        # load from masks
        actual = self.datahandler.rois2masks(self.expected, self.data)

        # assert equality
        self.assert_equal(actual, self.expected)

    def test_rois_not_list(self):
        # check that rois2masks fails when the rois are not a list
        with self.assertRaises(TypeError):
            self.datahandler.rois2masks({}, self.data)
        with self.assertRaises(TypeError):
            self.datahandler.rois2masks(self.polys[0], self.data)

    def test_polys_not_2d(self):
        # check that rois2masks fails when the polys are not 2d
        polys1d = [
            np.array([[39.,]]),
            np.array([[72.,]]),
        ]
        with self.assertRaises(ValueError):
            self.datahandler.rois2masks(polys1d, self.data)
        polys3d = [
            np.array([[39., 62., 0.], [60., 45., 0.], [48., 71., 0.]]),
            np.array([[72., 107., 0.], [78., 130., 0.], [100., 110., 0.]]),
        ]
        with self.assertRaises(ValueError):
            self.datahandler.rois2masks(polys3d, self.data)
