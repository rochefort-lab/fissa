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
        # should be a 3D array of shape (frame_number, x-coords, y-coords)
        self.expected = np.array(
            [
                [[1, 2, 3], [5, 6, 7], [8, 9, 10]],
                [[11, 12, 13], [15, 16, 17], [18, 19, 20]],
            ]
        )
        self.resources_dir = os.path.join(self.test_directory, 'resources', 'tiffs')
        self.datahandler = DataHandlerTifffile()

    def test_imsave_tiff(self):
        """
        Tiff generated from self.expected as

        >>> tifffile.imsave('test_imsave.tif', data)

        using tifffile.__version__ = 2021.4.8
        """
        # load from tif
        actual = self.datahandler.image2array(os.path.join(self.resources_dir, 'test_imsave.tif'))

        # assert equality
        self.assert_equal(actual, self.expected)

    def test_tiffwriter_tiff(self):
        """
        Tiff generated from self.expected as

        >>> with tifffile.TiffWriter('test_tiffwriter.tif') as tif:
        >>>     for i in range(data.shape[0]):
        >>>         tif.write(data[i, :, :], contiguous=True)

        using tifffile.__version__ = 2021.4.8
        """
        # load from tif
        actual = self.datahandler.image2array(os.path.join(self.resources_dir, 'test_tiffwriter.tif'))

        # assert equality
        self.assert_equal(actual, self.expected)

    def test_suite2p_tiff(self):
        """
        Tiff generated from self.expected as

        >>> with tifffile.TiffWriter('test_suite2p.tif') as tif:
        >>>     for frame in np.floor(data).astype(np.int16):
        >>>         tif.save(frame)

        using tifffile.__version__ = 2021.4.8

        As done by Suite2p:
        (https://github.com/MouseLand/suite2p/blob/4b6c3a95b53e5581dbab1feb26d67878db866068/suite2p/io/tiff.py#L59)
        """
        # load from tif
        actual = self.datahandler.image2array(os.path.join(self.resources_dir, 'test_suite2p.tif'))

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
        self.expected = np.array(
            [
                [[1, 2, 3], [5, 6, 7], [8, 9, 10]],
                [[11, 12, 13], [15, 16, 17], [18, 19, 20]],
            ],
            dtype=np.uint8,
        )
        # make tif
        imageio.imwrite('test.tif', self.expected)
        self.datahandler = DataHandlerPillow()

    def test_imageio_tiff(self):
        # load from tif
        actual = self.datahandler.image2array('test.tif')

        # assert equality
        self.assert_equal(np.asarray(actual), self.expected)

    def teardown_class(self):
        # remove tif
        os.remove('test.tif')


class TestRois2MasksPillow(TestRois2MasksTifffile):
    '''Tests for rois2masks using DataHandlerPillow.'''
    def setup_class(self):
        self.polys = [np.array([[39., 62.], [60., 45.], [48., 71.]]),
                      np.array([[72., 107.], [78., 130.], [100., 110.]])]
        self.expected = roitools.getmasks(self.polys, (176, 156))
        self.data = Image.fromarray(np.zeros((176, 156), dtype=np.uint8))
        self.datahandler = DataHandlerPillow()
