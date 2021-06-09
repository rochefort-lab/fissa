"""
Tests for extraction.py
"""
import os

import numpy as np
import tifffile
import imageio
from PIL import Image

from .base_test import BaseTestCase
from .. import extraction
from .. import roitools


class Image2ArrayBase():
    """Tests for image2array."""

    # should be a 3D array of shape (frame_number, x-coords, y-coords)

    expected = np.array(
        [
            [[1, 2, 3], [5, 6, 7], [8, 9, 10]],
            [[11, 12, 13], [15, 16, 17], [18, 19, 20]],
        ]
    )

    def setup_class(self):
        self.resources_dir = os.path.join(self.test_directory, 'resources', 'tiffs')
        self.datahandler = None

    def test_imsave_tiff(self):
        """
        Test loading of image saved with tifffile.imsave.

        Tiff resource generated from self.expected using command

        >>> tifffile.imsave('test_imsave.tif', data)

        with tifffile version 2021.4.8.
        """
        actual = self.datahandler.image2array(os.path.join(self.resources_dir, 'test_imsave.tif'))
        self.assert_equal(actual, self.expected)

    def test_tiffwriter_tiff(self):
        """
        Test loading of image saved with tifffile.TiffWriter().write().

        Tiff resource generated from self.expected using command

        >>> with tifffile.TiffWriter('test_tiffwriter.tif') as tif:
        >>>     for i in range(data.shape[0]):
        >>>         tif.write(data[i, :, :], contiguous=True)

        with tifffile version 2021.4.8.
        """
        actual = self.datahandler.image2array(os.path.join(self.resources_dir, 'test_tiffwriter.tif'))
        self.assert_equal(actual, self.expected)

    def test_suite2p_tiff(self):
        """
        Test loading of image saved with tifffile.TiffWriter().save().

        Tiff resource generated from self.expected using command

        >>> with tifffile.TiffWriter('test_suite2p.tif') as tif:
        >>>     for frame in np.floor(data).astype(np.int16):
        >>>         tif.save(frame)

        with tifffile version 2021.4.8.

        This is the saving method used by Suite2p:
        https://github.com/MouseLand/suite2p/blob/4b6c3a95b53e5581dbab1feb26d67878db866068/suite2p/io/tiff.py#L59
        """
        actual = self.datahandler.image2array(os.path.join(self.resources_dir, 'test_suite2p.tif'))
        self.assert_equal(actual, self.expected)

    def test_array(self):
        actual = self.datahandler.image2array(self.expected)
        self.assert_equal(actual, self.expected)


class TestImage2ArrayTifffile(BaseTestCase, Image2ArrayBase):
    """Tests for image2array using DataHandlerTifffile."""
    def setup_class(self, *args, **kwargs):
        super(TestImage2ArrayTifffile, self).setup_class(self, *args, **kwargs)
        self.datahandler = extraction.DataHandlerTifffile()


class TestImage2ArrayPillow(BaseTestCase):
    """Tests for image2array using DataHandlerPillow."""
    def setup_class(self):
        self.expected = np.array(
            [
                [[1, 2, 3], [5, 6, 7], [8, 9, 10]],
                [[11, 12, 13], [15, 16, 17], [18, 19, 20]],
            ],
            dtype=np.uint8,
        )
        imageio.imwrite('test.tif', self.expected)
        self.datahandler = extraction.DataHandlerPillow()

    def test_imageio_tiff(self):
        actual = self.datahandler.image2array('test.tif')
        self.assert_equal(np.asarray(actual), self.expected)

    def teardown_class(self):
        # remove tif
        os.remove('test.tif')


class Rois2MasksBase():
    """Tests for rois2masks using DataHandlerTifffile."""

    polys = [
        np.array([[39., 62.], [60., 45.], [48., 71.]]),
        np.array([[72., 107.], [78., 130.], [100., 110.]]),
    ]

    def setup_class(self):
        self.expected = roitools.getmasks(self.polys, (176, 156))
        self.data = np.zeros((1, 176, 156))
        self.datahandler = None

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


class TestRois2MasksTifffile(BaseTestCase, Rois2MasksBase):
    """Tests for rois2masks using DataHandlerTifffile."""

    def setup_class(self, *args, **kwargs):
        super(TestRois2MasksTifffile, self).setup_class(self, *args, **kwargs)
        self.datahandler = extraction.DataHandlerTifffile()


class TestRois2MasksPillow(BaseTestCase, Rois2MasksBase):
    '''Tests for rois2masks using DataHandlerPillow.'''
    def setup_class(self, *args, **kwargs):
        super(TestRois2MasksPillow, self).setup_class(self, *args, **kwargs)
        self.data = Image.fromarray(self.data.reshape(self.data.shape[-2:]).astype(np.uint8))
        self.datahandler = extraction.DataHandlerPillow()
