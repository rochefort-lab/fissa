"""
Tests for extraction.py
"""

from __future__ import division

import os

import numpy as np
import tifffile
import imageio
from PIL import Image
import pytest

from . import base_test
from .base_test import BaseTestCase
from .. import extraction
from .. import roitools


RESOURCES_DIR = os.path.join(base_test.TEST_DIRECTORY, 'resources', 'tiffs')


def get_dtyped_expected(expected, dtype):
    expected = np.copy(expected)
    if "uint" in str(dtype):
        expected = np.abs(expected)
    if "float" in str(dtype):
        expected = expected / 10
    return expected.astype(dtype)


@pytest.mark.parametrize(
    "dtype",
    ["uint8", "uint16", "uint64", "int16", "int64", "float16", "float32", "float64"],
)
@pytest.mark.parametrize("datahandler", [extraction.DataHandlerTifffile])
def test_single_frame_3d(dtype, datahandler):
    expected = np.array([[[-11, 12], [14, 15], [17, 18]]])
    expected = get_dtyped_expected(expected, dtype)
    fname = os.path.join(
        RESOURCES_DIR,
        "imageio.imwrite_{}.tif".format(dtype)
    )
    actual = datahandler.image2array(fname)
    base_test.assert_equal(actual, expected)


@pytest.mark.parametrize(
    "dtype",
    [
        "uint8",
        "uint16",
        pytest.param("uint64", marks=pytest.mark.xfail(reason="not supported")),
        "int16",
        pytest.param("int64", marks=pytest.mark.xfail(reason="not supported")),
        pytest.param("float16", marks=pytest.mark.xfail(reason="not supported")),
        "float32",
        pytest.param("float64", marks=pytest.mark.xfail(reason="not supported")),
    ],
)
@pytest.mark.parametrize("datahandler", [extraction.DataHandlerPillow])
def test_single_frame_2d(dtype, datahandler):
    expected = np.array([[-11, 12], [14, 15], [17, 18]])
    expected = get_dtyped_expected(expected, dtype)
    fname = os.path.join(
        RESOURCES_DIR,
        "imageio.imwrite_{}.tif".format(dtype)
    )
    actual = datahandler.image2array(fname)
    base_test.assert_equal(actual, expected)


def multiframe_image2array_tester(base_fname, dtype, datahandler):
    expected = np.array(
        [
            [[-11, 12], [14, 15], [17, 18]],
            [[21, 22], [24, 25], [27, 28]],
            [[31, 32], [34, 35], [37, 38]],
            [[41, 42], [44, 45], [47, 48]],
            [[51, 52], [54, 55], [57, 58]],
            [[61, 62], [64, 55], [67, 68]],
        ]
    )
    expected = get_dtyped_expected(expected, dtype)
    fname = os.path.join(
        RESOURCES_DIR,
        base_fname + "_{}.tif".format(dtype)
    )
    actual = datahandler.image2array(fname)
    base_test.assert_equal(actual, expected)


@pytest.mark.parametrize(
    "base_fname",
    [
        "tifffile.imsave",
        "tifffile.imsave.bigtiff",
        "TiffWriter.mixedA",
        "TiffWriter.mixedB",
        "TiffWriter.mixedC",
        "TiffWriter.save",
        "TiffWriter.write.contiguous",
        "TiffWriter.write.discontiguous",
    ],
)
@pytest.mark.parametrize(
    "dtype",
    ["uint8", "uint16", "uint64", "int16", "int64", "float16", "float32", "float64"],
)
@pytest.mark.parametrize("datahandler", [extraction.DataHandlerTifffile])
def test_multiframe_image2array(base_fname, dtype, datahandler):
    return multiframe_image2array_tester(
        base_fname=base_fname, dtype=dtype, datahandler=datahandler
    )


@pytest.mark.parametrize("dtype", ["uint8", "uint16", "float32"])
@pytest.mark.parametrize("datahandler", [extraction.DataHandlerTifffile])
def test_multiframe_image2array_imagejformat(dtype, datahandler):
    return multiframe_image2array_tester(
        base_fname="tifffile.imsave.imagej",
        dtype=dtype,
        datahandler=datahandler,
    )


@pytest.mark.parametrize(
    "base_fname",
    [
        "tifffile.imsave",
        "tifffile.imsave.bigtiff",
        "TiffWriter.save",
        "TiffWriter.write.contiguous",
        "TiffWriter.write.discontiguous",
    ],
)
@pytest.mark.parametrize("dtype", ["uint8"])
@pytest.mark.parametrize("shp", ["3,2,3,2", "2,1,3,3,2"])
@pytest.mark.parametrize("datahandler", [extraction.DataHandlerTifffile])
def test_multiframe_image2array_higherdim(base_fname, shp, dtype, datahandler):
    return multiframe_image2array_tester(
        base_fname=base_fname + "_" + shp,
        dtype=dtype,
        datahandler=datahandler,
    )


def multiframe_mean_tester(base_fname, dtype, datahandler):
    expected = np.array(
        [
            [[-11, 12], [14, 15], [17, 18]],
            [[21, 22], [24, 25], [27, 28]],
            [[31, 32], [34, 35], [37, 38]],
            [[41, 42], [44, 45], [47, 48]],
            [[51, 52], [54, 55], [57, 58]],
            [[61, 62], [64, 55], [67, 68]],
        ]
    )
    expected = get_dtyped_expected(expected, dtype)
    expected = np.mean(expected, axis=0)
    fname = os.path.join(
        RESOURCES_DIR,
        base_fname + "_{}.tif".format(dtype)
    )
    data = datahandler.image2array(fname)
    actual = datahandler.getmean(data)
    base_test.assert_allclose(actual, expected)


@pytest.mark.parametrize(
    "base_fname",
    [
        "tifffile.imsave",
        "tifffile.imsave.bigtiff",
        "TiffWriter.mixedA",
        "TiffWriter.mixedB",
        "TiffWriter.mixedC",
        "TiffWriter.save",
        "TiffWriter.write.contiguous",
        "TiffWriter.write.discontiguous",
    ],
)
@pytest.mark.parametrize(
    "dtype",
    ["uint8", "uint16", "uint64", "int16", "int64", "float16", "float32", "float64"],
)
@pytest.mark.parametrize("datahandler", [extraction.DataHandlerTifffile])
def test_multiframe_mean(base_fname, dtype, datahandler):
    return multiframe_mean_tester(
        base_fname=base_fname, dtype=dtype, datahandler=datahandler
    )


@pytest.mark.parametrize(
    "base_fname",
    [
        "tifffile.imsave",
        pytest.param("tifffile.imsave.bigtiff", marks=pytest.mark.xfail(reason="not supported")),
        "TiffWriter.mixedA",
        pytest.param("TiffWriter.mixedB", marks=pytest.mark.xfail(reason="not supported")),
        "TiffWriter.mixedC",
        "TiffWriter.save",
        "TiffWriter.write.contiguous",
        "TiffWriter.write.discontiguous",
    ],
)
@pytest.mark.parametrize(
    "dtype",
    [
        "uint8",
        "uint16",
        pytest.param("uint64", marks=pytest.mark.xfail(reason="not supported")),
        "int16",
        pytest.param("int64", marks=pytest.mark.xfail(reason="not supported")),
        pytest.param("float16", marks=pytest.mark.xfail(reason="not supported")),
        "float32",
        pytest.param("float64", marks=pytest.mark.xfail(reason="not supported")),
    ],
)
@pytest.mark.parametrize("datahandler", [extraction.DataHandlerPillow])
def test_multiframe_mean_pillow(base_fname, dtype, datahandler):
    return multiframe_mean_tester(
        base_fname=base_fname, dtype=dtype, datahandler=datahandler
    )


@pytest.mark.parametrize("dtype", ["uint8", "uint16", "float32"])
@pytest.mark.parametrize(
    "datahandler",
    [extraction.DataHandlerTifffile, extraction.DataHandlerPillow],
)
def test_multiframe_mean_imagejformat(dtype, datahandler):
    return multiframe_mean_tester(
        base_fname="tifffile.imsave.imagej",
        dtype=dtype,
        datahandler=datahandler,
    )


@pytest.mark.parametrize(
    "base_fname",
    [
        "tifffile.imsave",
        "tifffile.imsave.bigtiff",
        "TiffWriter.save",
        "TiffWriter.write.contiguous",
        "TiffWriter.write.discontiguous",
    ],
)
@pytest.mark.parametrize("dtype", ["uint8"])
@pytest.mark.parametrize("shp", ["3,2,3,2", "2,1,3,3,2"])
@pytest.mark.parametrize("datahandler", [extraction.DataHandlerTifffile])
def test_multiframe_mean_higherdim(base_fname, shp, dtype, datahandler):
    return multiframe_mean_tester(
        base_fname=base_fname + "_" + shp,
        dtype=dtype,
        datahandler=datahandler,
    )


@pytest.mark.parametrize(
    "base_fname",
    [
        "tifffile.imsave",
        pytest.param("tifffile.imsave.bigtiff", marks=pytest.mark.xfail(reason="not supported")),
        "TiffWriter.save",
        "TiffWriter.write.contiguous",
        "TiffWriter.write.discontiguous",
    ],
)
@pytest.mark.parametrize("dtype", ["uint8"])
@pytest.mark.parametrize(
    "shp",
    [
        "3,2,3,2",
        pytest.param("2,1,3,3,2", marks=pytest.mark.xfail(reason="looks like RGB")),
    ]
)
@pytest.mark.parametrize("datahandler", [extraction.DataHandlerPillow])
def test_multiframe_mean_higherdim_pillow(base_fname, shp, dtype, datahandler):
    return multiframe_mean_tester(
        base_fname=base_fname + "_" + shp,
        dtype=dtype,
        datahandler=datahandler,
    )


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
