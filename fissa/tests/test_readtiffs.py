'''
Tests for readtiffs.py
'''

import os.path
import csv
import ast

import pytest
import numpy as np
from PIL import Image

import base_test
from .. import readtiffs


# Get a path to the resources folder from the test directory
RESOURCE_DIRECTORY = os.path.join(base_test.TEST_DIRECTORY, 'resources')


def get_uniform_resources():
    '''
    Fixture

    Get the test database from `tiff_num_frames.csv` and store it
    as a dictionary in `self.resource_num_frames`. This should have
    two fields, the first is `filename`, containing the filename of
    the test image, and the second is `num_frames`, containing the
    number of images in the tiff stack.

    The images should be as follows:
    resources/lgrey_4x5x1.tif
        A single tiff (no stack), created with the command:
        convert -depth 8 -size 4x5 xc:#aaa lgrey_4x5x1.tif
    resources/lgrey_4x5x1_16bit.tif
        A single tiff (no stack), created with the command:
        convert -depth 16 -size 4x5 xc:#aaa lgrey_4x5x1_16bit.tif
    resources/black_1x2x3.tif
        3 frames of black with width 1px and height 2px.
        convert -depth 8 -size 1x2 xc:#000 xc:#000 xc:#000 black_1x2x3.tif
    resources/white_1x2x3.tif
        3 frames of white with width 1px and height 2px.
        convert -depth 8 -size 1x2 xc:#fff xc:#fff xc:#fff white_1x2x3.tif
    resources/white_1x2x3_1bit.tif
        3 frames of white with width 1px and height 2px, with single
        pixel depth (binary image).
        convert -depth 1 -size 1x2 xc:#fff xc:#fff xc:#fff white_1x2x3_1bit.tif
    resources/grey_1x1x5.tif
        A tiff stack (multipage tiff) containing 5 images with
        increasing shades of grey.
        convert -depth 8 -size 1x1 xc:#808080 xc:#828282 \
            xc:#848484 xc:#868686 xc:#888888 grey_1x1x5.tif
        ... or ...
        convert -depth 8 -size 1x1 xc:'rgb(128,128,128)' \
            xc:'rgb(130,130,130)' xc:'rgb(132,132,132)' xc:'rgb(134,134,134)' \
            xc:'rgb(136,136,136)' grey_1x1x5.tif
    resources/orange_2x1x4.tif
        A tiff stack (multipage tiff) containing 5 images with
        different shades of orange.
        convert -depth 8 -size 2x1 xc:'rgb(200,100,45)' xc:'rgb(210,112,80)' \
            xc:'rgb(220,123,21)' xc:'rgb(234,134,13)' orange_2x1x4.tif
    '''
    uniform_datafile = os.path.join(RESOURCE_DIRECTORY, 'uniform_tiffs.csv')
    resources = csv.DictReader(open(uniform_datafile), delimiter=';')
    return resources


class TestImage2Array(base_test.BaseTestCase):
    '''
    Tests image2array()

    Note that this function is also tested extensively by the tests of
    other, higher-level, functions.
    '''
    def test_8bit(self):
        expected = np.arange(8, dtype=np.uint8).reshape((2, 4))
        img = Image.fromarray(expected, 'L')
        actual = readtiffs.image2array(img, bit_depth=8)
        self.assert_equal(actual, expected)

    def test_8bit_uint8(self):
        expected = np.arange(8, dtype=np.uint8).reshape((2, 4))
        img = Image.fromarray(expected, 'L')
        actual = readtiffs.image2array(img, bit_depth=np.uint8)
        self.assert_equal(actual, expected)

    def test_8bit_implicit(self):
        expected = np.arange(8, dtype=np.uint8).reshape((2, 4))
        img = Image.fromarray(expected, 'L')
        actual = readtiffs.image2array(img)
        self.assert_equal(actual, expected)

    def test_16bit(self):
        expected = np.arange(257, 265, dtype=np.uint16).reshape((2, 4))
        img = Image.fromarray(expected, 'I;16')
        actual = readtiffs.image2array(img, bit_depth=16)
        self.assert_equal(actual, expected)

    def test_32bit(self):
        expected = np.arange(65536, 65544, dtype=np.uint32).reshape((2, 4))
        img = Image.fromarray(expected, 'I')
        actual = readtiffs.image2array(img, bit_depth=32)
        self.assert_equal(actual, expected)

    def test_1bit(self):
        expected = np.array([[0, 1, 1], [1, 0, 1]], dtype=bool)
        array = 255 * np.asarray(expected, dtype=np.uint8)
        img = Image.fromarray(array, 'L').convert('1', dither=None)
        actual = readtiffs.image2array(img, bit_depth=1)
        self.assert_equal(actual, expected)

    @pytest.mark.xfail
    def test_1bit_implicit(self):
        expected = np.array([[0, 1, 1], [1, 0, 1]], dtype=bool)
        array = 255 * np.asarray(expected, dtype=np.uint8)
        img = Image.fromarray(array, 'L').convert('1', dither=None)
        actual = readtiffs.image2array(img)
        self.assert_equal(actual, expected)

    def test_8bit_color(self):
        array = np.array([[(1, 101, 201), (2, 102, 202)]], np.uint8)
        img = Image.fromarray(array, 'RGB')
        # R channel
        self.assert_equal(readtiffs.image2array(img, band=0), array[:, :, 0])
        # G channel
        self.assert_equal(readtiffs.image2array(img, band=1), array[:, :, 1])
        # B channel
        self.assert_equal(readtiffs.image2array(img, band=2), array[:, :, 2])

    def test_raise_too_many_bands(self):
        img = Image.new('L', (4, 5), color=0)
        with self.assertRaises(ValueError):
            readtiffs.image2array(img, band=2)

    def test_raise_unspecified_band(self):
        img = Image.new('RGB', (4, 5), color=1)
        with self.assertRaises(ValueError):
            readtiffs.image2array(img, band=None)


class TestGetBox(base_test.BaseTestCase):
    '''
    Tests getbox()
    '''
    def test_getbox_even_even(self):
        # This box is not quite centered, it has to be offset to give the
        # desired output shape from the cut. After cut, x-axis will be:
        # length 4: [8 9 ,10, 11]
        self.assertEqual(readtiffs.getbox((20, 10), 2), (8, 18, 12, 22))

    def test_getbox_even_odd(self):
        # This box is not quite centered, it has to be offset to give the
        # desired output shape from the cut. After cut, x-axis will be:
        # length 6: [7 8 9 ,10, 11 12]
        self.assertEqual(readtiffs.getbox((20, 10), 3), (7, 17, 13, 23))

    def test_getbox_odd_even(self):
        # This box is not quite centered, it has to be offset to give the
        # desired output shape from the cut. After cut, x-axis will be:
        # length 4: [9 10 ,11, 12]
        self.assertEqual(readtiffs.getbox((21, 11), 2), (9, 19, 13, 23))

    def test_getbox_odd_odd(self):
        # This box is not quite centered, it has to be offset to give the
        # desired output shape from the cut. After cut, x-axis will be:
        # length 6: [8 9 10 ,11, 12 13]
        self.assertEqual(readtiffs.getbox((21, 11), 3), (8, 18, 14, 24))

    def test_getbox_int_half(self):
        # This box is centered. After cut, x-axis will be:
        # length 5: [9 10 ,11, 12 13]
        self.assertEqual(readtiffs.getbox((21, 11), 2.5), (9, 19, 14, 24))
        # length 5: [8 9 ,10, 11 12]
        self.assertEqual(readtiffs.getbox((20, 10), 2.5), (8, 18, 13, 23))

    def test_getbox_half_int(self):
        # This box is centered. After cut, x-axis will be:
        # length 4: [9 10 , 11 12]
        self.assertEqual(readtiffs.getbox((20.5, 10.5), 2), (9, 19, 13, 23))
        # length 6: [8 9 10 , 11 12 13]
        self.assertEqual(readtiffs.getbox((20.5, 10.5), 3), (8, 18, 14, 24))

    def test_getbox_half_half(self):
        # This box is not quite centered, it has to be offset to give the
        # desired output shape from the cut. After cut, x-axis will be:
        # length 5: [8 9 10 , 11 12]
        self.assertEqual(readtiffs.getbox((20.5, 10.5), 2.5), (8, 18, 13, 23))

    def test_getbox_float_floatdown(self):
        # This box cannot be centered exactly, but we get as close as we can.
        # After cut, x-axis will be:
        # length round(5.4)=5: [8 9 10, 11 12]
        # After cut, y-axis will be:
        # length round(5.4)=5: [19 20 ,21 22 23]
        self.assertEqual(readtiffs.getbox((20.8, 10.2), 2.7), (8, 19, 13, 24))

    def test_getbox_float_floatup(self):
        # This box cannot be centered exactly, but we get as close as we can.
        # After cut, x-axis will be:
        # length round(4.2)=4: [9 10 ,11 12]
        # After cut, y-axis will be:
        # length round(4.2)=4: [19 20, 21 22]
        self.assertEqual(readtiffs.getbox((20.1, 10.75), 2.1), (9, 19, 13, 23))


@pytest.mark.parametrize('row', get_uniform_resources())
def test_uniform__frame_number(row):
    '''
    Tests the function get_frame_number

    Use the small set of test resource TIFF files to confirm the
    number of frames is as expected.
    '''
    print(row)  # To debug any errors

    expected = float(row['num_frames'])
    img = Image.open(os.path.join(RESOURCE_DIRECTORY, row['filename']))
    actual = readtiffs.get_frame_number(img)
    base_test.assert_equal(actual, expected)


@pytest.mark.parametrize('row', get_uniform_resources())
def test_uniform__get_mean_tiff(row):
    '''
    Tests the function get_mean_tiff against the uniform TIFF test
    images.
    '''
    print(row)  # To debug any errors

    # We expect to get a uniform image
    expected_colour = float(row['mean_color'])
    # NB: dim1 and dim2 might be the other way around to what you
    # expect here, but this is correct!
    expected_dim1 = float(row['height'])
    expected_dim2 = float(row['width'])

    expected = expected_colour * np.ones((expected_dim1, expected_dim2),
                                         np.float64)

    # Take the mean of the image stack
    fname = os.path.join(RESOURCE_DIRECTORY, row['filename'])
    kwargs = {}
    if row['bit_depth'] is not '':
        kwargs['bit_depth'] = float(row['bit_depth'])
    if row['band'] is not '':
        kwargs['band'] = int(float(row['band']))
    actual = readtiffs.get_mean_tiff(fname, **kwargs)

    # Check they match
    base_test.assert_equal(actual, expected)


@pytest.mark.parametrize('row', get_uniform_resources())
@pytest.mark.parametrize('frame_indices', [None, [0]])
@pytest.mark.parametrize('fullbox', [False, True])
def test_uniform__getavg(row, frame_indices, fullbox):
    '''
    Tests the function getavg against the uniform TIFF test
    images.
    '''
    print(row)  # To debug any errors

    if fullbox:
        # Take the whole frame
        box = (0, 0, int(float(row['width'])), int(float(row['height'])))
    else:
        # Just take the first pixel
        box = (0, 0, 1, 1)

    if frame_indices is None:
        # Average over all frames
        expected_colour = float(row['mean_color'])
    else:
        # Average over a selection of frames
        trace = np.asarray(ast.literal_eval(row['trace']))
        expected_colour = np.mean(trace[frame_indices])
    # We expect to get a uniform image, the same size as the box
    expected = expected_colour * np.ones((box[3], box[2]))

    # Take the mean of the image stack within this box
    fname = os.path.join(RESOURCE_DIRECTORY, row['filename'])
    img = Image.open(fname)
    kwargs = {}
    if row['bit_depth'] is not '':
        kwargs['bit_depth'] = float(row['bit_depth'])
    if row['band'] is not '':
        kwargs['band'] = int(float(row['band']))
    actual = readtiffs.getavg(img, box, frame_indices, **kwargs)

    # Check they match
    base_test.assert_equal(actual, expected)


@pytest.mark.parametrize('row', get_uniform_resources())
def test_uniform__extract_from_single_tiff(row):
    '''
    Tests the function getavg against the uniform TIFF test
    images.
    '''
    print(row)  # To debug any errors

    # Get attributes of the TIFF needed for setting up the output
    shape = (float(row['height']), float(row['width']))
    trace = np.asarray(ast.literal_eval(row['trace']), dtype=np.float64)

    # Set up a few masks we can use
    mask_all = np.ones(shape, bool)
    mask_top_left = np.zeros(shape, bool)
    mask_top_left[0, 0] = True
    mask_bottom_right = np.zeros(shape, bool)
    mask_bottom_right[-1, -1] = True

    # Make masksets dictionary from the simple masks
    masksets = {
        'a': [mask_all],
        'b': [mask_top_left, mask_bottom_right]
    }

    # Assemble expected output, given the masksets and the trace
    expected = {}
    for key, value in masksets.items():
        expected[key] = np.tile(trace, (len(value), 1))

    # Get attributes needed to put into the function
    fname = os.path.join(RESOURCE_DIRECTORY, row['filename'])
    kwargs = {}
    if row['bit_depth'] is not '':
        kwargs['bit_depth'] = float(row['bit_depth'])
    if row['band'] is not '':
        kwargs['band'] = int(float(row['band']))
    # Compute the actual output
    actual = readtiffs.extract_from_single_tiff(fname, masksets, **kwargs)

    # Check they match
    base_test.assert_equal(actual, expected)


class Test2dGreyPoints(base_test.BaseTestCase):
    '''
    Tests as many readtiff functions as possible on a 2-dimensional
    unsigned 8-bit integer image resource.
    '''
    @classmethod
    def setUpClass(self):
        '''
        Load up the `points_grey_2d.tif` resource, which contains a 2D
        uint8 image with spot colors in different shades of grey. The
        contents are verbosely contained in the corresponding text
        file.
        '''
        self.filename = os.path.join(RESOURCE_DIRECTORY, 'points_grey_2d.tif')
        array_filename = os.path.join(RESOURCE_DIRECTORY, 'points_grey_2d.txt')
        # Load the expected values from text file
        self.expected_array = np.loadtxt(array_filename, dtype=np.uint8)

    def test_get_frame_number(self):
        expected = 1
        actual = readtiffs.get_frame_number(Image.open(self.filename))
        self.assertEqual(actual, expected)

    def test_tiff2array(self):
        # When we get a whole-imagestack version, we have an extra dimension
        # for frame indices
        expected = np.expand_dims(self.expected_array, axis=-1)
        actual = readtiffs.tiff2array(self.filename)
        self.assert_equal(actual, expected)

    def test_get_mean_tiff(self):
        # There is only one frame, so it should be the same as the image
        self.assert_equal(readtiffs.get_mean_tiff(self.filename),
                          self.expected_array)

    def test_getavg(self):
        img = Image.open(self.filename)
        boxes = [
            (0, 0, 1, 1),
            (1, 1, 2, 2),
            (0, 0, 2, 2),
            (0, 0, self.expected_array.shape[1], self.expected_array.shape[0]),
            (0, 1, 1, 3)]
        for box in boxes:
            print(box)
            expected = self.expected_array[box[1]:box[3], box[0]:box[2]]
            actual = readtiffs.getavg(img, box)
            # There is only one frame, so it should be the same the image
            self.assert_equal(actual, expected)
