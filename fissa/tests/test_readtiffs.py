'''
Tests for readtiffs.py
'''

import os.path
import csv
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
    resources/black_1x2x3.tif
        3 frames of black with width 1px and height 2px.
        convert -depth 8 -size 1x2 xc:#000 xc:#000 xc:#000 \
            black_1x2x3.tif
    resources/white_1x2x3.tif
        3 frames of white with width 1px and height 2px.
        convert -depth 8 -size 1x2 xc:#fff xc:#fff xc:#fff \
            white_1x2x3.tif
    resources/grey_1x1x5.tif
        A tiff stack (multipage tiff) containing 5 images with
        increasing shades of grey.
        convert -depth 8 -size 1x1 xc:#808080 xc:#828282 \
            xc:#848484 xc:#868686 xc:#888888 grey_1x1x5.tif
    '''
    uniform_datafile = os.path.join(RESOURCE_DIRECTORY, 'uniform_tiffs.csv')
    resources = csv.DictReader(open(uniform_datafile), delimiter=';')
    return resources


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
def test_frame_number(row):
    '''
    Tests the function get_frame_number

    Use the small set of test resource TIFF files to confirm the
    number of frames is as expected.
    '''
    expected = float(row['num_frames'])
    img = Image.open(os.path.join(RESOURCE_DIRECTORY, row['filename']))
    actual = readtiffs.get_frame_number(img)
    base_test.assert_equal(actual, expected)


@pytest.mark.parametrize('row', get_uniform_resources())
def test_get_mean_tiff_uniform(row):
    '''
    Tests the function get_mean_tiff against the uniform TIFF test
    images.
    '''
    # We expect to get a uniform image
    expected_colour = float(row['mean_color'])
    # NB: dim1 and dim2 might be the other way around to what you
    # expect here, but this is correct!
    expected_dim1 = float(row['height'])
    expected_dim2 = float(row['width'])
    expected = expected_colour * np.ones((expected_dim1, expected_dim2),
                                         dtype=np.uint8)
    # Take the mean of the image stack stack
    fname = os.path.join(RESOURCE_DIRECTORY, row['filename'])
    actual = readtiffs.get_mean_tiff(fname)
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
        unit8 image with spot colors in different shades of grey. The
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
        # There is only one frame, so it should be the same the image
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
            actual = readtiffs.getavg(img, box);
            # There is only one frame, so it should be the same the image
            self.assert_equal(actual, expected)
