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
