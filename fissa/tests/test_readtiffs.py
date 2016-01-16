'''
Tests for readtiffs.py
'''

import os.path
import csv
from PIL import Image

from .base_test import BaseTestCase

from .. import readtiffs


class TestReadTiffs(BaseTestCase):

    @classmethod
    def setUpClass(self):
        '''
        Setup the test class

        Get the test database from `tiff_num_frames.csv` and store it
        as a dictionary in `self.resource_num_frames`. This should have two fields,
        the first is `filename`, containing the filename of the test
        image, and the second is `num_frames`, containing the number
        of images in the tiff stack.

        The images should be as follows:
            resources/orange_3x3x1.tif
                A single tiff with a single pixel coloured (255,128,0)
                and created with the command:
                convert -depth 8 -size 3x3 xc:#ff8800 orange_3x3x1.tif
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
        self.resource_directory = os.path.join(self.test_directory,
                                                'resources')
        datafile = os.path.join(self.resource_directory, 'tiff_num_frames.csv')
        self.resource_num_frames = list(csv.DictReader(open(datafile),
                                                       delimiter=';'))

    def test_get_frame_number(self):
        '''
        Tests the function get_frame_number

        Use the small set of test resource TIFF files to confirm the
        number of frames is as expected.
        '''
        for row in self.resource_num_frames:
            expected = float(row['num_frames'])
            fname = os.path.join(self.resource_directory, row['filename'])
            img = Image.open(fname)
            actual = readtiffs.get_frame_number(img)
            self.assertEqual(actual, expected)
