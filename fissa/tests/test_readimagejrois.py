"""
Tests for readimagejrois.py.
"""

import json
import os
import sys
import unittest

import numpy as np

from .base_test import BaseTestCase
from .. import readimagejrois


class TestReadImageJRois(BaseTestCase):
    """readimagejrois testing class."""

    def __init__(self, *args, **kw):
        super(TestReadImageJRois, self).__init__(*args, **kw)

        self.data_dir = os.path.join(self.test_directory, "resources", "rois")

    def check_polygon(self, name):
        desired_arr = np.load(os.path.join(self.data_dir, name + ".npy"))
        desired = {"polygons": desired_arr}
        actual = readimagejrois.parse_roi_file(
            os.path.join(self.data_dir, name + ".roi")
        )
        self.assert_equal_dict_of_array(actual, desired)

    def check_mask(self, name):
        desired_arr = np.load(os.path.join(self.data_dir, name + ".npy"))
        desired = {"mask": desired_arr}
        actual = readimagejrois.parse_roi_file(
            os.path.join(self.data_dir, name + ".roi")
        )
        self.assert_equal_dict_of_array(actual, desired)

    def test_all(self):
        self.check_polygon("all")

    def test_brush(self):
        self.check_polygon("brush")

    def test_composite(self):
        # This is a retangle with a shape cut out of it, but still contiguous.
        # The type is displayed as "trace".
        self.check_polygon("composite-rectangle")

    def test_freehand(self):
        self.check_polygon("freehand")

    def test_freeline(self):
        self.check_polygon("freeline")

    @unittest.skipIf(sys.version_info < (3, 0), "multipoint rois only supported on Python 3")
    def test_multipoint(self):
        self.check_polygon("multipoint")

    @unittest.skipIf(sys.version_info >= (3, 0), "multipoint rois are supported on Python 3")
    def test_multipoint_py2_raises(self):
        with self.assertRaises(ValueError):
            self.check_polygon("multipoint")

    def test_polygon(self):
        self.check_polygon("polygon")

    def test_polyline(self):
        self.check_polygon("polyline")

    def test_rectangle(self):
        self.check_polygon("rectangle")

    @unittest.skipIf(sys.version_info < (3, 0), "Rotated rectangle rois only supported on Python 3")
    def test_rectangle_rotated(self):
        self.check_polygon("rectangle-rotated")

    @unittest.skipIf(sys.version_info >= (3, 0), "Rotated rectangle rois are supported on Python 3")
    def test_rectangle_rotated_py2_raises(self):
        with self.assertRaises(ValueError):
            self.check_polygon("rectangle-rotated")

    def test_rectangle_rounded(self):
        # We ignore the 'arc_size' parameter, and treat it like a regular
        # rectangle.
        self.check_polygon("rectangle-rounded")

    def test_oval(self):
        self.check_mask("oval")

    def test_oval_full(self):
        self.check_mask("oval-full")

    def test_ellipse(self):
        self.check_mask("ellipse")
