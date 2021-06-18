"""
Tests for readimagejrois.py.
"""

import json
import os
import sys
import unittest

import numpy as np

from .. import readimagejrois
from .base_test import BaseTestCase


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

    def test_polygon_left(self):
        self.check_polygon("polygon-left")

    def test_polygon_top(self):
        self.check_polygon("polygon-top")

    def test_polygon_right(self):
        self.check_polygon("polygon-right")

    def test_polygon_bottom(self):
        self.check_polygon("polygon-bottom")

    def test_polygon_left_offscreen(self):
        name = "polygon-left-offscreen"
        with self.assertRaises(ValueError):
            readimagejrois.parse_roi_file(
                os.path.join(self.data_dir, name + ".roi")
            )

    def test_polygon_top_offscreen(self):
        name = "polygon-top-offscreen"
        with self.assertRaises(ValueError):
            readimagejrois.parse_roi_file(
                os.path.join(self.data_dir, name + ".roi")
            )

    def test_polygon_right_offscreen(self):
        self.check_polygon("polygon-right-offscreen")

    def test_polygon_bottom_offscreen(self):
        self.check_polygon("polygon-bottom-offscreen")

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
        self.check_mask("oval-center")

    def test_oval_full(self):
        self.check_mask("oval-full")

    def test_oval_left(self):
        self.check_mask("oval-left")

    def test_oval_top(self):
        self.check_mask("oval-top")

    def test_oval_right(self):
        self.check_mask("oval-right")

    def test_oval_bottom(self):
        self.check_mask("oval-bottom")

    def test_oval_left_offscreen(self):
        name = "oval-left-offscreen"
        with self.assertRaises(ValueError):
            readimagejrois.parse_roi_file(
                os.path.join(self.data_dir, name + ".roi")
            )

    def test_oval_top_offscreen(self):
        name = "oval-top-offscreen"
        with self.assertRaises(ValueError):
            readimagejrois.parse_roi_file(
                os.path.join(self.data_dir, name + ".roi")
            )

    def test_oval_right_offscreen(self):
        self.check_mask("oval-right-offscreen")

    def test_oval_bottom_offscreen(self):
        self.check_mask("oval-bottom-offscreen")

    def test_ellipse(self):
        self.check_mask("ellipse-center")

    def test_ellipse_left(self):
        self.check_mask("ellipse-left")

    def test_ellipse_top(self):
        self.check_mask("ellipse-top")

    def test_ellipse_right(self):
        self.check_mask("ellipse-right")

    def test_ellipse_bottom(self):
        self.check_mask("ellipse-bottom")

    def test_ellipse_left_offscreen(self):
        name = "ellipse-left-offscreen"
        with self.assertRaises(ValueError):
            readimagejrois.parse_roi_file(
                os.path.join(self.data_dir, name + ".roi")
            )

    def test_ellipse_top_offscreen(self):
        name = "ellipse-top-offscreen"
        with self.assertRaises(ValueError):
            readimagejrois.parse_roi_file(
                os.path.join(self.data_dir, name + ".roi")
            )

    def test_ellipse_right_offscreen(self):
        self.check_mask("ellipse-right-offscreen")

    def test_ellipse_bottom_offscreen(self):
        self.check_mask("ellipse-bottom-offscreen")

    def test_ellipse_tiny(self):
        # ROI which is too small to cover a single pixel
        name = "ellipse-tiny"
        with self.assertRaises(ValueError):
            readimagejrois.parse_roi_file(
                os.path.join(self.data_dir, name + ".roi")
            )
