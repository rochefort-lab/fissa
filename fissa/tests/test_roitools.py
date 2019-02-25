'''Unit tests for roitools.py.'''

from __future__ import division

import unittest
import numpy as np

from .base_test import BaseTestCase
from .. import roitools


class TestGetMaskCom(BaseTestCase):
    '''Tests for get_mask_com.'''

    def test_trivial_list(self):
        actual = roitools.get_mask_com(
            [[True]]
        )
        desired = (0, 0)
        self.assert_equal(actual, desired)

    def test_trivial_np_bool(self):
        mask = np.ones((1, 1), dtype=bool)
        desired = (0, 0)
        actual = roitools.get_mask_com(mask)
        self.assert_equal(actual, desired)

    def test_trivial_np_int8(self):
        mask = np.ones((1, 1), dtype=np.int8)
        desired = (0, 0)
        actual = roitools.get_mask_com(mask)
        self.assert_equal(actual, desired)

    def test_trivial_np_float32(self):
        mask = np.ones((1, 1), dtype=np.float32)
        desired = (0, 0)
        actual = roitools.get_mask_com(mask)
        self.assert_equal(actual, desired)

    def test_main(self):
        mask = [
            [False, False],
            [True, False],
        ]
        actual = roitools.get_mask_com(mask)
        desired = (1, 0)
        self.assert_equal(actual, desired)
        actual = roitools.get_mask_com(np.array(mask))
        self.assert_equal(actual, desired)
        actual = roitools.get_mask_com(np.array(mask).astype(np.int8))
        self.assert_equal(actual, desired)
        actual = roitools.get_mask_com(np.array(mask).astype(np.float32))
        self.assert_equal(actual, desired)

        actual = roitools.get_mask_com(
            [[False, True],
             [False, False],
            ]
        )
        desired = (0, 1)
        self.assert_equal(actual, desired)

        actual = roitools.get_mask_com(
            [[False, True, True],
             [False, False, True],
            ]
        )
        desired = (1/3, 1 + 2/3)
        self.assert_allclose(actual, desired)

    def test_non2d_input(self):
        self.assertRaises(ValueError, roitools.get_mask_com, [])
        self.assertRaises(ValueError, roitools.get_mask_com, np.ones((1)))
        self.assertRaises(ValueError, roitools.get_mask_com, np.ones((1,1,1)))

    @unittest.expectedFailure
    def test_non_boolean(self):
        actual = roitools.get_mask_com(
            [[0, 1, 2],
             [0, 0, 1],
            ]
        )
        desired = (3/4, 1 + 3/4)
        self.assert_allclose(actual, desired)


class TestShift2dArray(BaseTestCase):
    '''Tests for shift_2d_array.'''

    def test_noop(self):
        x = np.array([4, 5, 1])
        desired = x.copy()
        actual = roitools.shift_2d_array(x, shift=0, axis=0)
        self.assert_equal(actual, desired)

        x = np.array([[4, 7], [5, 1], [2, -2]])
        desired = x.copy()
        actual = roitools.shift_2d_array(x, shift=0, axis=0)
        self.assert_equal(actual, desired)

        x = np.array([[4, 7], [5, 1], [2, -2]])
        desired = x.copy()
        actual = roitools.shift_2d_array(x, shift=0, axis=1)
        self.assert_equal(actual, desired)

        x = np.array([[4, 7], [5, 1], [2, -2]])
        desired = x.copy()
        actual = roitools.shift_2d_array(x, shift=0)
        self.assert_equal(actual, desired)

    def test_roll_axis0(self):
        x = np.array([4, 5, 1, 3])
        actual = roitools.shift_2d_array(x, shift=1, axis=0)
        desired = np.array([0, 4, 5, 1])
        self.assert_equal(actual, desired)

        x = np.expand_dims(x, -1)
        actual = roitools.shift_2d_array(x, shift=1, axis=0)
        desired = np.expand_dims(desired, -1)
        self.assert_equal(actual, desired)

        x = np.array([4, 5, 1, 3])
        actual = roitools.shift_2d_array(x, shift=2, axis=0)
        desired = np.array([0, 0, 4, 5])
        self.assert_equal(actual, desired)

        x = np.expand_dims(x, -1)
        actual = roitools.shift_2d_array(x, shift=2, axis=0)
        desired = np.expand_dims(desired, -1)
        self.assert_equal(actual, desired)

        x = np.array([4, 5, 1, 3])
        actual = roitools.shift_2d_array(x, shift=-1, axis=0)
        desired = np.array([5, 1, 3, 0])
        self.assert_equal(actual, desired)

        x = np.expand_dims(x, -1)
        actual = roitools.shift_2d_array(x, shift=-1, axis=0)
        desired = np.expand_dims(desired, -1)
        self.assert_equal(actual, desired)

        x = np.array([[4, 5, 1, 3]])
        actual = roitools.shift_2d_array(x, shift=1, axis=0)
        desired = np.array([[0, 0, 0, 0]])
        self.assert_equal(actual, desired)

    def test_roll_axis1(self):
        x = np.array([[4, 5, 1, 3], [7, -1, 9, 2]])
        actual = roitools.shift_2d_array(x, shift=1, axis=1)
        desired = np.array([[0, 4, 5, 1], [0, 7, -1, 9]])
        self.assert_equal(actual, desired)

        x = np.array([[4, 5, 1, 3], [7, -1, 9, 2]])
        actual = roitools.shift_2d_array(x, shift=2, axis=1)
        desired = np.array([[0, 0, 4, 5], [0, 0, 7, -1]])
        self.assert_equal(actual, desired)

        x = np.array([[4, 5, 1, 3], [7, -1, 9, 2]])
        actual = roitools.shift_2d_array(x, shift=-1, axis=1)
        desired = np.array([[5, 1, 3, 0], [-1, 9, 2, 0]])
        self.assert_equal(actual, desired)


class TestSplitNpil(BaseTestCase):
    '''Tests for split_npil.'''

    def test_2x2(self):
        mask = np.ones((2, 2))
        npil_masks = roitools.split_npil(mask, (0.5, 0.5), 4)
        desired_npil_masks = [
            np.array([[False,  True], [False, False]]),
            np.array([[False, False], [False,  True]]),
            np.array([[False, False], [ True, False]]),
            np.array([[ True, False], [False, False]]),
        ]
        self.assert_equal_list_of_array_perm_inv(desired_npil_masks,
                                                 npil_masks)

    def test_bottom(self):
        mask = [[0, 0], [1, 1]]
        npil_masks = roitools.split_npil(mask, (0.5, 0.5), 4)
        desired_npil_masks = [
            np.array([[False, False], [False,  True]]),
            np.array([[False, False], [False, False]]),
            np.array([[False, False], [False, False]]),
            np.array([[False, False], [ True, False]]),
        ]
        self.assert_equal_list_of_array_perm_inv(desired_npil_masks,
                                                 npil_masks)

    def test_bottom2(self):
        mask = [[0, 0], [1, 1]]
        npil_masks = roitools.split_npil(mask, (1, 1), 2)
        desired_npil_masks = [
            np.array([[False, False], [False,  True]]),
            np.array([[False, False], [ True, False]]),
        ]
        self.assert_equal_list_of_array_perm_inv(desired_npil_masks,
                                                 npil_masks)

    def test_bottom_adaptive(self):
        mask = [[0, 0], [1, 1]]
        npil_masks = roitools.split_npil(mask, (1, 1), 4,
                                               adaptive_num=True)
        desired_npil_masks = [
            np.array([[False, False], [False,  True]]),
            np.array([[False, False], [ True, False]]),
        ]
        self.assert_equal_list_of_array_perm_inv(desired_npil_masks,
                                                 npil_masks)


class TestGetNpilMask(BaseTestCase):
    '''Tests for get_npil_mask.'''

    def test_empty(self):
        mask = [
            [True, False, False],
            [False, False, False],
            [False, False, False],
        ]
        actual = roitools.get_npil_mask(mask, 0)
        desired = np.array([
            [False, False, False],
            [False, False, False],
            [False, False, False],
        ])
        self.assert_equal(actual, desired)

    def test_full(self):
        mask = [
            [True, False, False],
            [False, False, False],
            [False, False, False],
        ]
        desired = np.array([
            [False, True, True],
            [True, True, True],
            [True, True, True],
        ])

        actual = roitools.get_npil_mask(mask, 8)
        self.assert_equal(actual, desired)

        actual = roitools.get_npil_mask(mask, 1000)
        self.assert_equal(actual, desired)

    def test_corner(self):
        mask = [
            [True, False, False],
            [False, False, False],
            [False, False, False],
        ]

        desired = np.array([
            [False, True, False],
            [True, False, False],
            [False, False, False],
        ])
        for area in [1, 2]:
            actual = roitools.get_npil_mask(mask, area)
            self.assert_equal(actual, desired)

        desired = np.array([
            [False, True, False],
            [True, True, True],
            [False, True, False],
        ])
        for area in [3, 4, 5]:
            actual = roitools.get_npil_mask(mask, area)
            self.assert_equal(actual, desired)

        desired = np.array([
            [False, True, True],
            [True, True, True],
            [True, True, True],
        ])
        for area in [6, 7, 8]:
            actual = roitools.get_npil_mask(mask, area)
            self.assert_equal(actual, desired)

    def test_middle(self):
        mask = [
            [False, False, False],
            [False, True, False],
            [False, False, False],
        ]

        desired = np.array([
            [False, True, False],
            [True, False, True],
            [False, True, False],
        ])
        for area in [1, 2, 3, 4]:
            actual = roitools.get_npil_mask(mask, area)
            self.assert_equal(actual, desired)

        desired = np.array([
            [True, True, True],
            [True, False, True],
            [True, True, True],
        ])
        for area in [5, 6, 7, 8]:
            actual = roitools.get_npil_mask(mask, area)
            self.assert_equal(actual, desired)
