"""
Unit tests for delaf.py
"""
from __future__ import division

import unittest
import numpy as np

from .base_test import BaseTestCase

from .. import deltaf

class TestFindBaseline(BaseTestCase):

    @unittest.expectedFailure
    def test_trivial(self):
        # Test trivial input
        fs = 10
        array = np.zeros((1))
        desired = array
        actual = deltaf.findBaselineF0(array, fs)
        self.assert_allclose(actual, desired)

    def test_simple(self):
        # Simple sanity-check test
        fs = 10
        array = np.zeros((100))
        desired = np.array([0])
        actual = deltaf.findBaselineF0(array, fs)
        self.assert_allclose(actual, desired)

        # Test simple input
        fs = 10
        array = np.ones((1000))
        # Make a few of the values be notably larger
        array[0:200:] = 20
        # We should still be able to do this perfectly
        desired = np.array([1])
        actual = deltaf.findBaselineF0(array, fs)
        self.assert_allclose(actual, desired)

    def test_multidimensional(self):
        # Test multi-dimensional input
        fs = 10
        array = np.zeros((1000,3))
        array[:,0] = 2
        array[:,2] = -1
        desired = np.array([2,0,-1])
        actual = deltaf.findBaselineF0(array, fs)
        self.assert_allclose(actual, desired)

        # Test multi-dimensional input with keepdims
        desired = np.array([[2,0,-1]])
        actual = deltaf.findBaselineF0(array, fs, keepdims=True)
        self.assert_allclose(actual, desired)

        # Test multi-dimensional input along other dimension
        array = np.transpose(array)
        desired = np.array([[2],[0],[-1]])
        actual = deltaf.findBaselineF0(array, fs, axis=1, keepdims=True)
        self.assert_allclose(actual, desired)
