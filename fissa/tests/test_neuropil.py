"""Tests for the functions in neuropil.py.

Author: Sander Keemink (swkeemink@scimail.eu)
Created: 2015-11-06
"""
import numpy as np

from .base_test import BaseTestCase
from .. import neuropil as npil


class TestNeuropilFuns(BaseTestCase):
    """Neuropil functions testing class."""

    def setup_class(self):
        """Set up the basic variables."""
        # length of arrays in tests
        self.l = 100
        # setup basic x values for fake data
        self.x = np.linspace(0, np.pi * 2, self.l)
        # Generate simple source data
        self.data = np.array([np.sin(self.x), np.cos(3 * self.x)]) + 1
        # Mix to make test data
        self.W = np.array([[0.2, 0.8], [0.8, 0.2]])
        self.S = np.dot(self.W, self.data)

    def test_separate(self):
        """Tests the separate function data format."""
        # desired shapes
        shape_desired = (2, self.l)

        # function for testing a method
        def run_method(method, expected_converged=None, **kwargs):
            """Test a single method, with specific parameters."""
            # Run the separation routine
            S_sep, S_matched, A_sep, convergence = npil.separate(
                self.S, sep_method=method, **kwargs
            )
            # Ensure output shapes are as expected
            self.assert_equal(S_sep.shape, shape_desired)
            self.assert_equal(S_matched.shape, shape_desired)
            self.assert_equal(S_sep.shape, shape_desired)
            # If specified, assert that the result is as expected
            if expected_converged is not None:
                self.assert_equal(convergence['converged'], expected_converged)

        # Run tests
        i_subtest = 0
        for method in ["nmf", "ica", "FactorAnalysis"]:
            with self.subTest(i_subtest):
                run_method(method, expected_converged=True, maxtries=1, n=2)
            i_subtest += 1
            with self.subTest(i_subtest):
                run_method(method, expected_converged=True, maxtries=1)
            i_subtest += 1
            with self.subTest(i_subtest):
                run_method(method, expected_converged=True, maxtries=1, random_state=0)
            i_subtest += 1
            with self.subTest(i_subtest):
                run_method(method, maxiter=1, maxtries=3)
            i_subtest += 1

        with self.subTest(i_subtest):
            run_method('nmf', expected_converged=True, alpha=.2)
        i_subtest += 1

    def test_badmethod(self):
        with self.assertRaises(ValueError):
            npil.separate(self.S, sep_method='bogus_method')
