"""Tests for the functions in neuropil.py.

Author: Sander Keemink (swkeemink@scimail.eu)
Created: 2015-11-06
"""

import warnings

import numpy as np

from .base_test import BaseTestCase
from .. import neuropil as npil


class NeuropilMixin:
    """Neuropil functions testing class."""

    def setUp(self):
        """Set up the basic variables."""
        # length of arrays in tests
        self.l = 100
        self.shape_desired = (2, self.l)
        # setup basic x values for fake data
        self.x = np.linspace(0, np.pi * 2, self.l)
        # Generate simple source data
        self.data = np.array([np.sin(self.x), np.cos(3 * self.x)]) + 1
        # Mix to make test data
        self.W = np.array([[0.2, 0.8], [0.8, 0.2]])
        self.S = np.dot(self.W, self.data)

    def run_method(self, method, expected_converged=None, **kwargs):
        """Test a single method, with specific parameters."""
        # Run the separation routine
        S_sep, S_matched, A_sep, convergence = npil.separate(
            self.S, sep_method=method, **kwargs
        )
        # Ensure output shapes are as expected
        self.assert_equal(S_sep.shape, self.shape_desired)
        self.assert_equal(S_matched.shape, self.shape_desired)
        self.assert_equal(S_sep.shape, self.shape_desired)
        # If specified, assert that the result is as expected
        if expected_converged is not None:
            self.assert_equal(convergence['converged'], expected_converged)

    def test_method(self):
        self.run_method(self.method, expected_converged=True, maxtries=1)

    def test_reduce_dim(self):
        self.run_method(self.method, expected_converged=True, maxtries=1, n=2)

    def test_manual_seed(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.run_method(
                self.method,
                expected_converged=True,
                maxtries=1,
                random_state=0,
            )

    def test_retry(self):
        self.run_method(self.method, maxiter=1, maxtries=3)


class TestNeuropilNMF(BaseTestCase, NeuropilMixin):

    def setUp(self):
        NeuropilMixin.setUp(self)
        self.method = "nmf"

    def test_nmf_manual_alpha(self):
        self.run_method(self.method, expected_converged=True, alpha=.2)

    def test_badmethod(self):
        with self.assertRaises(ValueError):
            npil.separate(self.S, sep_method='bogus_method')


class TestNeuropilICA(BaseTestCase, NeuropilMixin):

    def setUp(self):
        NeuropilMixin.setUp(self)
        self.method = "ica"


class TestNeuropilFA(BaseTestCase, NeuropilMixin):

    def setUp(self):
        NeuropilMixin.setUp(self)
        self.method = "FactorAnalysis"
