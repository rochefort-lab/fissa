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

    def test_separate(self):
        """Tests the separate function data format."""
        # TODO: Successfullness of methods are hard to test with unittests.
        #       Need to make a test case where answer is known, and test
        #       against that.

        # desired shapes
        shape_desired = (2, self.l)
        con_desired_ica = {'converged': False,
                           'iterations': 1,
                           'max_iterations': 1,
                           'random_state': 892}
        con_desired_nmf = {'converged': True,
                           'iterations': 0,
                           'max_iterations': 1,
                           'random_state': 892}

        # setup fake data
        data = np.array([np.sin(self.x), np.cos(3 * self.x)]) + 1

        # mix fake data
        W = np.array([[0.2, 0.8], [0.8, 0.2]])
        S = np.dot(W, data)

        # function for testing a method
        def run_method(method):
            """Test a single method.

            Parameters
            ----------
            Method : string
                What method to test: 'nmf', 'ica', or 'nmf_sklearn')

            """
            # unmix data with ica
            S_sep, S_matched, A_sep, convergence = npil.separate(
                S, sep_method=method, n=2, maxiter=1, maxtries=1)

            # assert if formats are good
            self.assert_equal(S_sep.shape, shape_desired)
            self.assert_equal(S_matched.shape, shape_desired)
            self.assert_equal(S_sep.shape, shape_desired)
            if method == 'ica':
                self.assert_equal(convergence, con_desired_ica)
            elif method == 'nmf':
                self.assert_equal(convergence, con_desired_nmf)

        # test all two methods
        run_method('nmf')
        run_method('ica')
