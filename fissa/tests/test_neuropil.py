''' Tests for the function in neuropil.py

Author: Sander Keemink (swkeemink@scimail.eu)
Created: 2015-11-06
'''

import unittest
import numpy as np

from .base_test import BaseTestCase

# import neuropil functions
from .. import neuropil as npil


class TestNeuropilFuns(BaseTestCase):

    def setup_class(self):
        # length of arrays in tests
        self.l = 100
        # setup basic x values for fake data
        self.x = np.linspace(0, np.pi * 2, self.l)

    def test_separate(self):
        ''' Tests if the separate function returns data in the right format
        for all methods.

        TODO: Successfullness of methods are hard to test with unittests. Need to
        make a test case where answer is known, and test against that.
        '''
        # desired shapes
        shape_desired = (2, self.l)
        con_desired_ica = {'converged': False,
                           'iterations': 1,
                           'max_iterations': 1,
                           'random_state': 892}
        con_desired_nmf = {'converged': False,
                           'iterations': 1,
                           'max_iterations': 1,
                           'random_state': 'not yet implemented'}
        con_desired_nmfsklearn = {'converged': 'not yet implemented',
                                  'iterations': 'not yet implemented',
                                  'max_iterations': 1,
                                  'random_state': 'not yet implemented'}

        # setup fake data
        data = np.array([np.sin(self.x), np.cos(3 * self.x)]) + 1

        # mix fake data
        W = np.array([[0.2, 0.8], [0.8, 0.2]])
        S = np.dot(W, data)

        # function for testing a method
        def run_method(method):
            ''' Tests for a single method

            Parameters
            --------------------
            Method : string
                What method to test: 'nmf', 'ica', or 'nmf_sklearn')

            '''
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
            elif method == 'nmf_sklearn':
                self.assert_equal(convergence, con_desired_nmfsklearn)

        # test all three methods
        run_method('nmf')
        run_method('ica')
        run_method('nmf_sklearn')

    def test_subtract_pil(self):
        ''' Tests format and effectiveness of npil.subtract_pil()

        '''
        # setup fake data
        np.random.seed(0)
        data_main = np.random.rand(self.l)
        data_cont = np.random.rand(self.l)
        data_measured = data_main + data_cont

        sig_, a = npil.subtract_pil(data_measured, data_cont)
        self.assert_equal(a, 0.93112459478303866)
        self.assert_equal(sig_.shape, data_measured.shape)

    @unittest.expectedFailure
    def test_subtract_dict(self):
        raise NotImplementedError()
