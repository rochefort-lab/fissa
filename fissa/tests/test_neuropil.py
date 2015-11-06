''' Tests for the function in neuropil.py

Author: Sander Keemink (swkeemink@scimail.eu)
Created: 2015-11-06
'''

# use assert_() and related functions over the built in assert to ensure tests
# run properly, regardless of how python is started.
from numpy.testing import (
    assert_,
    assert_equal,
    assert_almost_equal,
    assert_array_almost_equal,
    assert_raises,
    assert_array_equal,
    dec,
    TestCase,
    run_module_suite,
    assert_allclose)

# import numpy
import numpy as np

# import neuropil functions
from .. import neuropil as npil

def test_separate():
    ''' Tests if the separate function returns data in the right format
    for all methods.    
    
    Successfullness of methods are hard to test with unittests.
    '''
    # desired shapes
    shape_desired = (2,100)    
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
    x = np.linspace(0,np.pi*2,100)    
    data = np.array([np.sin(x),np.cos(3*x)])+1
    
    # mix fake data
    W = np.array([[0.2,0.8],[0.8,0.2]])
    S = np.dot(W,data)

    # function for testing a method
    def test_method(method):
        ''' Tests for a single method
        
        Parameters
        --------------------
        Method : string
            What method to test: 'nmf', 'ica', or 'nmf_sklearn')
        
        '''
        # unmix data with ica
        S_sep,S_matched,A_sep,convergence = npil.separate(S,sep_method=method,n=2,maxiter=1,maxtries=1)
            
        # assert if formats are good
        assert_equal(S_sep.shape,shape_desired)
        assert_equal(S_matched.shape,shape_desired)
        assert_equal(S_sep.shape,shape_desired)
        if method == 'ica':
            assert_equal(convergence,con_desired_ica)
        elif method == 'nmf':
            assert_equal(convergence,con_desired_nmf)
        elif method == 'nmf_sklearn':
            assert_equal(convergence,con_desired_nmfsklearn)
    
    # test all three methods
    test_method('nmf')
    test_method('ica')
    test_method('nmf_sklearn')

