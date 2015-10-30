''' Functions neuropil removal

Author: Sander Keemink (swkeemink@scimail.eu)
Created: 2015-05-15
'''
import numpy as np
import numpy.random as rand
from sklearn.decomposition import FastICA, ProjectedGradientNMF,PCA
from scipy.optimize import minimize_scalar
import nimfa 
            
def separate(S,sep_method='ica',n=None,maxiter=500,tol=1e-5,random_state=892,maxtries=10):
    ''' 
    For the signals in S, finds the independent signals underlying it, using 
    ica or nmf. Several methods for signal picking are implemented, see below, 
    of which method 5 works best in general. 
    
    Parameters
    ---------------------
    S      : 2D array
        2d array with signals. S[i,j], j = each signal, i = signal content.
        j = 0 is considered the primary signal. (i.e. the somatic signal)
    sep_method : string {'ica','nmf','nmf_sklearn'}
        Which source separation to use, ica or nmf. 
        nmf uses the nimfa implementation. 
        nmf_sklearn uses the sklearn implementation, which is slower.
    n : int
        how many components to estimate. If None, use PCA to estimate 
        how many components would explain at least 95% of the variance
    maxiter : int {500}
        Number of maximally allowed iterations
    tol : float (1e-5)
        Error tolerance for termination
    random_state : int (892)
        Initial random state for seeding
    maxtries : int (10)
          maximum number of tries before algorithm should terminate
    
    Returns
    ---------------------
    S_sep,S_matched,A_sep,convergence = separate(args)
    S_sep  : The raw separated traces
    S_matched : The separated traces matched to the primary signal, in order 
                of matching quality (see Matching Method)
    convergence : a dictionary with [random_state,iterations,max_iterations,converged]
                random_state: seed for ica initiation
                iterations: number of iterations needed for convergence
                max_iterations: maximun number of iterations allowed
                converged: whether the algorithm converged or not (bool)
                
    Matching Method
    --------------------
    Concept by Scott Lowe.
    Normalize the columns in A so that sum(column)=1
    This results in a relative score of how strongly each separated signal
    is represented in each ROI signal.

    TODO
    --------------------
    - Return several candidates for each signal, so can more easily compare
    '''   
    # find number of input signals
    
    
    # estimate number of signals to find, if not given    
    if n == None:
        # do pca        
        pca = PCA(whiten=False) #?why not whiten?
        pca.fit(S)

        # find cummulative explained variance
        exp_var = np.cumsum(pca.explained_variance_ratio_)   
        
        # set number of components as moment when 90 % of variance is explained
        n =np. where(exp_var>0.95)[0][0]+1

    # set max iterations reached flag TODO: change this flag to a certain number of random_state changes
    flag = True
    
    # start tries counter
    counter = 0
    
    # do ica for increasing maximum iterations, until the algorithm terminates before the max iter is reached
    if sep_method == 'ica': # if ica is selected
        while flag:
            # define ica method, with whitening of data
            ica = FastICA(n_components = n,whiten=True,max_iter = maxiter,tol=tol,random_state=random_state)
             
            # do the ica and find separated signals
            S_sep = ica.fit_transform(S.T)
                
            # check if max number of iterations was reached
            if ica.n_iter_ == maxiter and counter == maxtries:
                flag = False
                print 'Warning: maximum number of allowed tries reached at ' + str(ica.n_iter_) + ' iterations for ' + str(counter) + ' tries.'
            elif ica.n_iter_ == maxiter:
                print 'failed to converge at ' + str(ica.n_iter_) + ' iterations, trying a new random_state.'
                random_state=rand.randint(8000) # iterate random_state
                counter += 1 # iterate counter
            else:
                flag = False # stops while loop
                print 'needed ' + str(ica.n_iter_) + ' iterations to converge'
        A_sep=  ica.mixing_
            
    elif sep_method == 'nmf_sklearn': # the sklearn nmf method, is slow and can't tell how many iterations were used
        # define nmf method (from sklearn)
        nmf = ProjectedGradientNMF(init='nndsvd',sparseness='data',n_components=n,tol=tol,max_iter=maxiter,random_state=random_state)
        
        # separate signals and get mixing matrix
        S_sep = nmf.fit_transform(S.T)
        A_sep  = nmf.components_.T
        
    elif sep_method == 'nmf': # the nimfa method, fast and reliable
        
        # define nmf method (from nimfa)
        nmf = nimfa.Nmf(S.T, max_iter=maxiter, rank=n, seed='nndsvd', method='snmnmf')
        
        # fit the model
        nmf_fit = nmf()
        
        # get fit summary
        fs = nmf_fit.summary()
        
        # check if max number of iterations was reached
        if fs['n_iter'] == maxiter:
            print 'Warning: maximum number of allowed iterations reached at ' + str(fs['n_iter']) + ' iterations.'
        else:
            print 'Nmf converged at ' + str(fs['n_iter']) + ' iterations.'

        # get the mixing matrix and estimated data
        A_sep = np.array(nmf_fit.coef()).T
        S_sep = np.array(nmf_fit.basis())
        
    else:
        raise ValueError ('Unknown separation method, can only use ica or nmf or nmf_sklearn')

    # make empty matched structure    
    S_matched = np.zeros(np.shape(S_sep))
    
    # Concept by Scott Lowe.
    # Normalize the columns in A so that sum(column)=1 (can be done in one line too)
    # This results in a relative score of how strongly each separated signal
    # is represented in each ROI signal. 
    A = abs(np.copy(A_sep))
    for j in range(n):
        A[:,j] /= np.sum(A[:,j])
    
    # get the scores for the somatic signal
    scores = abs(A[0,:])

    # get the order of scores
    order = np.argsort(scores)[::-1]
    
    # order the signals according to their scores
    for j in range(n):
        s_ = A_sep[0,order[j]]*S_sep[:,order[j]]
        S_matched[:,j] = s_

    # save the algorithm convergence info
    convergence = {}
    convergence['max_iterations'] = maxiter
    if sep_method == 'ica':
        convergence['random_state'] = random_state
        convergence['iterations']   = ica.n_iter_
        convergence['converged'] = not ica.n_iter_==maxiter
    elif sep_method =='nmf':
        convergence['random_state'] = 'not yet implemented'
        convergence['iterations']   = fs['n_iter']
        convergence['converged'] = not fs['n_iter'] == maxiter
    elif sep_method =='nmf_sklearn':
        convergence['random_state'] = 'not yet implemented'
        convergence['iterations']   ='not yet implemented'
        convergence['converged'] = 'not yet implemented'    
    
    return S_sep.T,S_matched.T,A_sep,convergence
   
def subtract_pil(sig,pil):
    ''' subtract the neuropil (pil) from the signal (sig), in such a manner 
    that that the correlation between the two is minimized:
    sig_ = sig - a*pil
    find 'a' such that cor(sig_,pil) is minimized. A is bound to be 0-1.5.    
    
    Parameters
    -------------------
    sig : array
        signal
    pil :  array
        neuropils
        
    Returns
    ---------------
    sig_,a = subtractpil(sig,pil)
    sig_ : the signal with neuropil subtracted
    a : the subtraction parameter that results in the best subtraction.
    '''
    def mincorr(x):
        ''' find the correlation between sig and pil, for subtraction with gain x '''
        sig_ = sig-x*pil
        corr = np.corrcoef(sig_,pil)[0,1]
        return np.sqrt(corr**2)
    
    res = minimize_scalar(mincorr, bounds=(0, 1.5), method='bounded')
    a = res.x # the resulting gain
    sig_ = sig-a*pil+np.mean(a*pil) # the output signal
    
    return sig_,a
    
def subtract_dict(S,n_noncell):
    '''
    Returns dictionary with the cell traces minus the background traces,
    with the subtraction method in subtractpil   
    
    Parameters
    -------------------------
    S : dictionary
        Dictionary containing sets of traces
    n_noncell : int
        How many noncells there are (i.e. ROIs without neuropils)
    '''    
    S_subtract = {}
    a = {}
    for i in range(n_noncell,len(S)):
        S_subtract[i],a[i] = subtractpil(S[i][:,0],np.mean(S[i][:,1:],axis=1))
    
    return S_subtract,a  
    
