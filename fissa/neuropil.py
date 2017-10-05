"""Functions for removal of neuropil from calcium signals.

Authors: Sander Keemink (swkeemink@scimail.eu) and Scott Lowe
Created: 2015-05-15
"""

import numpy as np
import scipy.signal as signal
import numpy.random as rand
from sklearn.decomposition import FastICA, NMF, PCA


def separate(
        S, sep_method='nmf', n=None, maxiter=10000, tol=1e-4,
        random_state=892, maxtries=10, W0=None, H0=None, alpha=0.1):
    """For the signals in S, finds the independent signals underlying it,
    using ica or nmf.

    Parameters
    ----------
    S : array_like
        2d array with signals. S[i,j], j = each signal, i = signal content.
        j = 0 is considered the primary signal. (i.e. the somatic signal)
    sep_method : {'ica','nmf','nmf_sklearn'}
        Which source separation method to use, ica or nmf.
            * ica: independent component analysis
            * nmf: Non-negative matrix factorization
    n : int, optional
        How many components to estimate. If None, use PCA to estimate
        how many components would explain at least 99% of the variance.
    maxiter : int, optional
        Number of maximally allowed iterations. Default is 500.
    tol : float, optional
        Error tolerance for termination. Default is 1e-5.
    random_state : int, optional
        Initial random state for seeding. Default is 892.
    maxtries : int, optional
        Maximum number of tries before algorithm should terminate.
        Default is 10.
    W0, H0 : arrays, optional
        Optional starting conditions for nmf
    alpha : float
        [expand explanation] Roughly the sparsity constraint

    Returns
    -------
    S_sep : numpy.ndarray
        The raw separated traces
    S_matched :
        The separated traces matched to the primary signal, in order
        of matching quality (see Implementation below).
    A_sep :
    convergence : dict
        Metadata for the convergence result, with keys:
            * random_state: seed for ica initiation
            * iterations: number of iterations needed for convergence
            * max_iterations: maximun number of iterations allowed
            * converged: whether the algorithm converged or not (bool)

    Implementation
    --------------
    Concept by Scott Lowe and Sander Keemink.
    Normalize the columns in estimated mixing matrix A so that sum(column)=1
    This results in a relative score of how strongly each separated signal
    is represented in each ROI signal.
    """
    # TODO for edge cases, reduce the number of npil regions according to
    #      possible orientations
    # TODO split into several functions. Maybe turn into a class.

    # Ensure array_like input is a numpy.ndarray
    S = np.asarray(S)

    # normalize
    median = np.median(S)
    S /= median

    # estimate number of signals to find, if not given
    if n is None:
        if sep_method == 'ica':
            # Perform PCA
            pca = PCA(whiten=False)
            pca.fit(S.T)

            # find number of components with at least x percent explained var
            n = sum(pca.explained_variance_ratio_ > 0.01)
        else:
            n = S.shape[0]

    if sep_method == 'ica':
        # Use sklearn's implementation of ICA.

        for ith_try in range(maxtries):
            # Make an instance of the FastICA class. We can do whitening of
            # the data now.
            ica = FastICA(n_components=n, whiten=True, max_iter=maxiter,
                          tol=tol, random_state=random_state)

            # Perform ICA and find separated signals
            S_sep = ica.fit_transform(S.T)

            # check if max number of iterations was reached
            if ica.n_iter_ < maxiter:
                print((
                    'ICA converged after {} iterations.'
                ).format(ica.n_iter_))
                break
            print((
                'Attempt {} failed to converge at {} iterations.'
            ).format(ith_try + 1, ica.n_iter_))
            if ith_try + 1 < maxtries:
                print('Trying a new random state.')
                # Change to a new random_state
                random_state = rand.randint(8000)

        if ica.n_iter_ == maxiter:
            print((
                'Warning: maximum number of allowed tries reached at {} '
                'iterations for {} tries of different random seed states.'
            ).format(ica.n_iter_, ith_try + 1))

        A_sep = ica.mixing_

    elif sep_method == 'nmf':
        for ith_try in range(maxtries):
            # nSignals = nRegions +1
            # ICA = FastICA(n_components=nSignals)
            # ica = ICA.fit_transform(mixed.T)  # Reconstruct signals
            # A_ica = ICA.mixing_  # Get estimated mixing matrix
            #
            #

            # Make an instance of the sklearn NMF class
            if W0 is None:
                nmf = NMF(
                    init='nndsvdar', n_components=n,
                    alpha=alpha, l1_ratio=0.5,
                    tol=tol, max_iter=maxiter, random_state=random_state)

                # Perform NMF and find separated signals
                S_sep = nmf.fit_transform(S.T)

            else:
                nmf = NMF(
                    init='custom', n_components=n,
                    alpha=alpha, l1_ratio=0.5,
                    tol=tol, max_iter=maxiter, random_state=random_state)

                # Perform NMF and find separated signals
                S_sep = nmf.fit_transform(S.T, W=W0, H=H0)

            # check if max number of iterations was reached
            if nmf.n_iter_ < maxiter - 1:
                print((
                    'NMF converged after {} iterations.'
                ).format(nmf.n_iter_ + 1))
                break
            print((
                'Attempt {} failed to converge at {} iterations.'
            ).format(ith_try, nmf.n_iter_ + 1))
            if ith_try + 1 < maxtries:
                print('Trying a new random state.')
                # Change to a new random_state
                random_state = rand.randint(8000)

        if nmf.n_iter_ == maxiter - 1:
            print((
                'Warning: maximum number of allowed tries reached at {} '
                'iterations for {} tries of different random seed states.'
            ).format(nmf.n_iter_ + 1, ith_try + 1))

        A_sep = nmf.components_.T

    else:
        raise ValueError('Unknown separation method "{}".'.format(sep_method))

    # make empty matched structure
    S_matched = np.zeros(np.shape(S_sep))

    # Normalize the columns in A so that sum(column)=1 (can be done in one line
    # too).
    # This results in a relative score of how strongly each separated signal
    # is represented in each ROI signal.
    A = abs(np.copy(A_sep))
    for j in range(n):
        if np.sum(A[:, j]) != 0:
            A[:, j] /= np.sum(A[:, j])

    # get the scores for the somatic signal
    scores = A[0, :]

    # get the order of scores
    order = np.argsort(scores)[::-1]

    # order the signals according to their scores
    for j in range(n):
        s_ = A_sep[0, order[j]] * S_sep[:, order[j]]
        S_matched[:, j] = s_

    # save the algorithm convergence info
    convergence = {}
    convergence['max_iterations'] = maxiter
    if sep_method == 'ica':
        convergence['random_state'] = random_state
        convergence['iterations'] = ica.n_iter_
        convergence['converged'] = not ica.n_iter_ == maxiter
    elif sep_method == 'nmf':
        convergence['random_state'] = random_state
        convergence['iterations'] = nmf.n_iter_
        convergence['converged'] = not nmf.n_iter_ == maxiter

    # scale back to raw magnitudes
    S_matched *= median
    S *= median
    return S_sep.T, S_matched.T, A_sep, convergence


def lowPassFilter(F, fs=40, nfilt=40, fw_base=10, axis=0):
    '''Low pass filters a fluorescence imaging trace line.

    Parameters
    ----------
    F : array_like
        Fluorescence signal.
    fs : float, optional
        Sampling frequency of F, in Hz. Default 40.
    nfilt : int, optional
        Number of taps to use in FIR filter, default 40
    fw_base : float, optional
        Cut-off frequency for lowpass filter, default 1
    axis : int, optional
        Along which axis to apply low pass filtering, default 0

    Returns
    -------
    array
        Low pass filtered signal of len(F)
    '''
    # The Nyquist rate of the signal is half the sampling frequency
    nyq_rate = fs / 2.0

    # Make a set of weights to use with our taps.
    # We use an FIR filter with a Hamming window.
    b = signal.firwin(nfilt, cutoff=fw_base / nyq_rate, window='hamming')

    # Use lfilter to filter with the FIR filter.
    # We filter along the second dimension because that represents time
    filtered_f = signal.filtfilt(b, [1.0], F, axis=axis)

    return filtered_f
