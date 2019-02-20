"""
Functions for removal of neuropil from calcium signals.

Authors:
    - Sander W Keemink (swkeemink@scimail.eu)
    - Scott C Lowe
Created:
    2015-05-15
"""

import numpy as np
import numpy.random as rand
import scipy.signal as signal
from sklearn.decomposition import FastICA, NMF, PCA


def separate(
        S, sep_method='nmf', n=None, maxiter=10000, tol=1e-4,
        random_state=892, maxtries=10, W0=None, H0=None, alpha=0.1):
    """For the signals in S, finds the independent signals underlying it,
    using ica or nmf.

    Parameters
    ----------
    S : array_like
        2-d array containing mixed input signals.
        Each column of `S` should be a different signal, and each row an
        observation of the signals. For `S[i,j]`, `j` = each signal,
        `i` = signal content.
        The first column, `j = 0`, is considered the primary signal and the
        one for which we will try to extract a decontaminated equivalent.

    sep_method : {'ica','nmf'}
        Which source separation method to use, either ICA or NMF.

        - `'ica'`: Independent Component Analysis
        - `'nmf'`: Non-negative Matrix Factorization

    n : int, optional
        How many components to estimate. If `None` (default), use PCA to
        estimate how many components would explain at least 99% of the
        variance and adopt this value for `n`.
    maxiter : int, optional
        Number of maximally allowed iterations. Default is 500.
    tol : float, optional
        Error tolerance for termination. Default is 1e-5.
    random_state : int, optional
        Initial state for the random number generator. Set to None to use
        the numpy.random default. Default seed is 892.
    maxtries : int, optional
        Maximum number of tries before algorithm should terminate.
        Default is 10.
    W0 : array_like, optional
        Optional starting condition for `W` in NMF algorithm.
        (Ignored when using the ICA method.)
    H0 : array_like, optional
        Optional starting condition for `H` in NMF algorithm.
        (Ignored when using the ICA method.)
    alpha : float, optional
        Sparsity regularizaton weight for NMF algorithm. Set to zero to
        remove regularization. Default is 0.1.
        (Ignored when using the ICA method.)

    Returns
    -------
    numpy.ndarray
        The raw separated traces.
    numpy.ndarray
        The separated traces matched to the primary signal, in order
        of matching quality (see Methods below).
    numpy.ndarray
        Mixing matrix.
    dict
        Metadata for the convergence result, with keys:

        - `'random_state'`: seed for ICA initiation
        - `'iterations'`: number of iterations needed for convergence
        - `'max_iterations'`: maximum number of iterations allowed
        - `'converged'`: whether the algorithm converged or not (bool)

    Notes
    -----
    Concept by Scott Lowe and Sander Keemink.
    Normalize the columns in estimated mixing matrix A so that `sum(column)=1`.
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
            if W0 is None and H0 is None:
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
        Sampling frequency of F, in Hz. Default is 40.
    nfilt : int, optional
        Number of taps to use in FIR filter. Default is 40.
    fw_base : float, optional
        Cut-off frequency for lowpass filter, in Hz. Default is 10.
    axis : int, optional
        Along which axis to apply low pass filtering. Default is 0.

    Returns
    -------
    numpy.ndarray
        Low pass filtered signal with the same shape as `F`.
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
