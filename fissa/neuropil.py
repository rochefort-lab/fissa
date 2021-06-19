"""
Functions for removal of neuropil from calcium signals.

Authors:
    - Sander W Keemink <swkeemink@scimail.eu>
    - Scott C Lowe <scott.code.lowe@gmail.com>
Created:
    2015-05-15
"""

import numpy as np
import numpy.random as rand
import sklearn.decomposition


def separate(
        S, sep_method='nmf', n=None, maxiter=10000, tol=1e-4,
        random_state=892, maxtries=10, W0=None, H0=None, alpha=0.1):
    """
    Find independent signals, sorted by matching score against the first input signal.

    Parameters
    ----------
    S : :term:`array_like` shaped (signals, observations)
        2-d array containing mixed input signals.
        Each column of `S` should be a different signal, and each row an
        observation of the signals. For ``S[i, j]``, ``j`` is a signal, and
        ``i`` is an observation.
        The first column, ``j = 0``, is considered the primary signal and the
        one for which we will try to extract a decontaminated equivalent.

    sep_method : {"ica", "nmf"}
        Which source separation method to use, either ICA or NMF.

        - ``"ica"``: Independent Component Analysis
        - ``"nmf"``: Non-negative Matrix Factorization

    n : int, optional
        How many components to estimate. If ``None`` (default), for the NMF
        method, ``n`` is the number of input signals; for the ICA method,
        we use PCA to estimate how many components would explain at least 99%
        of the variance and adopt this value for ``n``.
    maxiter : int, optional
        Number of maximally allowed iterations. Default is ``10000``.
    tol : float, optional
        Error tolerance for termination. Default is ``1e-4``.
    random_state : int or None, optional
        Initial state for the random number generator. Set to ``None`` to use
        the numpy.random default. Default seed is ``892``.
    maxtries : int, optional
        Maximum number of tries before algorithm should terminate.
        Default is ``10``.
    W0 : :term:`array_like`, optional
        Optional starting condition for ``W`` in NMF algorithm.
        (Ignored when using the ICA method.)
    H0 : :term:`array_like`, optional
        Optional starting condition for ``H`` in NMF algorithm.
        (Ignored when using the ICA method.)
    alpha : float, optional
        Sparsity regularizaton weight for NMF algorithm. Set to zero to
        remove regularization. Default is ``0.1``.
        (Ignored when using the ICA method.)

    Returns
    -------
    S_sep : :class:`numpy.ndarray` shaped (signals, observations)
        The raw separated traces.
    S_matched : :class:`numpy.ndarray` shaped (signals, observations)
        The separated traces matched to the primary signal, in order
        of matching quality (see Notes below).
    A_sep : :class:`numpy.ndarray` shaped (signals, signals)
        Mixing matrix.
    convergence : dict
        Metadata for the convergence result, with the following keys and
        values:

        ``convergence["random_state"]``
            Seed for estimator initiation.
        ``convergence["iterations"]``
            Number of iterations needed for convergence.
        ``convergence["max_iterations"]``
            Maximum number of iterations allowed.
        ``convergence["converged"]``
            Whether the algorithm converged or not (:class:`bool`).

    Notes
    -----
    To identify which independent signal matches the primary signal best,
    we first normalize the columns in the output mixing matrix `A` such that
    ``sum(A[:, separated]) = 1``. This results in a relative score of how
    strongly each raw signal contributes to each separated signal. From this,
    we find the separated signal to which the ROI trace makes the largest
    (relative) contribution.

    See Also
    --------
    sklearn.decomposition.NMF, sklearn.decomposition.FastICA
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
        if sep_method.lower() == "ica":
            # Perform PCA
            pca = sklearn.decomposition.PCA(whiten=False)
            pca.fit(S.T)

            # find number of components with at least x percent explained var
            n = sum(pca.explained_variance_ratio_ > 0.01)
        else:
            n = S.shape[0]

    for i_try in range(maxtries):

        if sep_method.lower() in {"ica", "fastica"}:
            # Use sklearn's implementation of ICA.

            # Make an instance of the FastICA class. We can do whitening of
            # the data now.
            estimator = sklearn.decomposition.FastICA(
                n_components=n,
                whiten=True,
                max_iter=maxiter,
                tol=tol,
                random_state=random_state,
            )

            # Perform ICA and find separated signals
            S_sep = estimator.fit_transform(S.T)

        elif sep_method.lower() in {"nmf", "nnmf"}:

            # Make an instance of the sklearn NMF class
            estimator = sklearn.decomposition.NMF(
                init="nndsvdar" if W0 is None and H0 is None else "custom",
                n_components=n,
                alpha=alpha,
                l1_ratio=0.5,
                tol=tol,
                max_iter=maxiter,
                random_state=random_state,
            )

            # Perform NMF and find separated signals
            S_sep = estimator.fit_transform(S.T, W=W0, H=H0)

        elif hasattr(sklearn.decomposition, sep_method):

            print(
                "Using ad hoc signal decomposition method"
                " sklearn.decomposition.{}. Only NMF and ICA are officially"
                " supported.".format(sep_method)
            )

            # Load up arbitrary decomposition algorithm from sklearn
            estimator = getattr(sklearn.decomposition, sep_method)(
                n_components=n,
                tol=tol,
                max_iter=maxiter,
                random_state=random_state,
            )
            S_sep = estimator.fit_transform(S.T)

        else:
            raise ValueError('Unknown separation method "{}".'.format(sep_method))

        # check if max number of iterations was reached
        if estimator.n_iter_ < maxiter:
            print(
                "{} converged after {} iterations.".format(
                    sep_method, estimator.n_iter_
                )
            )
            break

        print(
            "Attempt {} failed to converge at {} iterations.".format(
                i_try + 1, estimator.n_iter_
            )
        )
        if i_try + 1 < maxtries:
            print("Trying a new random state.")
            # Change to a new random_state
            if random_state is not None:
                random_state = (random_state + 1) % 2**32

    if estimator.n_iter_ == maxiter:
        print((
            'Warning: maximum number of allowed tries reached at {} '
            'iterations for {} tries of different random seed states.'
        ).format(estimator.n_iter_, i_try + 1))

    if hasattr(estimator, "mixing_"):
        A_sep = estimator.mixing_
    else:
        A_sep = estimator.components_.T

    # make empty matched structure
    S_matched = np.zeros(np.shape(S_sep))

    # Normalize the columns in A so that sum(column)=1 (can be done in one line
    # too).
    # This results in a relative score of how strongly each separated signal
    # is represented in each ROI signal.
    #
    # Our mixing matrix is shaped (input/raw, output/separated). For each
    # separated (output) signal, we find how much weighting each input (raw)
    # signal contributes to that separated signal, relative to the other input
    # signals.
    A = abs(np.copy(A_sep))
    for j in range(n):
        if np.sum(A[:, j]) != 0:
            A[:, j] /= np.sum(A[:, j])

    # get the scores for the somatic signal
    scores = A[0, :]

    # Rank the separated signals in descending ordering of their score.
    # The separated signal to which the somatic signal makes up the largest
    # contribution is sorted first.
    order = np.argsort(scores)[::-1]

    # order the signals according to their scores
    for j in range(n):
        s_ = A_sep[0, order[j]] * S_sep[:, order[j]]
        S_matched[:, j] = s_

    # save the algorithm convergence info
    convergence = {}
    convergence["max_iterations"] = maxiter
    convergence["random_state"] = random_state
    convergence["iterations"] = estimator.n_iter_
    convergence["converged"] = estimator.n_iter_ != maxiter

    # scale back to raw magnitudes
    S_matched *= median
    S *= median
    return S_sep.T, S_matched.T, A_sep, convergence
