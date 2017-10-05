'''
Functions for computing correcting fluorescence signals for changes in
baseline activity.

Author: Scott Lowe
'''

import numpy as np
import scipy.signal


def findBaselineF0(rawF, fs, axis=0, keepdims=False):
    """Find the baseline for a fluorescence imaging trace line.

    The baseline, F0, is the 5th-percentile of the 1Hz
    lowpass filtered signal.

    Parameters
    ----------
    rawF : array_like
        Raw fluorescence signal.
    fs : float
        Sampling frequency of rawF, in Hz.
    axis : int, optional
        Dimension which contains the time series. Default is 0.
    keepdims : bool, optional
        Whether to preserve the dimensionality of the input. Default is
        False.

    Returns
    -------
    baselineF0 : numpy.ndarray
        The baseline fluorescence of each recording, as an array.

    In typical usage, the input rawF is expected to be sized
    (numROI, numTimePoints, numRecs)
    and the output will then be sized (numROI, 1, numRecs)
    if keepdims is True.
    """
    # Parameters --------------------------------------------------------------
    nfilt = 30  # Number of taps to use in FIR filter
    fw_base = 1  # Cut-off frequency for lowpass filter
    base_pctle = 5  # Percentile to take as baseline value

    # Main --------------------------------------------------------------------
    # Ensure array_like input is a numpy.ndarray
    rawF = np.asarray(rawF)

    # Remove the first datapoint, because it can be an erroneous sample
    rawF = np.split(rawF, [1], axis)[1]

    # The Nyquist rate of the signal is half the sampling frequency
    nyq_rate = fs / 2.0

    # Make a set of weights to use with our taps.
    # We use an FIR filter with a Hamming window.
    b = scipy.signal.firwin(nfilt, cutoff=fw_base / nyq_rate, window='hamming')

    # Use lfilter to filter with the FIR filter.
    # We filter along the second dimension because that represents time
    filtered_f = scipy.signal.filtfilt(b, [1.0], rawF, axis=axis)

    # Take a percentile of the filtered signal
    baselineF0 = np.percentile(filtered_f, base_pctle, axis=axis,
                               keepdims=keepdims)

    return baselineF0
