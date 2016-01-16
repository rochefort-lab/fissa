'''
Functions for computing correcting fluorescence signals for changes in
baseline activity.

Author: Scott Lowe
'''

import numpy as np
import scipy.signal

def normaliseF(rawF, fs=40, ax_time=1, ax_recs=-1, output_f0=False):
    '''Normalises a fluorescence imaging trace line.
    
    Takes a raw fluorescence signal and normalises it by subtracting a
    baseline fluorescence from each recording and dividing by a
    fluorescence scale factor.
    The baseline, or subtractive, F0 is the 5th-percentile of the 1Hz
    lowpass filtered signal of each ROI in each recording.
    The scalefactor, or divisive, F0 is the mean of the baseline F0 in
    each recording, capped to be non-negative. The scaling F0 is capped
    to be no less than 1.
    
    Parameters
    ----------
    rawF : numpy.ndarray, list or numpy arrays
        Raw fluorescence signal.
    fs : float, optional
        The sampling frequency of rawF, measured in Hz. Default is 40Hz.
    ax_time : int, optional
        The dimension along which time is stored. This dimension will
        be used to find the baseline activity across time. Default is 1.
    ax_recs : int, optional
        The dimension along which independent recording samples are
        stored. This dimension will be used to take the average across
        recordings. Default is -1.
    output_f0 : bool, optional
        Whether to output the baseline and scaling f0 values as well as
        the normalised result. Default is False.
    
    Returns
    -------
    normalisedF : numpy.ndarray, sized the same as the input rawF.
        Normalised fluorescence.
    
    Additional Returns
    ------------------
    baselineF0 : np.ndarray
        The baseline fluorescence of each recording.
    scaleF0 : np.ndarray
        The scale factor F0 applied to all recordings.
    
    
    Expected usage is to either
    - input a np.ndarray of size (numROI, numTimePoints, numRecs)
        normaliseF then outputs a normalised array of the same size as
        the input.
        With output_f0 enabled, baselineF0 is sized (numROI, 1, numRecs),
        and scaleF0 is sized (numROI, 1, 1).
    - input a list of length numRecs containing arrays, each of size
        (numROI, numTimePoints_iRec).
        normaliseF then outputs a list of normalised arrays each with a
        size corresponding to the input list, (numROI, numTimePoints_iRec).
        With output_f0 enabled, baselineF0 is a list of arrays each sized
        (numROI, 1), and scaleF0 is sized (numROI, 1).
    If your input does not have the dimension representing time on axis
    1 or does not have the dimension for recordings as the final axis,
    you can change the inputs accordingly.
    '''
    
    # Parameters --------------------------------------------------------------
    min_f0_bl = 0.0 # Lowerbound on f0 for taking mean
    min_f0_sf = 1.0 # Lowerbound on divisive f0, applied after averaging
    
    # Main --------------------------------------------------------------------
    if isinstance(rawF, np.ndarray):
        # Find the baseline F0 for each ROI in each recording
        baselineF0 = findBaselineF0(rawF, fs, axis=ax_time, keepdims=True)
    else:
        # Check the axis for averaging over is set to -1, otherwise the input
        # doesn't make sense.
        if ax_recs is not -1:
            raise ValueError('A list of recordings was input, but the axis to ' +
                    'average over was also set.')
        # Find the baseline F0 for each ROI in each recording
        bl_list = [findBaselineF0(x, fs, axis=ax_time, keepdims=True)
                            for x in rawF]
        # Concatenate them together along a new, postpended, axis
        baselineF0 = np.concatenate([x[..., np.newaxis] for x in bl_list], -1)
    
    # Take the mean of the baseline F0 values to find a overall typical F0
    scaleF0 = np.mean(np.maximum(baselineF0, min_f0_bl), ax_recs, keepdims=True)
    # This should not be less than the declared minimum scale factor
    scaleF0 = np.maximum(scaleF0, min_f0_sf)
    
    # Perform the normalisation by subtracting by the baseline in each
    # recording, and normalising all recordings by a single scalefactor.
    if isinstance(rawF, np.ndarray):
        normalisedF = (rawF - baselineF0) / scaleF0
    else:
        # We were given a list as input, so lets make a list for the output too.
        # We need to drop the final dimension from scaleF0 again so the
        # dimensionality size matches that of the input arrays.
        scaleF0 = np.squeeze(scaleF0, -1)
        # Now normalise each element in the list
        normalisedF = []
        for i, x in enumerate(rawF):
            normalisedF.append((x-bl_list[i])/scaleF0)
        # We should output the baseline values as a list too
        baselineF0 = bl_list
    
    # Return a variable number of outputs
    if output_f0:
        return normalisedF, baselineF0, scaleF0
    else:
        return normalisedF


def findBaselineF0(rawF, fs, axis=0, keepdims=False):
    '''Finds the baseline for a fluorescence imaging trace line.
    
    Parameters
    ----------
    rawF : numpy.ndarray 
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
    '''
    
    # Parameters --------------------------------------------------------------
    nfilt      = 30 # Number of taps to use in FIR filter
    fw_base    =  1 # Cut-off frequency for lowpass filter
    base_pctle =  5 # Percentile to take as baseline value
    
    # Main --------------------------------------------------------------------
    # Remove the first datapoint, because it can be an erroneous sample
    rawF = np.split(rawF, [1], axis)[1]
    
    # The Nyquist rate of the signal is half the sampling frequency
    nyq_rate = fs / 2.0
    
    # Make a set of weights to use with our taps.
    # We use an FIR filter with a Hamming window.
    b = scipy.signal.firwin(nfilt, cutoff=fw_base/nyq_rate, window='hamming')
    
    # Use lfilter to filter with the FIR filter.
    # We filter along the second dimension because that represents time
    filtered_f = scipy.signal.filtfilt(b, [1.0], rawF, axis=axis)
    
    # Take a percentile of the filtered signal
    baselineF0 = np.percentile(filtered_f, base_pctle, axis=axis,
                                keepdims=keepdims)
    
    return baselineF0


def deltaFF0(S, T, avg_n=1):
    ''' Function to interface with commonly used S matrices from rest of 
    fissa easier. 
    
    Removes baseline and calculates deltaF/F0, for normal traces, and 
    uses a simple box kernel to smooth the traces if required. 
    
    Parameters
    ----------
    S : numpy.array
        A 2D array containing all traces across trials for a single cell, 
        and for all signals (somatic + neuropil). Should be of form
        S[frame,signal].
    T : int
        The number of frames per trial.
    avg_n : int, optional
        How many surrounding frames to average over for smoothing. Default
        is 1.
            
    Returns
    -------
    S_norm : numpy.ndarray
        Of same format as Sin
    '''
    # TODO: make this work with different length trials    
    # get some info
    numT = int(S.shape[0]/T) # number of trials
    numR = S.shape[1] # number of regions (cell + neuropils)
    
    # transform S to format expected by df.normaliseF
    traces = np.zeros((numR, T, numT))
    for t in range(numT):
        for n in range(numR):
            traces[n,:, t] = S[t*T:(t+1)*T, n]
    norm, baselineF0, scaleF0 = normaliseF(traces, fs=40, ax_time=1, ax_recs=-1, output_f0=True)
        
    # return to same format as S
    S_norm = np.zeros(S.shape)
    for n in range(numR):
        for t in range(numT):
            S_norm[t*T:(t+1)*T, n] = np.convolve(norm[n,:, t], np.ones(avg_n)/avg_n, mode='same')

    return S_norm


def RemoveBaseline(S, T, avg_n=1):
    ''' Function to interface with commonly used S matrices from rest of 
    fissa easier. 
    
    Removes baseline from every trial, and adds back the global baseline.
    
    Parameters
    ----------
    S : numpy.ndarray
        A 2D array containing all traces across trials for a single cell, 
        and for all signals (somatic + neuropil). Should be of form
        S[frame,signal].
    T : int
        The number of frames per trial.
    avg_n : int, optional
        How many surrounding frames to average over for smoothing. Default
        is 1.
            
    Returns
    -------
    S_norm : numpy.ndarray
        Of same format as Sin
    '''
    # TODO: make this work with different length trials    
    # get some info
    numT = int(S.shape[0]/T) # number of trials
    numR = S.shape[1] # number of regions (cell + neuropils)
    
    # transform S to format expected by df.normaliseF
    traces = np.zeros((numR, T, numT))
    for t in range(numT):
        for n in range(numR):
            traces[n,:, t] = S[t*T:(t+1)*T, n]
    norm, baselineF0, scaleF0 = normaliseF(traces, fs=40, ax_time=1, ax_recs=-1, output_f0=True)
        
    # return to same format as S
    S_norm = np.zeros(S.shape)
    for n in range(numR):
        for t in range(numT):
            S_norm[t*T:(t+1)*T, n] = np.convolve(norm[n,:, t], np.ones(avg_n)/avg_n, mode='same')
            S_norm[t*T:(t+1)*T, n] *= scaleF0[n, 0, 0]
            S_norm[t*T:(t+1)*T, n] += scaleF0[n, 0, 0]

    return S_norm
