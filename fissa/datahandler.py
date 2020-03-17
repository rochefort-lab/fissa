"""FISSA functions to handle image and roi objects and return the right format.

If a custom version of this file is used (which can be defined at the
declaration of the core FISSA Experiment class), it should have the same
functions as here, with the same inputs and outputs.

Authors:
    - Sander W Keemink <swkeemink@scimail.eu>
    - Scott C Lowe <scott.code.lowe@gmail.com>

"""

from past.builtins import basestring

import collections

import numpy as np
import tifffile

from . import roitools


def image2array(image):
    """Loads a TIFF image from disk.

    Parameters
    ----------
    image : str or array_like
        Either a path to a TIFF file, or array_like data.

    Returns
    -------
    numpy.ndarray
        A 3D array containing the data, with dimensions corresponding to
        `(frames, y_coordinate, x_coordinate)`.

    """
    if isinstance(image, basestring):
        return tifffile.imread(image)

    return np.array(image)


def getmean(data):
    """Determine the mean image across all frames.

    Parameters
    ----------
    data : array_like
        Data array as made by image2array. Should be shaped `(frames, y, x)`.

    Returns
    -------
    numpy.ndarray
        y by x array for the mean values

    """
    return data.mean(axis=0)


def extracttraces(data, masks):
    """Extracts a temporal trace for each spatial mask.

    Parameters
    ----------
    data : array_like
        Data array as made by image2array. Should be shaped
        `(frames, y, x)`.
    masks : list of array_like
        List of binary arrays.

    Returns
    -------
    numpy.ndarray
        Trace for each mask. Shaped `(len(masks), n_frames)`.

    """
    # get the number rois and frames
    nrois = len(masks)
    nframes = data.shape[0]

    # predefine output data
    out = np.zeros((nrois, nframes))

    # loop over masks
    for i in range(nrois):  # for masks
        # get mean data from mask
        out[i, :] = data[:, masks[i]].mean(axis=1)

    return out


# backward compatability
rois2masks = roitools.rois2masks
