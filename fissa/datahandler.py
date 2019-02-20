"""FISSA functions to handle image and roi objects and return the right format.

If a custom version of this file is used (which can be defined at the
declaration of the core FISSA Experiment class), it should have the same
functions as here, with the same inputs and outputs.

Authors:
    - Sander W Keemink <swkeemink@scimail.eu>
    - Scott C Lowe <scott.code.lowe@gmail.com>

"""

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
    if isinstance(image, str):
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


def rois2masks(rois, data):
    """Take the object `rois` and returns it as a list of binary masks.

    Parameters
    ----------
    rois : string or list of array_like
        Either a string with imagej roi zip location, list of arrays encoding
        polygons, or list of binary arrays representing masks
    data : array
        Data array as made by image2array. Must be shaped `(frames, y, x)`.

    Returns
    -------
    list
        List of binary arrays (i.e. masks)

    """
    # get the image shape
    shape = data.shape[1:]

    # if it's a list of strings
    if isinstance(rois, str):
        rois = roitools.readrois(rois)

    if not isinstance(rois, list):
        raise ValueError('Wrong ROIs input format: expected a list.')

    # if it's a something by 2 array (or vice versa), assume polygons
    if np.shape(rois[0])[1] == 2 or np.shape(rois[0])[0] == 2:
        return roitools.getmasks(rois, shape)
    # if it's a list of bigger arrays, assume masks
    elif np.shape(rois[0]) == shape:
        return rois

    raise ValueError('Wrong ROIs input format: unfamiliar shape.')


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
