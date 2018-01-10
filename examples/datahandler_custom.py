"""FISSA functions to handle image and roi objects and return the right format.

If a custom version of this file is used (which can be defined at the
declaration of the core FISSA Experiment class), it should have the same
functions as here, with the same inputs and outputs.

Authors:
    Sander W Keemink <swkeemink@scimail.eu>
    Scott C Lowe <scott.code.lowe@gmail.com>

"""

import numpy as np
import tifffile
from fissa import roitools


def image2array(image):
    """Take the object 'image' and returns an array.

    Parameters
    ---------
    image : unknown
        The data. Should be either a tif location, or a list
        of already loaded in data.

    Returns
    -------
    np.array
        A 3D array containing the data as (frames, y coordinate, x coordinate)

    """
    if isinstance(image, str):
        return tifffile.imread(image)

    if isinstance(image, np.ndarray):
        return image


def getmean(data):
    """Get the mean image for data.

    Parameters
    ----------
    data : array
        Data array as made by image2array. Should be of shape [frames,y,x]

    Returns
    -------
    array
        y by x array for the mean values

    """
    return data.mean(axis=0)


def rois2masks(rois, data):
    """Take the object 'rois' and returns it as a list of binary masks.

    Parameters
    ----------
    rois : unkown
        Either a string with imagej roi zip location, list of arrays encoding
        polygons, or binary arrays representing masks
    data : array
        Data array as made by image2array. Should be of shape [frames,y,x]

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
    if isinstance(rois, list):
        # if it's a something by 2 array (or vice versa), assume polygons
        if np.shape(rois[0])[1] == 2 or np.shape(rois[0])[0] == 2:
            return roitools.getmasks(rois, shape)
        # if it's a list of bigger arrays, assume masks
        elif np.shape(rois[0]) == shape:
            return rois

    else:
        raise ValueError('Wrong rois input format')



def extracttraces(data, masks):
    """Get the traces for each mask in masks from data.

    Inputs
    --------------------
    data : array
        Data array as made by image2array. Should be of shape [frames,y,x]
    masks : list
        list of binary arrays (masks)

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
