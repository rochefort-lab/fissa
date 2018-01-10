"""FISSA functions to handle image and roi objects and return the right format.

If a custom version of this file is used (which can be defined at the
declaration of the core FISSA Experiment class), it should have the same
functions as here, with the same inputs and outputs.

Authors:
    Sander W Keemink <swkeemink@scimail.eu>
    Scott C Lowe <scott.code.lowe@gmail.com>

"""

import numpy as np
import roitools
from PIL import Image, ImageSequence


def image2array(image):
    """Take the object 'image' and returns a pillow image object.

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
        return Image.open(image)

    else:
        raise ValueError('Only tiff locations are accepted as inputs')


def getmean(data):
    """Get the mean image for data.

    Parameters
    ----------
    data : array
        Data array as made by image2array. Should be a pillow image object.

    Returns
    -------
    array
        y by x array for the mean values

    """
    # We don't load the entire image into memory at once, because
    # it is likely to be rather large.
    # Initialise holding array with zeros
    avg = np.zeros(data.size[::-1])

    # Make sure we seek to the first frame before iterating. This is
    # because the Iterator outputs the value for the current frame for
    # `img` first, due to a bug in Pillow<=3.1.
    data.seek(0)

    # Loop over all frames and sum the pixel intensities together
    for frame in ImageSequence.Iterator(data):
        avg += np.asarray(frame)

    # Divide by number of frames to find the average
    avg /= data.n_frames
    return avg


def rois2masks(rois, data):
    """Take the object 'rois' and returns it as a list of binary masks.

    Parameters
    ----------
    rois : unkown
        Either a string with imagej roi zip location, list of arrays encoding
        polygons, or binary arrays representing masks
    data : array
        Data array as made by image2array. Should be a pillow image object.

    Returns
    -------
    list
        List of binary arrays (i.e. masks)

    """
    shape = data.size[::-1]

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
        Data array as made by image2array. Should be a pillow image object.
    masks : list
        list of binary arrays (masks)

    """
    # get the number rois
    nrois = len(masks)

    # get number of frames, and start at zeros
    data.seek(0)
    nframes = data.n_frames

    # predefine array with the data
    out = np.zeros((nrois, nframes))

    # for each frame, get the data
    for f in range(nframes):
        # set frame
        data.seek(f)

        # make numpy array
        curframe = np.asarray(data)

        # loop over masks
        for i in range(nrois):
            # get mean data from mask
            out[i, f] = np.mean(curframe[masks[i]])

    return out
