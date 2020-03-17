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
from PIL import Image, ImageSequence

from . import roitools


def image2array(image):
    """Open a given image file as a PIL.Image instance.

    Parameters
    ----------
    image : str or file
        A filename (string) of a TIFF image file, a pathlib.Path object,
        or a file object.

    Returns
    -------
    PIL.Image
        Handle from which frames can be loaded.

    """
    return Image.open(image)


def getmean(data):
    """Determine the mean image across all frames.

    Parameters
    ----------
    data : PIL.Image
        An open PIL.Image handle to a multi-frame TIFF image.

    Returns
    -------
    numpy.ndarray
        y-by-x array for the mean values.

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


def extracttraces(data, masks):
    """Get the traces for each mask in masks from data.

    Parameters
    ----------
    data : PIL.Image
        An open PIL.Image handle to a multi-frame TIFF image.
    masks : list of array_like
        List of binary arrays.

    Returns
    -------
    numpy.ndarray
        Trace for each mask. Shaped `(len(masks), n_frames)`.

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


# backward compatability
rois2masks = roitools.rois2masks
