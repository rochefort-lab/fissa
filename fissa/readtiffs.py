'''
Various functions to read and extract from Tiff stacks using the pillow library

Author: swk, swkeemink@scimail.eu
Initial date: 2015-05-29
Updated:      2015-10-18
'''

from __future__ import division

import numpy as np
from PIL import Image


def get_frame_number(img):
    '''
    Get the number of frames for pillow Image img

    Parameters
    ----------
    img : PIL.Image
        An image loaded using Pillow Image.

    Returns
    -------
    int
        Number of frames in img
    '''
    if hasattr(img, 'n_frames'):
        # Introduced in Pillow 2.9.0
        return img.n_frames
    # Keep going up one frame, until no such frame exists (resulting in
    # EOFEerror)
    while True:
        try:
            img.seek(img.tell() + 1)
        except EOFError:
            break  # end of sequence
    return img.tell() + 1


def getbox(com, size):
    '''
    Gives box coordinates given centre of mass and box size*2,
    formatted so that it can be used to crop images in a pillow Image
    object.

    Parameters
    ----------
    com : tuple, list, or array
        The 2d center of mass the box should have (x,y).
    size : int
        Half the height of the required output box

    Returns
    -------
    numpy.ndarray
        [left border, upper border, right border, lower border]

    Example
    -------
    >>> img = Image.load(filename)
    >>> box = getbox(com,size)
    >>> cropped_img = img.crop(box)
    '''
    return np.array([com[1]-size, com[0]-size, com[1]+size, com[0]+size])


def getavg(img, box, frames):
    '''
    Get the average for the box in pillow Image img, for the specified frames

    Parameters
    ----------
    img : PIL.Image
        Loaded from a tiff stack
    box : array
        Defining which box to get tuple (left, top, right, bottom)
    frames : array
        which frames to get

    Returns
    -------
    numpy.ndarray
        Array of shape of img, which is the averaged image
    '''
    size = box[3] - box[1]

    # where to store average
    avg = np.zeros((size, size))

    for i, f in enumerate(frames):
        # get frame (note: best to now get all relevant data from this frame,
        # so don't have to seek again (although takes neglible time)
        img.seek(f)
        avg[:] += img.crop(box)
    avg = avg / len(frames)

    return avg


def extract_from_single_tiff(filename, masks):
    '''
    Extract the traces from the tiff stack at filename for given rois and
    trials, using Pillow.

    Parameters
    ----------
    filename : string
        Path to source TIFF file.
    masks : dict
        A dictionary of masks, where the keys of each map to a list of
        `numpy.ndarray`s. The keys of the dictionary will be used to
        map to their outputs.
        Each maskset should be a list of binary arrays for each mask in that
        set, with 1's for pixels part of the roi, and 0's if not.

    Returns
    -------
    traces : dict
        A dictionary such that traces[roiset] is data for each roi set
        (cell/noncell + local neuropils).

        out[roiset] is an array such that out[roi][:,mask] gives you the trace
        for the corresponding mask. Usually:
             mask = 0 : somatic roi
             mask > 0 : any defined neuropil rois
    '''
    # Get useful info based on reference image
    img = Image.open(filename)  # pillow loaded reference image

    # Extract the data and put in output dictionary
    # Initiaise an empty dictionary for containing the output
    traces = {}
    # loop over all roi sets
    for label, mask in masks.items():
        # Extract traces for this maskset
        traces[label] = extract_traces(img, mask)

    # Return data
    return traces


def extract_traces(img, masks):
    '''
    Get the traces for each mask in masks from the pillow object img for
    nframes.

    Parameters
    ----------
    img : PIL.Image
        Pillow loaded tiff stack
    masks : list
        list of masks (boolean arrays)

    Returns
    -------
    An array of shape (nrois,nframes), with nrois being the number of ROIs in
    masks, and nframes as above.
    '''
    # TODO: Try loading in entire image into memory (as a memory intensive
    #       alternative) to see if that speeds up the algorithm much.
    #       Would increase memory needs, but could be worth it.

    # get the number rois
    nrois = len(masks)

    # get number of frames
    nframes = img.n_frames

    # predfine list with the data
    data = np.zeros((nrois, nframes))

    # for each frame, get the data
    for f in range(nframes):  # for frames
        # set frame
        img.seek(f)

        # make numpy array
        tempframe = np.array(img)

        for i in range(nrois):  # for masks
            # get mean data from mask
            data[i, f] = np.mean(tempframe[masks[i]])
    return data


def tiff2array(filename):
    '''
    Loads an entire greyscale tiff stack as a single numpy array.

    Parameters
    ----------
    filename : string
        Path to greyscale TIFF file.

    Returns
    -------
    numpy.ndarray
        The contents of the TIFF file, parsed into a three-dimensional
        array of [] shape(tiff), with all frames

    '''
    # define the pillow image
    img = Image.open(filename)

    # get an average image
    nframes = img.n_frames

    # predefine data
    data = np.zeros((img.size[1], img.size[0], nframes))

    # loop over all frames
    for i in range(nframes):
        # switch frame and load into data
        img.seek(i)
        data[:, :, i] = img

    return data


def get_mean_tiff(filename):
    ''' Get the mean data for the tiff stack in filename

    Parameters
    ----------------------
    filename : string
        filename for tiff

    Returns
    ---------------------
    numpy.ndarray
        Array of shape(tiff), averaged across frames
    '''
    # define the pillow image
    img = Image.open(filename)

    # get an average image
    nframes = img.n_frames

    # predefine average
    avg = np.zeros(img.size[::-1])

    # loop over all frames
    for i in range(nframes):
        img.seek(i)
        avg += img
    # divide by number of frames to average
    avg /= nframes

    return avg
