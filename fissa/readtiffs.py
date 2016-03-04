'''
Various functions to read and extract from Tiff stacks using the pillow library

Authors: Sander Keemink (swkeemink@scimail.eu), Scott Lowe
Initial date: 2015-05-29
'''

from __future__ import division

import numpy as np
import roitools as ROI
from PIL import Image, ImageSequence


def image2array(img, bit_depth=None, band=0):
    '''
    Convert a single PIL/Pillow image into a numpy array.

    Parameters
    ----------
    img : PIL.Image
        Source image, loaded with PIL or Pillow.
    bit_depth : int or data-type or None, optional
        The bit depth of the image, also known as bits-per-channel.
        This can be provided as an integer, where `8` and `16` bit
        depth is the most common value. Alternatively, this can be
        a `numpy.dtype` instance, such as `numpy.uint8` or
        `numpy.uint16`, in which case the bit depth is implied by the
        encoding used by the dtype, and the output `numpy.ndarray`
        will have the data-type specified. If this is `None` (default),
        the bit depth will be inferred from the image, however this
        is not 100% reliable.
    band : int, optional
        Which band (color channel) to extract the data from. If `img`
        is greyscale, this must be 0. If `img` is in RGB, BGR or CMYK,
        format, the data will be taken from band number `band`. Default
        is 0.

    Returns
    -------
    numpy.ndarray
        2D output array, sized `(height, width)`, with data type
        `dtype`.

    Raises
    ------
    ValueError
        Requested `band` is larger than the number of available channels
    '''
    # Get the height and width from the size of the image. Note that
    # width is the first and height is the second dimension here.
    (width, height) = img.size
    # Check how many channels the image has, and that we don't try to
    # get data from a channel which doesn't exist.
    num_bands = len(img.getbands())
    if band >= num_bands:
        raise ValueError((
            'Requested band number {} exceeds the number of bands available '
            'in the image, which is {}.'
            ).format(band, num_bands))
    # If there is only one band, we need to request None instead of 0.
    if num_bands == 1:
        band = None
    # And if there isn't, make sure we aren't trying to load the data
    # from multiple channels.
    elif band is None:
        raise ValueError(
            'This function can not load the entire contents of a multi-'
            'channel (i.e. color) image.')

    # If bit depth is known, we need to pick the appropriate (unsigned
    # integer) data-type to use when making the numpy array, so it will
    # read the encoded bits correctly.
    if bit_depth is None:
        pass
    elif bit_depth == 1:
        dtype = bool
    elif bit_depth == 8:
        dtype = np.uint8
    elif bit_depth == 16:
        dtype = np.uint16
    elif bit_depth == 32:
        dtype = np.uint32
    elif bit_depth == 64:
        dtype = np.uint64
    elif isinstance(bit_depth, int):
        raise ValueError((
            'Unfamiliar bit_depth value: {}'
            ).format(bit_depth))
    else:
        # Assume bit_depth is a data-type instance
        dtype = bit_depth

    if bit_depth is not None:
        # Use the `getdata` method to get an iterable sequence which runs
        # through every pixel in the image. Since this is flat, we need
        # to reshape the result. Note here that the width is the second
        # and height is the first dimension in the converted numpy array.
        return np.asarray(img.getdata(band), dtype).reshape((height, width))

    elif num_bands == 1:
        # Let numpy interpret the datastream it gets from img as best it can
        return np.asarray(img)

    else:
        # Let numpy interpret the datastream as best as it can, then cut down
        # the third dimension to only have the relevant band
        return np.asarray(img)[:, :, band]


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
        Number of frames in img.
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


def imgstack2array(img, bit_depth=None, band=0):
    '''
    Loads an entire greyscale tiff stack as a single numpy array.

    Parameters
    ----------
    img : PIL.Image
        An image loaded using Pillow Image.
    bit_depth : int or data-type or None, optional
        The bit depth of the image, also known as bits-per-channel.
        This can be provided as an integer, where `8` and `16` bit
        depth is the most common value. Alternatively, this can be
        a `numpy.dtype` instance, such as `numpy.uint8` or
        `numpy.uint16`, in which case the bit depth is implied by the
        encoding used by the dtype, and the output `numpy.ndarray`
        will have the data-type specified. If this is `None` (default),
        the bit depth will be inferred from the image, however this
        is not 100% reliable.
    band : int, optional
        Which band (color channel) to get extract data from. If `img`
        is greyscale, this must be 0. If `img` is in RGB, BGR or CMYK,
        format, the data will be taken from band number `band`. Default
        is 0.

    Returns
    -------
    numpy.ndarray
        The contents of the TIFF file, parsed into a 3-dimensional
        array, depending on whether the image is
        greyscale or has multiple color channels (a.k.a. bands).
        The output shape is either `(height, width, num_frames)` or
        `(height, width, num_channel, num_frames)` correspondingly.

    See also
    --------
    tiff2array
    '''
    # Get the first frame
    array0 = image2array(img, bit_depth=bit_depth, band=band)
    # From the first frame, we can tell the dtype we need to
    # initialize with, as well as the shape of each frame in the image
    shape = list(array0.shape)
    # We need to add an extra dimension to contain the time series
    shape.append(img.n_frames)
    # Initialise output array
    contents = np.zeros(shape, dtype=array0.dtype)

    # Make sure we seek to the first frame before iterating. This is
    # because the Iterator outputs the value for the current frame for
    # `img` first, due to a bug in Pillow<=3.1.
    img.seek(0)
    # Loop over all frames
    for index, frame in enumerate(ImageSequence.Iterator(img)):
        contents[..., index] = image2array(frame, bit_depth=bit_depth,
                                           band=band)

    return contents


def tiff2array(filename, bit_depth=None, band=0):
    '''
    Loads an entire greyscale tiff stack as a single numpy array.

    Parameters
    ----------
    filename : string
        Path to greyscale TIFF file.
    bit_depth : int or data-type or None, optional
        The bit depth of the image, also known as bits-per-channel.
        This can be provided as an integer, where `8` and `16` bit
        depth is the most common value. Alternatively, this can be
        a `numpy.dtype` instance, such as `numpy.uint8` or
        `numpy.uint16`, in which case the bit depth is implied by the
        encoding used by the dtype, and the output `numpy.ndarray`
        will have the data-type specified. If this is `None` (default),
        the bit depth will be inferred from the image, however this
        is not 100% reliable.
    band : int, optional
        Which band (color channel) to get extract data from. If `img`
        is greyscale, this must be 0. If `img` is in RGB, BGR or CMYK,
        format, the data will be taken from band number `band`. Default
        is 0.

    Returns
    -------
    numpy.ndarray
        The contents of the TIFF file, parsed into a three-dimensional
        array of [] shape(tiff), with all frames

    See also
    --------
    imgstack2array
    '''
    return imgstack2array(Image.open(filename), bit_depth=bit_depth, band=band)


def get_imgstack_mean(img, bit_depth=None, band=0):
    '''
    Get the mean data for an Pillow image stack or animation.

    Parameters
    ----------
    img : PIL.Image
        An animated image loaded using Pillow Image.
    bit_depth : int or data-type or None, optional
        The bit depth of the image, also known as bits-per-channel.
        This can be provided as an integer, where `8` and `16` bit
        depth is the most common value. Alternatively, this can be
        a `numpy.dtype` instance, such as `numpy.uint8` or
        `numpy.uint16`, in which case the bit depth is implied by the
        encoding used by the dtype, and the output `numpy.ndarray`
        will have the data-type specified. If this is `None` (default),
        the bit depth will be inferred from the image, however this
        is not 100% reliable.
    band : int, optional
        Which band (color channel) to get extract data from. If `img`
        is greyscale, this must be 0. If `img` is in RGB, BGR or CMYK,
        format, the data will be taken from band number `band`. Default
        is 0.

    Returns
    -------
    numpy.ndarray
        Average pixel values across all frames in the animation or
        image stack, within the channel numbered `band`.

    See also
    --------
    get_mean_tiff
    '''
    # We don't load the entire image into memory at once, because
    # it is likely to be rather large.
    # Initialise holding array with zeros
    avg = np.zeros(img.size[::-1], dtype=np.float64)

    # Make sure we seek to the first frame before iterating. This is
    # because the Iterator outputs the value for the current frame for
    # `img` first, due to a bug in Pillow<=3.1.
    img.seek(0)
    # Loop over all frames and sum the pixel intensities together
    for frame in ImageSequence.Iterator(img):
        avg += image2array(frame, bit_depth=bit_depth, band=band)

    # Divide by number of frames to find the average
    avg /= img.n_frames
    return avg


def get_mean_tiff(filename, bit_depth=None, band=0):
    '''
    Get the mean frame for a tiff stack.

    Parameters
    ----------
    filename : string
        Path to TIFF file.
    bit_depth : int or data-type or None, optional
        The bit depth of the image, also known as bits-per-channel.
        This can be provided as an integer, where `8` and `16` bit
        depth is the most common value. Alternatively, this can be
        a `numpy.dtype` instance, such as `numpy.uint8` or
        `numpy.uint16`, in which case the bit depth is implied by the
        encoding used by the dtype, and the output `numpy.ndarray`
        will have the data-type specified. If this is `None` (default),
        the bit depth will be inferred from the image, however this
        is not 100% reliable.
    band : int, optional
        Which band (color channel) to get extract data from. If `img`
        is greyscale, this must be 0. If `img` is in RGB, BGR or CMYK,
        format, the data will be taken from band number `band`. Default
        is 0.

    Returns
    -------
    numpy.ndarray
        Average pixel values across all frames in the animation or
        image stack.

    See also
    --------
    get_imgstack_mean
    '''
    return get_imgstack_mean(Image.open(filename), bit_depth=bit_depth,
                             band=band)


def getbox(center, half_length):
    '''
    Find the edge co-ordinates of a square box with given center and
    length.

    Parameters
    ----------
    center : tuple or list
        The center of the box. Should be a two-dimensional tuple with
        each entry corresponding to the location of the center in that
        dimension, `(y, x)`.
    half_length : scalar
        Half the length of the required output box.

    Returns
    -------
    tuple
        Pixel co-ordinates to use when cutting an image down to a
        square of length `2 * half_length`. This output is appropriate
        for using with `PIL.Image.crop` to get the desired output shape.
        Output has elements `(x_0, y_0, x_1, y_1)`, which is
        `(left border, upper border, right border, lower border)`.

    Example
    -------
    >>> img = Image.load(filename)
    >>> box = getbox(center, half_length)
    >>> cropped_img = img.crop(box)
    '''
    half_length = np.rint(2 * half_length) / 2
    x0 = np.ceil(center[1]-half_length)
    y0 = np.ceil(center[0]-half_length)
    x1 = np.rint(x0 + 2*half_length)
    y1 = np.rint(y0 + 2*half_length)
    return (x0, y0, x1, y1)


def getavg(img, box, frame_indices=None, bit_depth=None, band=0):
    '''
    Get the average for the box in pillow Image img, for the specified frames

    Parameters
    ----------
    img : PIL.Image
        Loaded from a tiff stack
    box : tuple
        Box defining which co-ordinates to extract from the image,
        `(left, top, right, bottom)`.
    frame_indices : list or None, optional
        which frames to get. If None, all frames are used
    bit_depth : int or data-type or None, optional
        The bit depth of the image, also known as bits-per-channel.
        This can be provided as an integer, where `8` and `16` bit
        depth is the most common value. Alternatively, this can be
        a `numpy.dtype` instance, such as `numpy.uint8` or
        `numpy.uint16`, in which case the bit depth is implied by the
        encoding used by the dtype, and the output `numpy.ndarray`
        will have the data-type specified. If this is `None` (default),
        the bit depth will be inferred from the image, however this
        is not 100% reliable.
    band : int, optional
        Which band (color channel) to get extract data from. If `img`
        is greyscale, this must be 0. If `img` is in RGB, BGR or CMYK,
        format, the data will be taken from band number `band`. Default
        is 0.

    Returns
    -------
    numpy.ndarray
        Array of shape of img, which is the averaged image
    '''
    if frame_indices is None:
        frame_indices = range(img.n_frames)

    width = box[2] - box[0]
    height = box[3] - box[1]

    # Initialise floating point array to hold running total
    avg = np.zeros((height, width), dtype=np.float64)

    for frame_index in frame_indices:
        # get frame (note: best to now get all relevant data from this frame,
        # so don't have to seek again (although takes neglible time)
        img.seek(frame_index)
        avg[:] += image2array(img.crop(box), bit_depth=bit_depth, band=band)
    avg = avg / len(frame_indices)

    return avg


def extract_from_single_tiff(filename, masksets, bit_depth=None,
                             band=0):
    '''
    Extract the traces from the tiff stack at filename for given rois and
    trials, using Pillow.

    Parameters
    ----------
    filename : string
        Path to source TIFF file.
    masksets : dict of lists of boolean arrays
        A dictionary of masks, where each key maps to a maskset. A
        maskset is a list of boolean arrays masks, with each array
        indicating the spatial extent of a region of interest (ROI)
        within the image - `True` inside the ROI and `False` outside.
    bit_depth : int or data-type or None, optional
        The bit depth of the image, also known as bits-per-channel.
        This can be provided as an integer, where `8` and `16` bit
        depth is the most common value. Alternatively, this can be
        a `numpy.dtype` instance, such as `numpy.uint8` or
        `numpy.uint16`, in which case the bit depth is implied by the
        encoding used by the dtype, and the output `numpy.ndarray`
        will have the data-type specified. If this is `None` (default),
        the bit depth will be inferred from the image, however this
        is not 100% reliable.
    band : int, optional
        Which band (color channel) to get extract data from. If `img`
        is greyscale, this must be 0. If `img` is in RGB, BGR or CMYK,
        format, the data will be taken from band number `band`. Default
        is 0.

    Returns
    -------
    traces : dict of numpy.ndarray
        A dictionary with keys the same as that of `masksets`, each
        mapping to a two-dimensional `numpy.ndarray`, sized
        `(num_masks, num_frames)`, of traces extracted from the pixels
        constituting each mask.
    '''
    img = Image.open(filename)
    # Initiaise an empty dictionary for containing the output
    traces = {}
    # Extract the data and put in output dictionary
    for label, maskset in masksets.items():
        # Extract traces for this maskset
        traces[label] = extract_traces(img, maskset, bit_depth=bit_depth,
                                       band=band)
    return traces


def extract_traces(img, masks, bit_depth=None, band=0):
    '''
    Get the traces for each mask in masks from the pillow object img for
    nframes.

    Parameters
    ----------
    img : PIL.Image
        Pillow loaded tiff stack
    masks : list of boolean arrays
        list of masks (boolean arrays)
    bit_depth : int or data-type or None, optional
        The bit depth of the image, also known as bits-per-channel.
        This can be provided as an integer, where `8` and `16` bit
        depth is the most common value. Alternatively, this can be
        a `numpy.dtype` instance, such as `numpy.uint8` or
        `numpy.uint16`, in which case the bit depth is implied by the
        encoding used by the dtype, and the output `numpy.ndarray`
        will have the data-type specified. If this is `None` (default),
        the bit depth will be inferred from the image, however this
        is not 100% reliable.
    band : int, optional
        Which band (color channel) to get extract data from. If `img`
        is greyscale, this must be 0. If `img` is in RGB, BGR or CMYK,
        format, the data will be taken from band number `band`. Default
        is 0.

    Returns
    -------
    numpy.ndarray
        Traces across frames in image stack `img` for each mask, sized
        `(num_masks, num_frames)`.
    '''
    # TODO: Try loading in entire image into memory (as a memory intensive
    #       alternative) to see if that speeds up the algorithm much.
    #       Would increase memory needs, but could be worth it.
    #       - Early tries seem to indicate it is actually slower

    # Initialise an array predfine list with the data
    traces = np.zeros((len(masks), img.n_frames), dtype=np.float64)

    # Make sure we seek to the first frame before iterating. This is
    # because the Iterator outputs the value for the current frame for
    # `img` first, due to a bug in Pillow<=3.1.
    img.seek(0)
    # For each frame, get the data
    for frame_index, frame in enumerate(ImageSequence.Iterator(img)):
        # Get the pixel values for this frame
        frame_array = image2array(frame, bit_depth=bit_depth, band=band)
        # For each mask, extract the contents for this frame
        for mask_index, mask in enumerate(masks):
            traces[mask_index, frame_index] = np.mean(frame_array[mask])

    return traces


def extract_dataset(tiffs, ROIs,nNpil=4,iterations=15):
    ''' Sets up and saves an entire dataset, given a set of ImageJ ROI zips,
    and a set of TIFF files. Either there should be one zip for every tiff,
    or there should be one zip for all tiffs. 
    
    This will extract the given ROIs' traces, find the surrounding neuropil,
    and extract the neuropil traces.
    
    Every tiff is seen as a trial, from 0 to N as they are ordered in the 
    inputlist 'tiffs'. 
    
    Make sure that the input lists (tiffs and ROIs, see below), are sorted
    as you want them to be sorted (use the python 'sorted()' command to 
    sort them by filename).
    
    Parameters
    ------------------------
    tiffs : list
        A list of strings, listing the locations of every tiff.
    ROIs : list
        A list of strings, listing the location of every ImageJ ROI zip. 
        Should be either len(tiffs) or length 1. In the former case every
        tiff has their own set of ROIs, in the latter every tiff is assumed
        to have the same ROIs. The ROIs are only assumed to have moved across
        trials, there should be the same ROIs (and number thereof) in every 
        zip.
    nNpil : int, optional [4]
        Number of neuropils to be generated.
    iterations : int, optional [15]
        Number of iterations used for growing the neuropil. 
    
    Returns
    -----------------------
    dictionary
        Returns a dictionary S with the following keys:
        'Data'
            This is where the data will be stored. The content itself is a 
            dictionary, with the keys 0 to M being the traces for each cell. 
            A trace from a given cell from a given trial is called as follows:
            S['Data'][ROI number][trial number] = trace
            Where 'trace' is a 2D array containing the input ROI's trace,
            as well as the surrounding neuropil traces, as specified.
        'ROIs'
            This contains the outlines for each ROI.
    
    TODO
    -------------------------
    - Variable trial length for each tiff.
    - Add bit depth etc.
    - Write a test function
    '''
    # find number of cells (needs to be same in every roi set in ROIs!)
    l    = ROIs[0]
    rois = ROI.readrois(l)
    nRois = len(rois)

    # setup data structures where data will be stored
    S = {} # for traces and ROI outlines
    S['Data'] = {} # where traces will be stored
    S['ROIs'] = {} # where ROI outlines will be stored
    # gener
    for m in range(nRois):
        S['Data'][m] = {}
        S['ROIs'][m] = {}

    # loop over the trials
    for i,tiffloc in enumerate(tiffs):
        # get ROI number (checking if there's a single zip, or one for 
        # every trial)
        if len(ROIs) == 1:
            ROIid = 0
        elif len(ROIs) == len(tiffs):
            ROIid = i
        else:
            raise ValueError('Length of ROIs should be either 1' +
                             ' or len(tiffs)')
    
        # load tiff
        img = Image.open(tiffloc) # pillow loaded reference image
        shpe = img.size[::-1]

        # load in basic rois
        rois = ROI.readrois(ROIs[ROIid])

        # calculate masks (can take a while for large ROIs)
        soma_masks = ROI.getmasks(rois, shpe)

        # for each soma mask, calculate the surrounding neuropil rois
        Masks = {} # where to store the masks
        Mask_outlines = {} # where to store mask outlines

        # loop over masks (skipping the first)
        for m,mask in enumerate(soma_masks):
            # get neuropil masks
            npil_masks = ROI.getmasks_npil(mask, nNpil=nNpil,
                                           iterations=iterations)

            # store the masks
            Masks[m] = [mask]
            Masks[m]+= npil_masks

            # calculate the outlines
            Mask_outlines[m] = {}
            for j in range(len(Masks[m])):
                # calculate outline
                outline = ROI.find_roi_edge(Masks[m][j])

                # store the outline coordinates
                Mask_outlines[m][j] = outline

        # extract traces for current trial and rois
        data = extract_from_single_tiff(tiffloc,Masks)

        # loop over rois again
        for m in range(nRois):
            # store traces
            S['Data'][m][i]  = data[m]

            # store mask outlines
            S['ROIs'][m][i] = Mask_outlines[m]

    return S
