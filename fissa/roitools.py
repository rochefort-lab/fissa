'''Functions used for ROI manipulation.

Authors:
    Sander W Keemink <swkeemink@scimail.eu>
'''

from __future__ import division
from builtins import range

import numpy as np
from skimage.measure import find_contours

from readimagejrois import read_imagej_roi_zip
from ROI import poly2mask


def get_mask_com(mask):
    '''
    Get the center of mass for a boolean mask.

    Parameters
    ----------
    mask : array_like
        A two-dimensional boolean-mask.

    Returns
    -------
    float
        Center of mass along first dimension.
    float
        Center of mass along second dimension.
    '''
    # Ensure array_like input is a numpy.ndarray
    mask = np.asarray(mask)

    # TODO: make this work for non-boolean masks too
    x, y = mask.nonzero()
    return np.mean(x), np.mean(y)


def split_npil(mask, centre, num_slices, adaptive_num=False):
    '''
    Splits a mask into a number of approximately equal slices by area around
    the center of the mask.

    Parameters
    ----------
    mask : array_like
        Mask as a 2d boolean array.
    centre : tuple
        The center co-ordinates around which the mask will be split.
    num_slices : int
        The number of slices into which the mask will be divided.
    adaptive_num : bool, optional
        If True, the `num_slices` input is treated as the number of
        slices to use if the ROI is surrounded by valid pixels, and
        automatically reduces the number of slices if it is on the
        boundary of the sampled region. NOT IMPLEMENTED.

    Returns
    -------
    dict
        A dictionary with `num_slices` many masks, each of which is a 2d
        boolean numpy array.

    Notes
    -----
    This should be an iterable.
    '''
    # Ensure array_like input is a numpy.ndarray
    mask = np.asarray(mask)

    # Get the (x,y) co-ordinates of the pixels in the mask
    x, y = mask.nonzero()

    # Find the angle of the vector from the mask centre to each pixel
    theta = np.arctan2(x - centre[0], y - centre[1])

    # Find where the mask comes closest to the centre. We will put a
    # slice boundary here, to prevent one slice being non-contiguous
    # for masks near the image boundary.
    # TODO: give it the bins to use
    n_bins = 20
    bins = np.linspace(-np.pi, np.pi, n_bins + 1)
    bin_counts, bins = np.histogram(theta, bins=bins)
    bin_min_index = np.argmin(bin_counts)

    if adaptive_num:
        # Change the number of slices we will used based on the
        # proportion of these bins which are empty
        num_slices = round(num_slices * sum(bin_counts > 0) / n_bins)

    # Change theta so it is the angle relative to a new zero-point,
    # the middle of the bin which is least populated by mask pixels.
    theta_offset = bins[bin_min_index] + np.pi / n_bins
    theta = (theta - theta_offset) % (2 * np.pi) - np.pi

    # get the boundaries
    bounds = [np.percentile(theta, 100.0 * (i + 1) / num_slices)
              for i in range(num_slices)]

    # predefine the masks
    masks = []
    # get the first mask
    # empty predefinition
    masks += [np.zeros(np.shape(mask), dtype=bool)]
    # set relevant pixels to True
    masks[0][x[theta <= bounds[0]], y[theta <= bounds[0]]] = True
    # get the rest of the masks
    for i in range(1, num_slices):
        # find which pixels are within bounds
        truths = (theta > bounds[i - 1]) * (theta <= bounds[i])
        # empty predefinition
        masks += [np.zeros(np.shape(mask), dtype=bool)]
        # set relevant pixels to True
        masks[i][x[truths], y[truths]] = True

    return masks


def shift_2d_array(a, shift=1, axis=None):
    '''
    Shifts an entire array in the direction of axis by the amount shift,
    without refilling the array.

    Uses numpy.roll the shift, then empties the refilled parts of the array.

    Parameters
    ----------
    a : array_like
        input array
    shift : int
        how much to shift array by
    axis : int
        From numpy.roll doc:
        The axis along which elements are shifted.
        By default, the array is flattened before shifting,
        after which the original shape is restored.

    Returns
    -------
    Array of same shape as a, but shifted as per above
    '''
    # Ensure array_like input is a numpy.ndarray
    a = np.asarray(a)

    # do initial shift
    out = np.roll(a, shift, axis)

    # then fill in refilled parts of the array
    if axis == 0:
        if shift > 0:
            out[:shift, :] = 0
        elif shift < 0:
            out[shift:, :] = 0
    elif axis == 1:
        if shift > 0:
            out[:, :shift] = 0
        elif shift < 0:
            out[:, shift:] = 0

    # return shifted array
    return out


def get_npil_mask(mask, totalexpansion=4):
    '''
    Given the masks for a ROI, find the surrounding neuropil.

    Parameters
    ----------
    mask : array_like
        The reference ROI mask to expand the neuropil from. The array
        should contain only boolean values.
    expansion : float, optional
        How much larger to make the neuropil total area than mask area.

    Returns
    -------
    array
        A boolean numpy.ndarray mask, where the region surrounding
        the input is now True and the region of the input mask is
        False.

    Implementation
    --------------
    Our implementation is as follows:
        - On even iterations (where indexing begins at zero), expand
          the mask in each of the 4 cardinal directions.
        - On odd numbered iterations, expand the mask in each of the 4
          diagonal directions.
    This procedure generates a neuropil whose shape is similar to the
    shape of the input ROI mask.

    Notes
    -----
    For fixed number of `iterations`, more square input masks will have
    larger output neuropil masks.
    '''
    # Ensure array_like input is a numpy.ndarray
    mask = np.asarray(mask)

    # Make a copy of original mask which will be grown
    grown_mask = np.copy(mask)
    area_orig = grown_mask.sum()  # original area
    area_current = 0  # current size
    shpe = np.shape(mask)
    area_total = shpe[0] * shpe[1]
    count = 0

    # for count in range(iterations):
    while area_current < totalexpansion * \
            area_orig and area_current < area_total - area_orig:
        # Check which case to use. In current version, we alternate
        # between case 0 (cardinals) and case 1 (diagonals).
        case = count % 2

        # Make a copy of the mask without any new additions. We will
        # need to keep using this mask to mark new changes, so we
        # don't use a partially updated version.
        refmask = np.copy(grown_mask)

        if case == 2:
            # Move polygon around one pixel in each 8 directions
            # N, NE, E, SE, S, SW, W, NW, (the centre is also redone)
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    movedmask = shift_2d_array(refmask, dx, 0)
                    movedmask = shift_2d_array(movedmask, dy, 1)
                    grown_mask[movedmask] = True

        elif case == 0:
            # Move polygon around one pixel in each of the 4 cardinal
            # directions: N, E, S, W.
            for dx in [-1, 1]:
                grown_mask[shift_2d_array(refmask, dx, 0)] = True
            for dy in [-1, 1]:
                grown_mask[shift_2d_array(refmask, dy, 1)] = True

        elif case == 1:
            # Move polygon around one pixel in each of the 4 diagonal
            # directions: NE, SE, SW, NW
            for dx in [-1, 1]:
                for dy in [-1, 1]:
                    movedmask = shift_2d_array(refmask, dx, 0)
                    movedmask = shift_2d_array(movedmask, dy, 1)
                    grown_mask[movedmask] = True

        # Don't expand based on the original mask; any expansion into
        # this region is marked as False once more.
        grown_mask[mask] = False

        # update area
        area_current = grown_mask.sum()

        # iterate counter
        count += 1

    # Return the finished neuropil mask
    return grown_mask


def getmasks_npil(cellMask, nNpil=4, expansion=1):
    '''Generate neuropil masks using the get_npil_mask function.

    Parameters
    ----------
    cellMask : array_like
        the cell mask (boolean 2d arrays)
    nNpil : int
        number of neuropil subregions
    expansion : float
        How much larger to make neuropil subregion area than cellMask's

    Returns
    -------
    Returns a list with soma + neuropil masks (boolean 2d arrays)
    '''
    # Ensure array_like input is a numpy.ndarray
    cellMask = np.asarray(cellMask)

    # get the total neuropil for this cell
    mask = get_npil_mask(cellMask, totalexpansion=expansion * nNpil)

    # get the center of mass for the cell
    centre = get_mask_com(cellMask)

    # split it up in nNpil neuropils
    masks_split = split_npil(mask, centre, nNpil)

    return masks_split


def readrois(roiset):
    ''' read the imagej rois in the zipfile roiset, and make sure that the
    third dimension (i.e. frame number) is always zero.

    Parameters
    ----------
    roiset : string
        folder to a zip file with rois

    Returns
    -------
    Returns the rois as polygons
    '''
    # read rois
    rois = read_imagej_roi_zip(roiset)

    # set frame number to 0 for every roi
    for i in range(len(rois)):
        # check if we are looking at an oval roi
        if rois[i].keys()[0] == 'mask':
            # this is an oval roi, which gets imported as a 3D mask.
            # First get the frame that has the mask in it by finding the
            # nonzero frame
            mask_frame = np.nonzero(rois[i]['mask'])[0][0]

            # get the mask
            mask = rois[i]['mask'][mask_frame, :, :]

            # finally, get the outline coordinates
            rois[i] = find_roi_edge(mask)[0]
        else:
            rois[i] = rois[i]['polygons'][:, :2]

    return rois


def getmasks(rois, shpe):
    '''Get the masks for the specified rois.

    Parameters
    ----------
    rois : list
        list of roi coordinates. Each roi coordinate should be a 2d-array
        or equivalent list. I.e.:
        roi = [[0,0], [0,1], [1,1], [1,0]]
        or
        roi = np.array([[0,0], [0,1], [1,1], [1,0]])
        I.e. a n by 2 array, where n is the number of coordinates.
        If a 2 by n array is given, this will be transposed.
    shpe : array/list
        shape of underlying image [width,height]

    Returns
    -------
    List of masks for each roi in the rois list
    '''
    # get number of rois
    nrois = len(rois)

    # start empty mask list
    masks = [''] * nrois

    for i in range(nrois):
        # transpose if array of 2 by n
        if np.asarray(rois[i]).shape[0] == 2:
            rois[i] = np.asarray(rois[i]).T

        # transform current roi to mask
        mask = poly2mask(rois[i], shpe)
        # store in list
        masks[i] = np.array(mask[0].todense())

    return masks


def find_roi_edge(mask):
    '''
    Finds the outline of a mask, using the find_contour function from
    skimage.measure.

    Parameters
    ----------
    mask : array_like
        the mask, a binary array

    Returns
    -------
    Array with coordinates of pixels in the outline of the mask
    '''

    # Ensure array_like input is a numpy.ndarray
    mask = np.asarray(mask)

    # Pad with 0s to make sure that edge ROIs are properly estimated
    mask_shape = np.shape(mask)
    padded_shape = (mask_shape[0] + 2, mask_shape[1] + 2)
    padded_mask = np.zeros(padded_shape)
    padded_mask[1:-1, 1:-1] = mask

    # detect contours
    outline = find_contours(padded_mask, level=0.5)

    # update coordinates to take into account padding and set so that the
    # coordinates are defined from the corners (as in the mask2poly function
    # in SIMA https://github.com/losonczylab/sima/blob/master/sima/ROI.py)
    for i in range(len(outline)):
        outline[i] -= 0.5

    return outline
