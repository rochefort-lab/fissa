'''
Functions used for ROI manipulation.

Author: S W Keemink swkeemink@scimail.eu
'''

from builtins import range

import numpy as np

from sima_borrowed.readimagejrois import read_imagej_roi_zip
from sima_borrowed.ROI import poly2mask

from skimage.measure import find_contours


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


def split_npil(mask, com, num_slices):
    '''
    Splits a mask into a number of approximately equal slices by area around
    the center of the mask.

    Parameters
    ----------
    mask : array_like
        Mask as a 2d boolean array.
    com : tuple
        The center co-ordinates around which the mask will be split.
    num_slices : int
        The number of slices into which the mask will be divided.

    Returns
    -------
    dict
        A dictionary with `num_slices` many masks, each of which is a 2d
        boolean numpy array.
    '''
    # Ensure array_like input is a numpy.ndarray
    mask = np.asarray(mask)

    # get the percentage for each slice
    slice_perc = 100.0/num_slices

    # get the x,y positions of the pixels that are in the mask
    x, y = mask.nonzero()

    # find the positional angle of each point
    theta = np.arctan2(x-com[0], y-com[1])

    # find smallest bin
    # TODO: give it the bins to use
    bins = np.linspace(-np.pi, np.pi, 21)
    n, bins = np.histogram(theta, bins=bins)
    binmin = np.argmin(n)

    # adjust angles so that they start at the smallest bin
    nmin = bins[binmin]+np.pi/40
    theta = (theta-nmin) % (2*np.pi)-np.pi

    # get the boundaries
    bounds = {}
    for i in range(num_slices):
        bounds[i] = np.percentile(theta, slice_perc*(i+1))

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
        truths = (theta > bounds[i-1])*(theta <= bounds[i])
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


def get_npil_mask(mask, iterations=15):
    '''
    Given the masks for cell rois, find the surround neuropil as follows:
        for all iterations
            move original roi either
                - move polygon around one pixel in each 4 cardinal directions
                - move polygon around one pixel in each 4 diagonal directions
            Fill in all overlapped pixels
    This will generate a neuropil close the the roi shape (more square, the
    bigger the neuropil).

    Parameters
    ----------
    mask : array_like
        the reference mask to expand the neuropil from
    iterations : int
        number of iterations for neuropil

    Returns
    -------
    A dictionary with a boolean 2d array containing the neuropil mask for each
    iteration
    '''
    # Ensure array_like input is a numpy.ndarray
    mask = np.asarray(mask)

    # initate masks
    masks = {}
    masks[0] = np.copy(mask)    # initial mask

    # keep adding area until enough is added
    for count in range(iterations):
        # get reference mask
        refmask = np.copy(masks[count])

        # initiate next mask
        masks[count+1] = np.copy(refmask)

        # define case, it swaps between 0 and 1
        case = count % 2

        if case == 2:
            # move polygon around one pixel in each 8 directions
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    movedmask = shift_2d_array(refmask, dx, 0)
                    movedmask = shift_2d_array(movedmask, dy, 1)
                    masks[count+1][movedmask] = True
        elif case == 0:
            # move polygon around one pixel in each 4 cardinal direction
            for dx in [-1, 1]:
                movedmask = shift_2d_array(refmask, dx, 0)
                masks[count+1][movedmask] = True
            for dy in [-1, 1]:
                movedmask = shift_2d_array(refmask, dy, 1)
                masks[count+1][movedmask] = True
        elif case == 1:
            # move polygon around one pixel in each 4 diagonal direction
            for dx in [-1, 1]:
                for dy in [-1, 1]:
                    movedmask = shift_2d_array(refmask, dx, 0)
                    movedmask = shift_2d_array(movedmask, dy, 1)
                    masks[count+1][movedmask] = True

        masks[count+1][mask] = False

    masks[0][:] = False

    # return the masks
    return masks


def getmasks_npil(cellMask, nNpil=4, iterations=15):
    '''
    Generates neuropil masks using the get_npil_mask function.

    Parameters
    ----------
    cellMask : array_like
        the cell mask (boolean 2d arrays)
    nNpil : int
        number of neuropils
    iterations : int
        number of iterations for neuropil expansion

    Returns
    -------
    Returns a list with soma + neuropil masks (boolean 2d arrays)
    '''
    # Ensure array_like input is a numpy.ndarray
    cellMask = np.asarray(cellMask)

    # get the total neuropil for this cell
    mask = get_npil_mask(cellMask, iterations=iterations)[iterations]

    # get the center of mass for the cell
    com = get_mask_com(cellMask)

    # split it up in nNpil neuropils
    masks_split = split_npil(mask, com, nNpil)

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
    ''' get the masks for the specified rois

    Parameters
    ----------
    rois : list
        list of roi coordinates. Each roi coordinate should be a 2d-array
        or equivalent list. I.e.:
        roi = [[0,0], [0,1], [1,1], [1,0]]
        or
        roi = np.array([[0,0], [0,1], [1,1], [1,0]])
    shpe : array/list
        shape of underlying image [width,height]

    Returns
    -------
    List of masks for each roi in the rois list
    '''
    # get number of rois
    nrois = len(rois)

    # start empty mask list
    masks = ['']*nrois

    for i in range(nrois):
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
    padded_shape = (mask_shape[0]+2, mask_shape[1]+2)
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
