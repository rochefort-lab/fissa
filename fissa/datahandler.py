'''
FISSA functions to handle image and roi objects and return the right
format.

Authors:
    Sander W Keemink <swkeemink@scimail.eu>
    Scott C Lowe <scott.code.lowe@gmail.com>

'''

import numpy as np
import tifffile
import roitools

    
    

def image2array(image):
    ''' Takes the object 'image' and returns an array.

    Parameters
    ---------
    image : unknown
        The data. Should be either a tif location, or a list
        of already loaded in data.

    Returns
    -------
    np.array
        A 3D array containing the data as (frames, y coordinate, x coordinate)
    '''
    if type(image) == str:
        return tifffile.imread(image)

    if type(image) == np.array:
        return image


def rois2masks(rois, shape):
    ''' Takes the object 'rois' and returns it as a list of binary masks.

    Parameters
    ----------
    rois : unkown
        Either a string with imagej roi zip location, list of arrays encoding
        polygons, or binary arrays representing masks.
    shape : tuple
        Shape of the original data in x and y coordinates (x,y)
    Returns
    -------
    list
        List of binary arrays (i.e. masks)

    '''
    # if it's a list of strings
    if type(rois) == str:
        rois = roitools.readrois(rois)
    if type(rois) == list:
        # if it's a something by 2 array, assume polygons
        if rois[0].shape[1] == 2:
            return roitools.getmasks(rois, shape)

        # if it's a list bigger arrays, assume masks
        elif rois[0].shape == shape:
            return rois

    else:
        raise ValueError('Wrong rois input format')
