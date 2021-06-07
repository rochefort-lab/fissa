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
from PIL import Image, ImageSequence
import imageio

from . import roitools


class DataHandlerAbstract():
    """Abstract class for data handling.

    The main thing to keep consistent is that the input into getmean(), roise2mask(), and extracttraces() depend
    on the output of the image2array and rois2mask.

    See the below DataHandler() and DataHandlerPillow() classes for example usages.
    """
    def __init__(self):
        pass

    def image2array(self, image):
        """Loads an image as an array.

        Parameters
        ----------
        image : type
            Some string/object class/etc.

        Returns
        -------
        type
            Whatever format you want the data to be saved in, and used by the other functions.
        """
        raise NotImplementedError()

    def getmean(self, data):
        """Determine the mean image across all frames.
        Should return a 2D numpy.ndarray.

        Parameters
        ----------
        data : type
            Whatever format is returned by self.image2array().

        Returns
        -------
        numpy.ndarray
            y by x array for the mean values

        """
        raise NotImplementedError()

    def rois2masks(self, rois, data):
        """Take the object `rois` and returns it as a list of binary masks.

        Parameters
        ----------
        rois : type
            However the ROIs are formatted before becoming binary masks.
        data : type
            Whatever format is returned by self.image2array()

        Returns
        -------
        type
            Whatever format is needed in self.extracttraces().
        """
        raise NotImplementedError()

    def extracttraces(self, data, masks):
        """Extracts a temporal trace for each spatial mask.

        Should return a 2D numpy.ndarray.

        Parameters
        ----------
        data : type
            Whatever format is returned by self.image2array().
        masks : type
            Whatever format is returned by self.rois2masks().

        Returns
        -------
        numpy.ndarray
            Trace for each mask. Shaped `(len(masks), n_frames)`.

        """
        raise NotImplementedError()

class DataHandlerTifffile(DataHandlerAbstract):
    """Using tifffile to interact with TIFF images."""

    def image2array(self, image):
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

    def getmean(self, data):
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

    def rois2masks(self, rois, data):
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
        if isinstance(rois, basestring):
            rois = roitools.readrois(rois)

        if not isinstance(rois, collections.Sequence):
            raise TypeError(
                'Wrong ROIs input format: expected a list or sequence, but got'
                ' a {}'.format(rois.__class__)
            )

        # if it's a something by 2 array (or vice versa), assume polygons
        if np.shape(rois[0])[1] == 2 or np.shape(rois[0])[0] == 2:
            return roitools.getmasks(rois, shape)
        # if it's a list of bigger arrays, assume masks
        elif np.shape(rois[0]) == shape:
            return rois

        raise ValueError('Wrong ROIs input format: unfamiliar shape.')

    def extracttraces(self, data, masks):
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

class DataHandlerPillow(DataHandlerAbstract):
    '''Use Pillow to read TIFF files frame-by-frame. Slower but less memory intensive than the DataHandler() class.'''

    def image2array(self, image):
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

    def getmean(self, data):
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

    def rois2masks(self, rois, data):
        """Take the object 'rois' and returns it as a list of binary masks.

        Parameters
        ----------
        rois : str or list of array_like
            Either a string with imagej roi zip location, list of arrays encoding
            polygons, or list of binary arrays representing masks
        data : PIL.Image
            An open PIL.Image handle to a multi-frame TIFF image.

        Returns
        -------
        list
            List of binary arrays (i.e. masks).

        """
        # get the image shape
        shape = data.size[::-1]

        # If rois is string, we first need to read the contents of the file
        if isinstance(rois, basestring):
            rois = roitools.readrois(rois)

        if not isinstance(rois, collections.Sequence):
            raise TypeError(
                'Wrong ROIs input format: expected a list or sequence, but got'
                ' a {}'.format(rois.__class__)
            )

        # if it's a something by 2 array (or vice versa), assume polygons
        if np.shape(rois[0])[1] == 2 or np.shape(rois[0])[0] == 2:
            return roitools.getmasks(rois, shape)
        # if it's a list of bigger arrays, assume masks
        elif np.shape(rois[0]) == shape:
            return rois

        raise ValueError('Wrong ROIs input format: unfamiliar shape.')

    def extracttraces(self, data, masks):
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
