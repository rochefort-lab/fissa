"""
DataHandler classes to handle and manipulate image and ROI objects.

Authors:
    - Sander W Keemink <swkeemink@scimail.eu>
    - Scott C Lowe <scott.code.lowe@gmail.com>
"""

from past.builtins import basestring

try:
    from collections import abc
except ImportError:
    import collections as abc
import warnings

import numpy as np
import tifffile
from PIL import Image, ImageSequence

from . import roitools


class DataHandlerAbstract():
    """
    Abstract class for a data handler.

    Note
    ----
    - The `data` input into :meth:`getmean`, :meth:`rois2masks`, and
      :meth:`extracttraces` must be the same format as the output to
      :meth:`image2array`.
    - The `masks` input into :meth:`extracttraces` must be the same format
      as the output of :meth:`rois2masks`.

    See Also
    --------
    DataHandlerTifffile, DataHandlerPillow
    """
    def __repr__(self):
        return "{}.{}()".format(__name__, self.__class__.__name__)

    @staticmethod
    def image2array(image):
        """
        Load data (from a path) as an array, or similar internal structure.

        Parameters
        ----------
        image : image_type
            Some handle to, or representation of, the raw imagery data.

        Returns
        -------
        data : data_type
            Internal representation of the images which will be used by all
            the other methods in this class.
        """
        raise NotImplementedError()

    @staticmethod
    def getmean(data):
        """
        Determine the mean image across all frames.

        Must return a 2D :class:`numpy.ndarray`.

        Parameters
        ----------
        data : data_type
            The same object as returned by :meth:`image2array`.

        Returns
        -------
        mean : numpy.ndarray
            Mean image as a 2D, y-by-x, array.
        """
        raise NotImplementedError()

    @staticmethod
    def get_frame_size(data):
        """
        Determine the shape of each frame within the recording.

        Parameters
        ----------
        data : data_type
            The same object as returned by :meth:`image2array`.

        Returns
        -------
        shape : tuple of ints
            The 2D, y-by-x, shape of each frame in the movie.
        """
        raise NotImplementedError()

    @classmethod
    def rois2masks(cls, rois, data):
        """
        Convert ROIs into a collection of binary masks.

        Parameters
        ----------
        rois : str or :term:`list` of :term:`array_like`
            Either a string containing a path to an ImageJ roi zip file,
            or a list of arrays encoding polygons, or list of binary arrays
            representing masks.
        data : data_type
            The same object as returned by :meth:`image2array`.

        Returns
        -------
        masks : mask_type
            Masks, in a format accepted by :meth:`extracttraces`.

        See Also
        --------
        fissa.roitools.getmasks, fissa.roitools.readrois
        """
        return roitools.rois2masks(rois, cls.get_frame_size(data))

    @staticmethod
    def extracttraces(data, masks):
        """
        Extract from data the average signal within each mask, across time.

        Must return a 2D :class:`numpy.ndarray`.

        Parameters
        ----------
        data : data_type
            The same object as returned by :meth:`image2array`.
        masks : mask_type
            The same object as returned by :meth:`rois2masks`.

        Returns
        -------
        traces : numpy.ndarray
            Trace for each mask, shaped ``(len(masks), n_frames)``.
        """
        raise NotImplementedError()


class DataHandlerTifffile(DataHandlerAbstract):
    """
    Extract data from TIFF images using tifffile.
    """
    @staticmethod
    def image2array(image):
        """
        Load a TIFF image from disk.

        Parameters
        ----------
        image : str or :term:`array_like` shaped (time, height, width)
            Either a path to a TIFF file, or :term:`array_like` data.

        Returns
        -------
        numpy.ndarray
            A 3D array containing the data, with dimensions corresponding to
            ``(frames, y_coordinate, x_coordinate)``.
        """
        if not isinstance(image, basestring):
            return np.asarray(image)

        with tifffile.TiffFile(image) as tif:
            frames = []
            n_pages = len(tif.pages)
            for page in tif.pages:
                page = page.asarray()
                if page.ndim < 2:
                    raise EnvironmentError(
                        "TIFF {} has pages with {} dimensions (page shaped {})."
                        " Pages must have at least 2 dimensions".format(
                            image, page.ndim, page.shape
                        )
                    )
                if (
                    n_pages > 1 and
                    page.ndim > 2 and
                    (np.array(page.shape[:-2]) > 1).sum() > 0
                ):
                    warnings.warn(
                        "Multipage TIFF {} with {} pages has at least one page"
                        " with {} dimensions (page shaped {})."
                        " All dimensions before the final two (height and"
                        " width) will be treated as time-like and flattened."
                        "".format(
                            image, n_pages, page.ndim, page.shape
                        )
                    )
                elif page.ndim > 3 and (np.array(page.shape[:-2]) > 1).sum() > 1:
                    warnings.warn(
                        "TIFF {} has at least one page with {} dimensions"
                        " (page shaped {})."
                        " All dimensions before the final two (height and"
                        " width) will be treated as time-like and flattened."
                        "".format(
                            image, page.ndim, page.shape
                        )
                    )
                shp = [-1] + list(page.shape[-2:])
                frames.append(page.reshape(shp))

        return np.concatenate(frames, axis=0)

    @staticmethod
    def getmean(data):
        """
        Determine the mean image across all frames.

        Parameters
        ----------
        data : :term:`array_like`
            Data array as made by :meth:`image2array`, shaped ``(frames, y, x)``.

        Returns
        -------
        numpy.ndarray
            y by x array for the mean values
        """
        return data.mean(axis=0, dtype=np.float64)

    @staticmethod
    def get_frame_size(data):
        """
        Determine the shape of each frame within the recording.

        Parameters
        ----------
        data : data_type
            The same object as returned by :meth:`image2array`.

        Returns
        -------
        shape : tuple of ints
            The 2D, y-by-x, shape of each frame in the movie.
        """
        return data.shape[-2:]

    @staticmethod
    def extracttraces(data, masks):
        """
        Extract a temporal trace for each spatial mask.

        Parameters
        ----------
        data : :term:`array_like`
            Data array as made by :meth:`image2array`, shaped ``(frames, y, x)``.
        masks : :class:`list` of :term:`array_like`
            List of binary arrays.

        Returns
        -------
        traces : numpy.ndarray
            Trace for each mask, shaped ``(len(masks), n_frames)``.
        """
        # get the number rois and frames
        nrois = len(masks)
        nframes = data.shape[0]

        # predefine output data
        out = np.zeros((nrois, nframes))

        # loop over masks
        for i in range(nrois):  # for masks
            # get mean data from mask
            out[i, :] = data[:, masks[i]].mean(axis=1, dtype=np.float64)

        return out


class DataHandlerTifffileLazy(DataHandlerAbstract):
    """
    Extract data from TIFF images using tifffile, with lazy loading.
    """

    @staticmethod
    def image2array(image):
        """
        Load a TIFF image from disk.

        Parameters
        ----------
        image : str
            A path to a TIFF file.

        Returns
        -------
        data : tifffile.TiffFile
            Open tifffile.TiffFile object.
        """
        return tifffile.TiffFile(image)

    @staticmethod
    def getmean(data):
        """
        Determine the mean image across all frames.

        Parameters
        ----------
        data : tifffile.TiffFile
            Open tifffile.TiffFile object.

        Returns
        -------
        numpy.ndarray
            y by x array for the mean values
        """
        # We don't load the entire image into memory at once, because
        # it is likely to be rather large.
        memory = None
        n_frames = 0

        n_pages = len(data.pages)
        for page in data.pages:
            page = page.asarray()
            if (
                n_pages > 1 and
                page.ndim > 2 and
                (np.array(page.shape[:-2]) > 1).sum() > 0
            ):
                warnings.warn(
                    "Multipage TIFF {} with {} pages has at least one page"
                    " with {} dimensions (page shaped {})."
                    " All dimensions before the final two (height and"
                    " width) will be treated as time-like and flattened."
                    "".format(
                        "", n_pages, page.ndim, page.shape
                    )
                )
            elif page.ndim > 3 and (np.array(page.shape[:-2]) > 1).sum() > 1:
                warnings.warn(
                    "TIFF {} has at least one page with {} dimensions"
                    " (page shaped {})."
                    " All dimensions before the final two (height and"
                    " width) will be treated as time-like and flattened."
                    "".format(
                        "", page.ndim, page.shape
                    )
                )
            shp = [-1] + list(page.shape[-2:])
            page = page.reshape(shp)
            if memory is None:
                # Initialise holding array with zeros, now we know the shape
                # of the image frames
                memory = np.zeros(page.shape[-2:], dtype=np.float64)
            memory += np.mean(page, dtype=np.float64, axis=0) * page.shape[0]
            n_frames += page.shape[0]

        return memory / n_frames

    @staticmethod
    def get_frame_size(data):
        """
        Determine the shape of each frame within the recording.

        Parameters
        ----------
        data : data_type
            The same object as returned by :meth:`image2array`.

        Returns
        -------
        shape : tuple of ints
            The 2D, y-by-x, shape of each frame in the movie.
        """
        return data.pages[0].shape[-2:]

    @staticmethod
    def extracttraces(data, masks):
        """
        Extract a temporal trace for each spatial mask.

        Parameters
        ----------
        data : tifffile.TiffFile
            Open tifffile.TiffFile object.
        masks : list of array_like
            List of binary arrays.

        Returns
        -------
        traces : numpy.ndarray
            Trace for each mask, shaped ``(len(masks), n_frames)``.
        """
        # Get the number rois
        nrois = len(masks)

        # Initialise output as a list, because we don't know how many frames
        # there will be
        out = []

        # For each frame, get the data
        for page in data.pages:
            page = page.asarray()
            shp = [-1] + list(page.shape[-2:])
            page = page.reshape(shp)

            page_traces = np.zeros((nrois, page.shape[0]), dtype=np.float64)
            for i in range(nrois):
                # Get mean data from mask
                page_traces[i, :] = np.mean(
                    page[..., masks[i]],
                    dtype=np.float64,
                    axis=-1,
                )
            out.append(page_traces)

        out = np.concatenate(out, axis=-1)
        return out


class DataHandlerPillow(DataHandlerAbstract):
    """
    Extract data from TIFF images frame-by-frame using Pillow (:class:`PIL.Image`).

    Slower, but less memory-intensive than :class:`DataHandlerTifffile`.
    """
    @staticmethod
    def image2array(image):
        """
        Open an image file as a :class:`PIL.Image` instance.

        Parameters
        ----------
        image : str or file
            A filename (string) of a TIFF image file, a :class:`pathlib.Path`
            object, or a file object.

        Returns
        -------
        data : PIL.Image
            Handle from which frames can be loaded.
        """
        return Image.open(image)

    @staticmethod
    def getmean(data):
        """
        Determine the mean image across all frames.

        Parameters
        ----------
        data : PIL.Image
            An open :class:`PIL.Image` handle to a multi-frame TIFF image.

        Returns
        -------
        mean : numpy.ndarray
            y-by-x array for the mean values.
        """
        # We don't load the entire image into memory at once, because
        # it is likely to be rather large.
        # Initialise holding array with zeros
        avg = np.zeros(data.size[::-1], dtype=np.float64)

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

    @staticmethod
    def get_frame_size(data):
        """
        Determine the shape of each frame within the recording.

        Parameters
        ----------
        data : PIL.Image
            An open :class:`PIL.Image` handle to a multi-frame TIFF image.

        Returns
        -------
        shape : tuple of ints
            The 2D, y-by-x, shape of each frame in the movie.
        """
        return data.size[::-1]

    @staticmethod
    def extracttraces(data, masks):
        """
        Extract the average signal within each mask across the data.

        Parameters
        ----------
        data : PIL.Image
            An open :class:`PIL.Image` handle to a multi-frame TIFF image.
        masks : list of :term:`array_like`
            List of binary arrays.

        Returns
        -------
        traces : numpy.ndarray
            Trace for each mask, shaped ``(len(masks), n_frames)``.
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
                out[i, f] = np.mean(curframe[masks[i]], dtype=np.float64)

        return out
