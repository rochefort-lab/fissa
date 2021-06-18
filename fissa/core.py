# -*- coding: utf-8 -*-
"""
Main FISSA user interface.

Authors:
    - Sander W Keemink <swkeemink@scimail.eu>
    - Scott C Lowe <scott.code.lowe@gmail.com>
"""

from __future__ import print_function

import collections
import functools
import glob
import itertools
import multiprocessing
import os.path
import sys
import warnings

from past.builtins import basestring

try:
    from collections import abc
except ImportError:
    import collections as abc

import numpy as np
from scipy.io import savemat

from . import deltaf, extraction
from . import neuropil as npil
from . import roitools


def extract(image, rois, nRegions=4, expansion=1, datahandler=None):
    r"""
    Extract data for all ROIs in a single 3d array or TIFF file.

    .. versionadded:: 1.0.0

    Parameters
    ----------
    image : str or :term:`array_like` shaped (time, height, width)
        Either a path to a multipage TIFF file, or 3d :term:`array_like` data.
    rois : str or :term:`list` of :term:`array_like`
        Either a string containing a path to an ImageJ roi zip file,
        or a list of arrays encoding polygons, or list of binary arrays
        representing masks.
    nRegions : int, default=4
        Number of neuropil regions to draw. Use a higher number for
        densely labelled tissue. Default is ``4``.
    expansion : float, default=1
        Expansion factor for the neuropil region, relative to the
        ROI area. Default is ``1``. The total neuropil area will be
        ``nRegions * expansion * area(ROI)``.
    datahandler : fissa.extraction.DataHandlerAbstract, optional
        A datahandler object for handling ROIs and calcium data.
        The default is :class:`~fissa.extraction.DataHandlerTifffile`.

    Returns
    -------
    data : dict
        Data across cells.
    roi_polys : dict
        Polygons for each ROI.
    mean : :class:`numpy.ndarray` shaped (height, width)
        Mean image.
    """
    if datahandler is None:
        datahandler = extraction.DataHandlerTifffile()

    # get data as arrays and rois as masks
    curdata = datahandler.image2array(image)
    base_masks = datahandler.rois2masks(rois, curdata)

    # get the mean image
    mean = datahandler.getmean(curdata)

    # predefine dictionaries
    data = collections.OrderedDict()
    roi_polys = collections.OrderedDict()

    # get neuropil masks and extract signals
    for cell in range(len(base_masks)):
        # neuropil masks
        npil_masks = roitools.getmasks_npil(
            base_masks[cell], nNpil=nRegions, expansion=expansion
        )
        # add all current masks together
        masks = [base_masks[cell]] + npil_masks

        # extract traces
        data[cell] = datahandler.extracttraces(curdata, masks)

        # store ROI outlines
        roi_polys[cell] = [roitools.find_roi_edge(mask) for mask in masks]

    return data, roi_polys, mean


def separate_trials(raw, roi_label=None, alpha=0.1, method="nmf"):
    r"""
    Separate signals within a set of 2d arrays.

    .. versionadded:: 1.0.0

    Parameters
    ----------
    raw : list of n_trials :term:`array_like`, each shaped (nRegions, observations)
        Raw signals.
        A list of 2-d arrays, each of which contains observations of mixed
        signals, mixed in the same way across all trials.
        The `nRegions` signals must be the same for each trial, and the 0-th
        region, ``raw[trial][0]``, should be from the region of interest for
        which a matching source signal should be identified.

    roi_label : str or int, optional
        Label/name or index of the ROI currently being processed.
        Only used for progress messages.

    alpha : float, default=0.1
        Sparsity regularizaton weight for NMF algorithm. Set to zero to
        remove regularization. Default is ``0.1``.
        (Only used for ``method="nmf"``.)

    method : {"nmf", "ica"}, default="nmf"
        Which blind source-separation method to use. Either ``"nmf"``
        for non-negative matrix factorization, or ``"ica"`` for
        independent component analysis. Default is ``"nmf"``.

    Returns
    -------
    Xsep : list of n_trials :class:`numpy.ndarray`, each shaped (nRegions, observations)
        The separated signals, unordered.

    Xmatch : list of n_trials :class:`numpy.ndarray`, each shaped (nRegions, observations)
        The separated traces, ordered by matching score against the raw ROI
        signal.

    Xmixmat : :class:`numpy.ndarray`, shaped (nRegions, nRegions)
        Mixing matrix.

    convergence : dict
        Metadata for the convergence result, with the following keys and
        values:

        converged : bool
            Whether the separation model converged, or if it ended due to
            reaching the maximum number of iterations.
        iterations : int
            The number of iterations which were needed for the separation model
            to converge.
        max_iterations : int
            Maximum number of iterations to use when fitting the
            separation model.
        random_state : int or None
            Random seed used to initialise the separation model.
    """
    # Join together the raw data across trials, collapsing down the trials
    X = np.concatenate(raw, axis=1)

    # Check for values below 0
    if X.min() < 0:
        message_extra = ""
        if roi_label is not None:
            message_extra = " for ROI {}".format(roi_label)
        warnings.warn(
            "Found values below zero in raw signal{}. Offsetting so minimum is 0."
            "".format(message_extra)
        )
        X -= X.min()

    # Separate the signals
    Xsep, Xmatch, Xmixmat, convergence = npil.separate(
        X, method, maxiter=20000, tol=1e-4, maxtries=1, alpha=alpha
    )
    # Unravel observations from multiple trials into a list of arrays
    trial_lengths = [r.shape[1] for r in raw]
    indices = np.cumsum(trial_lengths[:-1])
    Xsep = np.split(Xsep, indices, axis=1)
    Xmatch = np.split(Xmatch, indices, axis=1)
    # Report status
    message = "Finished separating ROI"
    if roi_label is not None:
        message += " number {}".format(roi_label)
    print(message)
    return Xsep, Xmatch, Xmixmat, convergence


if sys.version_info < (3, 0):
    # Define helper functions which are needed on Python 2.7, which does not
    # have multiprocessing.Pool.starmap.

    def _extract_wrapper(args):
        return extract(*args)

    def _separate_wrapper(args):
        return separate_trials(*args)


class Experiment():
    r"""
    FISSA Experiment.

    Uses the methodology described in
    `FISSA: A neuropil decontamination toolbox for calcium imaging signals <doi_>`_.

    .. _doi: https://www.doi.org/10.1038/s41598-018-21640-2

    Parameters
    ----------
    images : str or list
        The raw recording data.
        Should be one of:

        - the path to a directory containing TIFF files (string),
        - a list of paths to TIFF files (list of strings),
        - a list of :term:`array_like` data already loaded into memory,
          each shaped ``(n_frames, height, width)``.

        Note that each TIFF or array is considered a single trial.

    rois : str or list
        The region of interest (ROI) definitions.
        Should be one of:

        - the path to a directory containing ImageJ ZIP files (string),
        - the path of a single ImageJ ZIP file (string),
        - a list of ImageJ ZIP files (list of strings),
        - a list of arrays, each encoding a ROI polygons,
        - a list of lists of binary arrays, each representing a ROI mask.

        This can either be a single roiset for all trials, or a different
        roiset for each trial.

    folder : str, optional
        Path to a cache directory from which pre-extracted data will
        be loaded if present, and saved to otherwise. If `folder` is
        unset, the experiment data will not be saved.

    nRegions : int, default=4
        Number of neuropil regions and signals to use. Default is ``4``.
        Use a higher number for densely labelled tissue.

    expansion : float, default=1
        Expansion factor for each neuropil region, relative to the
        ROI area. Default is ``1``. The total neuropil area will be
        ``nRegions * expansion * area(ROI)``.

    alpha : float, default=0.1
        Sparsity regularizaton weight for NMF algorithm. Set to zero to
        remove regularization. Default is ``0.1``.

    ncores_preparation : int or None, default=None
        The number of parallel subprocesses to use during the data
        preparation steps of :meth:`separation_prep`.
        These are ROI and neuropil subregion definitions, and extracting
        raw signals from TIFFs.

        If set to ``None`` (default), the number of processes used will
        equal the number of threads on the machine. Note that this
        behaviour can, especially for the data preparation step,
        be very memory-intensive.

    ncores_separation : int or None, default=None
        The number of parallel subprocesses to use during the signal
        separation steps of :meth:`separate`.
        The separation steps requires less memory per subprocess than
        the preparation steps, and so can be often be set higher than
        `ncores_preparation`.

        If set to ``None`` (default), the number of processes used will
        equal the number of threads on the machine. Note that this
        behaviour can, especially for the data preparation step,
        be very memory-intensive.

    method : "nmf" or "ica", default="nmf"
        Which blind source-separation method to use. Either ``"nmf"``
        for non-negative matrix factorization, or ``"ica"`` for
        independent component analysis. Default is ``"nmf"`` (recommended).

    lowmemory_mode : bool, optional
        If ``True``, FISSA will load TIFF files into memory frame-by-frame
        instead of holding the entire TIFF in memory at once. This
        option reduces the memory load, and may be necessary for very
        large inputs. Default is ``False``.

    datahandler : :class:`extraction.DataHandlerAbstract`, optional
        A custom datahandler object for handling ROIs and calcium data can
        be given here. See :mod:`fissa.extraction` for example datahandler
        classes. The default datahandler is
        :class:`~extraction.DataHandlerTifffile`.
        If `datahandler` is set, the `lowmemory_mode` parameter is
        ignored.

    Attributes
    ----------
    result : :class:`numpy.ndarray`
        A :class:`numpy.ndarray` of shape ``(n_rois, n_trials)``, each element
        of which is itself a :class:`numpy.ndarray` shaped
        ``(n_signals, n_timepoints)``.

        The final output of FISSA, with separated signals ranked in order of
        their weighting toward the raw cell ROI signal relative to their
        weighting toward other mixed raw signals.
        The ordering is such that ``experiment.result[cell][trial][0, :]``
        is the signal with highest score in its contribution to the raw cell
        signal. Subsequent signals are sorted in order of diminishing score.
        The units are same as `raw` (candelas per unit area).

        This field is only populated after :meth:`separate` has been run; until
        then, it is set to ``None``.

    roi_polys : :class:`numpy.ndarray`
        A :class:`numpy.ndarray` of shape ``(n_rois, n_trials)``, each element
        of which is itself a list of length ``nRegions + 1``, each element of
        which is a list of length ``1``, containing a :class:`numpy.ndarray`
        of shape ``(n_nodes, 2)``.

        The nodes describe the polygon outline of each region as ``(y, x)``
        points.
        The outline of a ROI is given by
        ``experiment.roi_polys[i_roi][i_trial][0][0]``,
        and the :attr:`nRegions` neuropil regions by
        ``experiment.roi_polys[i_roi][i_trial][1 + i_region][0]``.

    means : list of n_trials :class:`numpy.ndarray`s, each shaped ``(height, width)``
        The temporal-mean image for each trial (i.e. for each TIFF file,
        the average image over all of its frames).

    raw : :class:`numpy.ndarray`
        A :class:`numpy.ndarray` of shape ``(n_rois, n_trials)``, each element
        of which is itself a :class:`numpy.ndarray` shaped
        ``(n_signals, n_timepoints)``.

        For each ROI and trial (``raw[i_roi, i_trial]``) we extract a temporal
        trace of the average value within the spatial area of each of the
        ``nRegions + 1`` regions.
        The 0-th region is the ``i_roi``-th ROI (``raw[i_roi, i_trial][0]``).
        The subsequent ``nRegions`` vectors are the traces for each of the
        neuropil regions.

        The units are the same as the supplied imagery (candelas per unit
        area).

    sep : :class:`numpy.ndarray`
        A :class:`numpy.ndarray` of shape ``(n_rois, n_trials)``, each element
        of which is itself a :class:`numpy.ndarray` shaped
        ``(n_signals, n_timepoints)``.

        The separated signals, before output signals are ranked according to
        their matching against the raw signal from within the ROI.
        Separated signal ``i`` for a specific cell and trial can be found at
        ``experiment.sep[cell][trial][i, :]``.

        This field is only populated after :meth:`separate` has been run; until
        then, it is set to ``None``.

    mixmat : :class:`numpy.ndarray`
        A :class:`numpy.ndarray` of shape ``(n_rois, n_trials)``, each element
        of which is itself a :class:`numpy.ndarray` shaped
        ``(n_rois, n_signals)``.

        The mixing matrix, which maps from ``experiment.raw`` to
        ``experiment.sep``.
        Because we use the collate the traces from all trials to determine
        separate the signals, the mixing matrices for a given ROI are the
        same across all trials.
        This means all ``n_trials`` elements in ``mixmat[i_roi, :]`` are
        identical.

        This field is only populated after :meth:`separate` has been run; until
        then, it is set to ``None``.

    info : :class:`numpy.ndarray` shaped ``(n_rois, n_trials)`` of dicts
        Information about the separation routine.

        Each dictionary in the array has the following fields:

        converged : bool
            Whether the separation model converged, or if it ended due to
            reaching the maximum number of iterations.
        iterations : int
            The number of iterations which were needed for the separation model
            to converge.
        max_iterations : int
            Maximum number of iterations to use when fitting the
            separation model.
        random_state : int or None
            Random seed used to initialise the separation model.

        This field is only populated after :meth:`separate` has been run; until
        then, it is set to ``None``.

    deltaf_raw : :class:`numpy.ndarray`
        A :class:`numpy.ndarray` of shape ``(n_rois, n_trials)``, each element
        of which is itself a :class:`numpy.ndarray` shaped ``(n_timepoint, )``.

        The amount of change in fluorence relative to the baseline fluorence
        (Δf/f\ :sub:`0`).

        This field is only populated after :meth:`calc_deltaf` has been run;
        until then, it is set to ``None``.

    deltaf_result : :class:`numpy.ndarray`
        A :class:`numpy.ndarray` of shape ``(n_rois, n_trials)``, each element
        of which is itself a :class:`numpy.ndarray` shaped
        ``(n_signals, n_timepoints)``.

        The amount of change in fluorence relative to the baseline fluorence
        (Δf/f\ :sub:`0`).
        By default, the baseline is taken from :attr:`raw` because the
        minimum values in :attr:`result` are typically zero.
        See :meth:`calc_deltaf` for details.

        This field is only populated after :meth:`calc_deltaf` has been run;
        until then, it is set to ``None``.
    """
    def __init__(self, images, rois, folder=None, nRegions=4,
                 expansion=1, alpha=0.1, ncores_preparation=None,
                 ncores_separation=None, method='nmf',
                 lowmemory_mode=False, datahandler=None):

        # Initialise internal variables
        self.clear()

        if isinstance(images, basestring):
            self.images = sorted(glob.glob(os.path.join(images, '*.tif*')))
        elif isinstance(images, abc.Sequence):
            self.images = images
        else:
            raise ValueError('images should either be string or list')

        if isinstance(rois, basestring):
            if rois[-3:] == 'zip':
                self.rois = [rois] * len(self.images)
            else:
                self.rois = sorted(glob.glob(os.path.join(rois, '*.zip')))
        elif isinstance(rois, abc.Sequence):
            self.rois = rois
            if len(rois) == 1:  # if only one roiset is specified
                self.rois *= len(self.images)
        else:
            raise ValueError('rois should either be string or list')

        if datahandler is not None and lowmemory_mode:
            raise ValueError(
                "Only one of lowmemory_mode and datahandler should be set."
            )
        elif lowmemory_mode:
            self.datahandler = extraction.DataHandlerTifffileLazy()
        else:
            self.datahandler = datahandler

        # define class variables
        self.folder = folder
        self.nRegions = nRegions
        self.expansion = expansion
        self.alpha = alpha
        self.nTrials = len(self.images)  # number of trials
        self.ncores_preparation = ncores_preparation
        self.ncores_separation = ncores_separation
        self.method = method

        # check if any data already exists
        if folder is None:
            pass
        elif folder and not os.path.exists(folder):
            os.makedirs(folder)
        else:
            self.load()

    def __str__(self):
        if isinstance(self.images, basestring):
            str_images = repr(self.images)
        elif isinstance(self.images, abc.Sequence):
            str_images = "<{} of length {}>".format(
                self.images.__class__.__name__, len(self.images)
            )
        else:
            str_images = repr(self.images)

        if isinstance(self.rois, basestring):
            str_rois = repr(self.rois)
        elif isinstance(self.rois, abc.Sequence):
            str_rois = "<{} of length {}>".format(
                self.rois.__class__.__name__, len(self.rois)
            )
        else:
            str_images = repr(self.rois)

        fields = [
            "folder",
            "nRegions",
            "expansion",
            "alpha",
            "ncores_preparation",
            "ncores_separation",
            "method",
            "datahandler",
        ]
        str_parts = [
            "{}={}".format(field, repr(getattr(self, field))) for field in fields
        ]
        return "{}.{}(images={}, rois={}, {})".format(
            __name__,
            self.__class__.__name__,
            str_images,
            str_rois,
            ", ".join(str_parts),
        )

    def __repr__(self):
        fields = [
            "images",
            "rois",
            "folder",
            "nRegions",
            "expansion",
            "alpha",
            "ncores_preparation",
            "ncores_separation",
            "method",
            "datahandler",
        ]
        repr_parts = [
            "{}={}".format(field, repr(getattr(self, field))) for field in fields
        ]
        return "{}.{}({})".format(
            __name__, self.__class__.__name__, ", ".join(repr_parts)
        )

    def clear(self):
        r"""
        Clear prepared data, and all data downstream of prepared data.

        .. versionadded:: 1.0.0
        """
        # Wipe outputs
        self.means = []
        self.nCell = None
        self.raw = None
        self.roi_polys = None
        # Wipe outputs of calc_deltaf(), as it no longer matches self.result
        self.deltaf_raw = None
        # Wipe outputs of separate(), as they no longer match self.raw
        self.clear_separated()

    def clear_separated(self):
        r"""
        Clear separated data, and all data downstream of separated data.

        .. versionadded:: 1.0.0
        """
        # Wipe outputs
        self.info = None
        self.mixmat = None
        self.sep = None
        self.result = None
        # Wipe deltaf_result, as it no longer matches self.result
        self.deltaf_result = None

    def load(self, path=None):
        r"""
        Load data from cache file in npz format.

        .. versionadded:: 1.0.0

        Parameters
        ----------
        path : str, optional
            Path to cache file (.npz format) or a directory containing
            ``"preparation.npz"`` and/or ``"separated.npz"`` files.
            Default behaviour is to use the :attr:`folder` parameter which was
            provided when the object was initialised is used
            (``experiment.folder``).
        """
        if path is None:
            if self.folder is None:
                raise ValueError(
                    "path must be provided if experiment folder is not defined"
                )
            path = self.folder
        if os.path.isdir(path) or path == "":
            for fname in ("preparation.npz", "separated.npz"):
                fullfname = os.path.join(path, fname)
                if not os.path.exists(fullfname):
                    continue
                self.load(fullfname)
            return
        print("Reloading data from cache {}...".format(path))
        cache = np.load(path, allow_pickle=True)
        for field in cache.files:
            value = cache[field]
            if np.array_equal(value, None):
                value = None
            setattr(self, field, value)

    def separation_prep(self, redo=False):
        r"""
        Prepare and extract the data to be separated.

        For each trial, performs the following steps:

        - Load in data as arrays.
        - Load in ROIs as masks.
        - Grow and seaparate ROIs to define neuropil regions.
        - Using neuropil and original ROI regions, extract traces from data.

        After running this you can access the raw data (i.e. pre-separation)
        as ``self.raw`` and ``self.rois``. self.raw is a list of arrays.
        ``self.raw[cell][trial]`` gives you the traces of a specific cell and
        trial, across cell and neuropil regions. ``self.roi_polys`` is a list of
        lists of arrays. ``self.roi_polys[cell][trial][region][0]`` gives you the
        polygon for the region for a specific cell, trial and region. ``region=0``
        is the cell, and ``region>0`` gives the different neuropil regions.
        For separateable masks, it is possible multiple outlines are found,
        which can be accessed as ``self.roi_polys[cell][trial][region][i]``,
        where ``i`` is the outline index.

        Parameters
        ----------
        redo : bool, optional
            If ``False``, we load previously prepared data when possible.
            If ``True``, we re-run the preparation, even if it has previously
            been run. Default is ``False``.
        """
        # define filename where data will be present
        if self.folder is None:
            fname = None
            redo = True
        else:
            fname = os.path.join(self.folder, "preparation.npz")

        # try to load data from filename
        if fname is None or not os.path.isfile(fname):
            redo = True
        if not redo:
            try:
                self.clear()
                self.load(fname)
                if self.raw is not None:
                    return
            except BaseException as err:
                print("An error occurred while loading {}".format(fname))
                print(err)
                print("Extraction will be redone and {} overwritten".format(fname))

        # Wipe outputs
        self.clear()
        # Extract signals
        print('Doing region growing and data extraction....')

        # Make a handle to the extraction function with parameters configured
        _extract_cfg = functools.partial(
            extract,
            nRegions=self.nRegions,
            expansion=self.expansion,
            datahandler=self.datahandler,
        )

        # Check whether we should use multiprocessing
        use_multiprocessing = (
            (self.ncores_preparation is None or self.ncores_preparation > 1)
        )
        # Do the extraction
        if use_multiprocessing and sys.version_info < (3, 0):
            # define pool
            pool = multiprocessing.Pool(self.ncores_preparation)
            # run extraction
            outputs = pool.map(
                _extract_wrapper,
                zip(
                    self.images,
                    self.rois,
                    itertools.repeat(self.nRegions, len(self.images)),
                    itertools.repeat(self.expansion, len(self.images)),
                    itertools.repeat(self.datahandler, len(self.images)),
                ),
            )
            pool.close()
            pool.join()

        elif use_multiprocessing:
            with multiprocessing.Pool(self.ncores_preparation) as pool:
                # run extraction
                outputs = pool.starmap(_extract_cfg, zip(self.images, self.rois))

        else:
            outputs = [_extract_cfg(*args) for args in zip(self.images, self.rois)]

        # get number of cells
        nCell = len(outputs[0][1])

        # predefine data structures
        raw = np.empty((nCell, self.nTrials), dtype=object)
        roi_polys = np.empty_like(raw)

        # Set outputs
        for trial in range(self.nTrials):
            self.means.append(outputs[trial][2])
            for cell in range(nCell):
                raw[cell][trial] = outputs[trial][0][cell]
                roi_polys[cell][trial] = outputs[trial][1][cell]

        self.nCell = nCell  # number of cells
        self.raw = raw
        self.roi_polys = roi_polys
        # Maybe save to cache file
        if self.folder is not None:
            self.save_prep()

    def save_prep(self, destination=None):
        r"""
        Save prepared raw signals, extracted from images, to an npz file.

        .. versionadded:: 1.0.0

        Parameters
        ----------
        destination : str, optional
            Path to output file. The default destination is ``"separated.npz"``
            within the cache directory ``self.folder``.
        """
        fields = ["means", "nCell", "raw", "roi_polys"]
        if destination is None:
            if self.folder is None:
                raise ValueError(
                    "The folder attribute must be declared in order to save"
                    " preparation outputs the cache."
                )
            destination = os.path.join(self.folder, 'preparation.npz')
        destdir = os.path.dirname(destination)
        if destdir and not os.path.isdir(destdir):
            os.makedirs(destdir)
        np.savez_compressed(
            destination,
            **{
                field: getattr(self, field)
                for field in fields
                if getattr(self, field) is not None
            }
        )

    def separate(self, redo_prep=False, redo_sep=False):
        r"""
        Separate all the trials with FISSA algorithm.

        After running ``separate``, data can be found as follows:

        experiment.sep
            Raw separation output, without being matched. Signal ``i`` for
            a specific cell and trial can be found as
            ``experiment.sep[cell][trial][i,:]``.
        experiment.result
            Final output, in order of presence in cell ROI.
            Signal ``i`` for a specific cell and trial can be found at
            ``experiment.result[cell][trial][i, :]``.
            Note that the ordering is such that ``i = 0`` is the signal
            most strongly present in the ROI, and subsequent entries
            are in diminishing order.
        experiment.mixmat
            The mixing matrix, which maps from ``experiment.raw`` to
            ``experiment.sep``.
        experiment.info
            Information about separation routine, iterations needed, etc.

        Parameters
        ----------
        redo_prep : bool, optional
            Whether to redo the preparation. Default is ``False.`` Note that
            if this is true, we set ``redo_sep = True`` as well.
        redo_sep : bool, optional
            Whether to redo the separation. Default is ``False``. Note that
            this parameter is ignored if `redo_prep` is set to ``True``.
        """
        # Do data preparation
        if redo_prep or self.raw is None:
            self.separation_prep(redo_prep)
        if redo_prep:
            redo_sep = True

        # Define filename to store data in
        if self.folder is None:
            fname = None
            redo_sep = True
        else:
            fname = os.path.join(self.folder, "separated.npz")
        if fname is None or not os.path.isfile(fname):
            redo_sep = True
        if not redo_sep:
            try:
                self.clear_separated()
                self.load(fname)
                if self.result is not None:
                    return
            except BaseException as err:
                print("An error occurred while loading {}".format(fname))
                print(err)
                print(
                    "Signal separation will be redone and {} overwritten"
                    "".format(fname)
                )

        # Wipe outputs
        self.clear_separated()
        # Separate data
        print('Doing signal separation....')

        # Check size of the input arrays
        n_roi = len(self.raw)
        n_trial = len(self.raw[0])

        # Make a handle to the separation function with parameters configured
        _separate_cfg = functools.partial(
            separate_trials,
            alpha=self.alpha,
            method=self.method,
        )

        # Check whether we should use multiprocessing
        use_multiprocessing = (
            (self.ncores_separation is None or self.ncores_separation > 1)
        )
        # Do the extraction
        if use_multiprocessing and sys.version_info < (3, 0):
            # define pool
            pool = multiprocessing.Pool(self.ncores_separation)
            # run separation
            outputs = pool.map(
                _separate_wrapper,
                zip(
                    self.raw,
                    range(n_roi),
                    itertools.repeat(self.alpha, n_roi),
                    itertools.repeat(self.method, n_roi),
                ),
            )
            pool.close()
            pool.join()

        elif use_multiprocessing:
            with multiprocessing.Pool(self.ncores_separation) as pool:
                # run separation
                outputs = pool.starmap(_separate_cfg, zip(self.raw, range(n_roi)))
        else:
            outputs = [_separate_cfg(X, roi_label=i) for i, X in enumerate(self.raw)]

        # Define output shape as an array of objects shaped (n_roi, n_trial)
        sep = np.empty((n_roi, n_trial), dtype=object)
        result = np.empty_like(sep)
        mixmat = np.empty_like(sep)
        info = np.empty_like(sep)

        # Place our outputs into the initialised arrays
        for i_roi, (sep_i, match_i, mixmat_i, conv_i) in enumerate(outputs):
            sep[i_roi, :] = sep_i
            result[i_roi, :] = match_i
            mixmat[i_roi, :] = [mixmat_i] * n_trial
            info[i_roi, :] = conv_i

        # Set outputs
        self.info = info
        self.mixmat = mixmat
        self.sep = sep
        self.result = result
        # Maybe save to cache file
        if self.folder is not None:
            self.save_separated()

    def save_separated(self, destination=None):
        r"""
        Save separated signals to an npz file.

        .. versionadded:: 1.0.0

        Parameters
        ----------
        destination : str, optional
            Path to output file. The default destination is ``"separated.npz"``
            within the cache directory ``self.folder``.
        """
        fields = ["deltaf_raw", "deltaf_result", "info", "mixmat", "sep", "result"]
        if destination is None:
            if self.folder is None:
                raise ValueError(
                    "The folder attribute must be declared in order to save"
                    " separation outputs to the cache."
                )
            destination = os.path.join(self.folder, "separated.npz")
        destdir = os.path.dirname(destination)
        if destdir and not os.path.isdir(destdir):
            os.makedirs(destdir)
        np.savez_compressed(
            destination,
            **{
                field: getattr(self, field)
                for field in fields
                if getattr(self, field) is not None
            }
        )

    def calc_deltaf(self, freq, use_raw_f0=True, across_trials=True):
        r"""
        Calculate deltaf/f0 for raw and result traces.

        The outputs are found in the :attr:`deltaf_raw` and
        :attr:`deltaf_result` attributes, which can be accessed at
        ``experiment.deltaf_raw`` and ``experiment.deltaf_result``.

        Parameters
        ----------
        freq : float
            Imaging frequency, in Hz.
        use_raw_f0 : bool, optional
            If ``True`` (default), use an f0 estimate from the raw ROI trace
            for both raw and result traces. If ``False``, use individual f0
            estimates for each of the traces.
        across_trials : bool, optional
            If ``True``, we estimate a single baseline f0 value across all
            trials. If ``False``, each trial will have their own baseline f0,
            and Δf/f\ :sub:`0` value will be relative to the trial-specific f0.
            Default is ``True``.
        """
        deltaf_raw = np.empty_like(self.raw)
        deltaf_result = np.empty_like(self.result)

        # loop over cells
        for cell in range(self.nCell):
            # if deltaf should be calculated across all trials
            if across_trials:
                # get concatenated traces
                raw_conc = np.concatenate(self.raw[cell], axis=1)[0, :]
                result_conc = np.concatenate(self.result[cell], axis=1)

                # calculate deltaf/f0
                raw_f0 = deltaf.findBaselineF0(raw_conc, freq)
                raw_conc = (raw_conc - raw_f0) / raw_f0
                result_f0 = deltaf.findBaselineF0(
                    result_conc, freq, 1
                ).T[:, None]
                if use_raw_f0:
                    result_conc = (result_conc - result_f0) / raw_f0
                else:
                    result_conc = (result_conc - result_f0) / result_f0

                # store deltaf/f0s
                curTrial = 0
                for trial in range(self.nTrials):
                    nextTrial = curTrial + self.raw[cell][trial].shape[1]
                    signal = raw_conc[curTrial:nextTrial]
                    deltaf_raw[cell][trial] = signal
                    signal = result_conc[:, curTrial:nextTrial]
                    deltaf_result[cell][trial] = signal
                    curTrial = nextTrial
            else:
                # loop across trials
                for trial in range(self.nTrials):
                    # get current signals
                    raw_sig = self.raw[cell][trial][0, :]
                    result_sig = self.result[cell][trial]

                    # calculate deltaf/fo
                    raw_f0 = deltaf.findBaselineF0(raw_sig, freq)
                    result_f0 = deltaf.findBaselineF0(
                        result_sig, freq, 1
                    ).T[:, None]
                    result_f0[result_f0 < 0] = 0
                    raw_sig = (raw_sig - raw_f0) / raw_f0
                    if use_raw_f0:
                        result_sig = (result_sig - result_f0) / raw_f0
                    else:
                        result_sig = (result_sig - result_f0) / result_f0

                    # store deltaf/f0s
                    deltaf_raw[cell][trial] = raw_sig
                    deltaf_result[cell][trial] = result_sig

        self.deltaf_raw = deltaf_raw
        self.deltaf_result = deltaf_result

        # Maybe save to cache file
        if self.folder is not None:
            self.save_separated()

    def save_to_matlab(self, fname=None):
        r"""
        Save the results to a MATLAB file.

        This will generate a .mat file which can be loaded into MATLAB to
        provide structs: ROIs, result, raw.

        If Δf/f\ :sub:`0` was calculated, these will also be stored as ``df_result``
        and ``df_raw``, which will have the same format as ``result`` and
        ``raw``.

        These can be interfaced with as follows, for cell 0, trial 0:

        ``ROIs.cell0.trial0{1}``
            Polygon outlining the ROI.
        ``ROIs.cell0.trial0{2}``
            Polygon outlining the first (of ``nRegions``) neuropil region.
        ``result.cell0.trial0(1, :)``
            Final extracted cell signal.
        ``result.cell0.trial0(2, :)``
            Contaminating signal.
        ``raw.cell0.trial0(1, :)``
            Raw measured cell signal, average over the ROI.
        ``raw.cell0.trial0(2, :)``
            Raw signal from first (of ``nRegions``) neuropil region.

        Parameters
        ----------
        fname : str, optional
            Destination for output file. Default is a file named
            ``"matlab.mat"`` within the cache save directory for the experiment
            (the `folder` argument when the ``Experiment`` instance was created).
        """
        # define filename
        if fname is None:
            if self.folder is None:
                raise ValueError(
                    'fname must be provided if experiment folder is undefined'
                )
            fname = os.path.join(self.folder, 'matlab.mat')

        # initialize dictionary to save
        M = collections.OrderedDict()

        def reformat_dict_for_matlab(orig_dict):
            new_dict = collections.OrderedDict()
            # loop over cells and trial
            for cell in range(self.nCell):
                # get current cell label
                c_lab = 'cell' + str(cell)
                # update dictionary
                new_dict[c_lab] = collections.OrderedDict()
                for trial in range(self.nTrials):
                    # get current trial label
                    t_lab = 'trial' + str(trial)
                    # update dictionary
                    new_dict[c_lab][t_lab] = orig_dict[cell][trial]
            return new_dict

        M['ROIs'] = reformat_dict_for_matlab(self.roi_polys)
        M['raw'] = reformat_dict_for_matlab(self.raw)
        M['result'] = reformat_dict_for_matlab(self.result)
        if getattr(self, 'deltaf_raw', None) is not None:
            M['df_raw'] = reformat_dict_for_matlab(self.deltaf_raw)
        if getattr(self, 'deltaf_result', None) is not None:
            M['df_result'] = reformat_dict_for_matlab(self.deltaf_result)

        savemat(fname, M)


def run_fissa(
    images,
    rois,
    folder=None,
    freq=None,
    return_deltaf=False,
    deltaf_across_trials=True,
    export_to_matlab=False,
    **kwargs
):
    r"""
    Functional interface to run FISSA.

    .. versionadded:: 1.0.0

    Parameters
    ----------
    images : str or list
        The raw recording data.
        Should be one of:

        - the path to a directory containing TIFF files (string),
        - a list of paths to TIFF files (list of strings),
        - a list of :term:`array_like` data already loaded into memory, each
          shaped ``(n_frames, height, width)``.

        Note that each TIFF/array is considered a single trial.

    rois : str or list
        The roi definitions.
        Should be one of:

        - the path to a directory containing ImageJ ZIP files (string),
        - the path of a single ImageJ ZIP file (string),
        - a list of ImageJ ZIP files (list of strings),
        - a list of arrays, each encoding a ROI polygons,
        - a list of lists of binary arrays, each representing a ROI mask.

        This can either be a single roiset for all trials, or a different
        roiset for each trial.

    folder : str, optional
        Path to a cache directory from which pre-extracted data will
        be loaded if present, and saved to otherwise. If `folder` is
        unset, the experiment data will not be saved.

    freq : float, optional
        Imaging frequency, in Hz. Required if ``return_deltaf=True``.

    return_deltaf : bool, optional
        Whether to return Δf/f\ :sub:`0`. Otherwise, the decontaminated signal
        is returned scaled against the raw recording. Default is ``False``.

    deltaf_across_trials : bool, default=True
        If ``True``, we estimate a single baseline f0 value across all
        trials when computing Δf/f\ :sub:`0`.
        If ``False``, each trial will have their own baseline f0, and
        Δf/f\ :sub:`0` value will be relative to the trial-specific f0.
        Default is ``True``.

    export_to_matlab : bool or str or None, default=False
        Whether to export the data to a MATLAB-compatible .mat file.
        If `export_to_matlab` is a string, it is used as the path to the output
        file. If ``export_to_matlab=True``, the matfile is saved to the
        default path of ``"matlab.mat"`` within the `folder` directory, and
        `folder` must be set. If this is ``None``, the matfile is exported to
        the default path if `folder` is set, and otherwise is not exported.
        Default is ``False``.

    **kwargs
        Additional keyword arguments as per :class:`Experiment`.

    Returns
    -------
    result : 2d numpy.ndarray of 2d numpy.ndarrays of np.float64
        The vector ``result[c, t][0, :]`` is the trace from cell ``c`` in
        trial ``t``. If ``return_deltaf=True``, this is Δf/f\ :sub:`0`;
        otherwise, it is the decontaminated signal scaled as per the raw
        signal.

    See Also
    --------
    fissa.core.Experiment
    """
    # Parse arguments
    if export_to_matlab is None:
        export_to_matlab = folder is not None
    if return_deltaf and freq is None:
        raise ValueError("The argument `freq` must be set to determine df/f0.")
    # Make a new Experiment object
    experiment = Experiment(images, rois, folder=folder, **kwargs)
    # Run separation
    experiment.separate()
    # Calculate df/f0
    if return_deltaf or (export_to_matlab and freq is not None):
        experiment.calc_deltaf(freq=freq, across_trials=deltaf_across_trials)
    # Save to matfile
    if export_to_matlab:
        matlab_fname = None if isinstance(export_to_matlab, bool) else export_to_matlab
        experiment.save_to_matlab(matlab_fname)
    # Return appropriate data
    if return_deltaf:
        return experiment.deltaf_result
    return experiment.result
