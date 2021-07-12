# -*- coding: utf-8 -*-
"""
Main FISSA user interface.

Authors:
    - Sander W Keemink <swkeemink@scimail.eu>
    - Scott C Lowe <scott.code.lowe@gmail.com>
"""

from __future__ import print_function

import collections
import datetime
import functools
import glob
import itertools
import os.path
import sys
import time
import warnings

try:
    from collections import abc
except ImportError:
    import collections as abc

import numpy as np
from joblib import Parallel, delayed
from past.builtins import basestring
from scipy.io import savemat
from tqdm.auto import tqdm

from . import deltaf, extraction
from . import neuropil as npil
from . import roitools


def _pretty_timedelta(td=None, **kwargs):
    """
    Represent a difference in time as a human-readable string.

    Parameters
    ----------
    td : datetime.timedelta, optional
        The amount of time elapsed.
    **kwargs
        Additional arguments as per :class:`datetime.timedelta` constructor.

    Returns
    -------
    str
        Representation of the amount of time elapsed.
    """
    if td is None:
        td = datetime.timedelta(**kwargs)
    elif not isinstance(td, datetime.timedelta):
        raise ValueError(
            "First argument should be a datetime.timedelta instance,"
            " but {} was given.".format(type(td))
        )
    elif kwargs:
        raise ValueError(
            "Either a timedelta object or its arguments should be given, not both."
        )
    if td.total_seconds() < 2:
        return "{:.3f} seconds".format(td.total_seconds())
    if td.total_seconds() < 10:
        return "{:.2f} seconds".format(td.total_seconds())
    if td.total_seconds() < 60:
        return "{:.1f} seconds".format(td.total_seconds())
    if td.total_seconds() < 3600:
        s = td.total_seconds()
        m = int(s // 60)
        s -= m * 60
        return "{:d} min, {:.0f} sec".format(m, s)
    # For durations longer than one hour, we use the default string
    # representation for a datetime.timedelta, H:MM:SS.microseconds
    return str(td)


def extract(
    image,
    rois,
    nRegions=4,
    expansion=1,
    datahandler=None,
    verbosity=1,
    label=None,
    total=None,
):
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
    verbosity : int, default=1
        Level of verbosity. The options are:

        - ``0``: No outputs.
        - ``1``: Print separation progress.

    label : str or int, optional
        The label for the current trial. Only used for reporting progress.
    total : int, optional
        Total number of trials. Only used for reporting progress.

    Returns
    -------
    data : dict
        Data across cells.
    roi_polys : dict
        Polygons for each ROI.
    mean : :class:`numpy.ndarray` shaped (height, width)
        Mean image.
    """
    # Get the timestamp for program start
    t0 = time.time()

    mheader = ""
    if verbosity >= 1:
        # Set up message header
        # Use the label, if this was provided
        if label is None:
            header = ""
        elif isinstance(label, int) and isinstance(total, int):
            # Pad left based on the total number of jobs, so it is [ 1/10] etc
            fmtstr = "{:" + str(int(np.maximum(1, np.ceil(np.log10(total))))) + "d}"
            header = fmtstr.format(label + 1)
        else:
            header = str(label)
        # Try to label with [1/5] to indicate progess, if possible
        if header and total is not None:
            header += "/{}".format(total)
        if header:
            header = "[Extraction " + header + "] "
        # Try to include the path to the image as a footer
        footer = ""
        if isinstance(image, basestring):
            # Include the image path as a footer
            footer = " ({})".format(image)
        # Done with header and footer
        # Inner header is indented further
        mheader = "    " + header

        # Build intro message
        message = header + "Extraction starting" + footer
        # Wait briefly to prevent messages colliding when using multiprocessing
        if isinstance(label, int) and label < 12:
            time.sleep(label / 50.0)
        print(message, flush=True)

    if datahandler is None:
        datahandler = extraction.DataHandlerTifffile()

    # get data as arrays and rois as masks
    if verbosity >= 3:
        print("{}Loading imagery".format(mheader))
    curdata = datahandler.image2array(image)
    if verbosity >= 3:
        print("{}Converting ROIs to masks".format(mheader))
    base_masks = datahandler.rois2masks(rois, curdata)

    # get the mean image
    mean = datahandler.getmean(curdata)

    # predefine dictionaries
    data = collections.OrderedDict()
    roi_polys = collections.OrderedDict()

    if verbosity == 3:
        print("{}Growing neuropil regions and extracting traces".format(mheader))
    # get neuropil masks and extract signals
    for cell in tqdm(
        range(len(base_masks)),
        total=len(base_masks),
        desc="{}Neuropil extraction".format(mheader),
        disable=verbosity < 4,
    ):
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

    if verbosity >= 2:
        # Build end message
        message = header + "Extraction finished" + footer
        message += " in {}".format(_pretty_timedelta(seconds=time.time() - t0))
        print(message, flush=True)

    return data, roi_polys, mean


def separate_trials(
    raw,
    alpha=0.1,
    max_iter=20000,
    tol=1e-4,
    max_tries=1,
    method="nmf",
    verbosity=1,
    label=None,
    total=None,
):
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

    alpha : float, default=0.1
        Sparsity regularizaton weight for NMF algorithm. Set to zero to
        remove regularization. Default is ``0.1``.
        (Only used for ``method="nmf"``.)

    max_iter : int, default=20000
        Maximum number of iterations before timing out on an attempt.

    tol : float, default=1e-4
        Tolerance of the stopping condition.

    max_tries : int, default=1
        Maximum number of random initial states to try. Each random state will
        be optimized for `max_iter` iterations before timing out.

    method : {"nmf", "ica"}, default="nmf"
        Which blind source-separation method to use. Either ``"nmf"``
        for non-negative matrix factorization, or ``"ica"`` for
        independent component analysis. Default is ``"nmf"``.

    verbosity : int, default=1
        Level of verbosity. The options are:

        - ``0``: No outputs.
        - ``1``: Print separation progress.

    label : str or int, optional
        Label/name or index of the ROI currently being processed.
        Only used for progress messages.

    total : int, optional
        Total number of ROIs. Only used for reporting progress.

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
    # Get the timestamp for program start
    t0 = time.time()

    header = ""
    if verbosity >= 1:
        # Set up message header
        # Use the label, if this was provided
        if label is None:
            header = ""
        elif isinstance(label, int) and isinstance(total, int):
            # Pad left based on the total number of jobs, so it is [ 1/10] etc
            fmtstr = "{:" + str(int(np.maximum(1, np.ceil(np.log10(total))))) + "d}"
            header = fmtstr.format(label + 1)
        else:
            header = str(label)
        # Try to label with [1/5] to indicate progess, if possible
        if header and total is not None:
            header += "/{}".format(total)
        if header:
            header = "[Separation " + header + "] "
        # Include the ROI label as a footer
        footer = ""
        if isinstance(label, int) and isinstance(total, int):
            # Include the ROI label as a footer
            footer = " (ROI {})".format(label)
        # Done with header and footer

        # Build intro message
        message = header + "Signal separation starting" + footer
        # Wait briefly to prevent messages colliding when using multiprocessing
        if isinstance(label, int) and label < 12:
            time.sleep(label / 50.0)
        print(message, flush=True)

    # Join together the raw data across trials, collapsing down the trials
    X = np.concatenate(raw, axis=1)

    # Check for values below 0
    if X.min() < 0:
        message_extra = ""
        if label is not None:
            message_extra = " for ROI {}".format(label)
        warnings.warn(
            "{}Found values below zero in raw signal{}. Offsetting so minimum is 0."
            "".format(header, message_extra)
        )
        X -= X.min()

    # Separate the signals
    Xsep, Xmatch, Xmixmat, convergence = npil.separate(
        X,
        method,
        max_iter=max_iter,
        tol=tol,
        max_tries=max_tries,
        alpha=alpha,
        verbosity=verbosity - 2,
        prefix="    " + header,
    )
    # Unravel observations from multiple trials into a list of arrays
    trial_lengths = [r.shape[1] for r in raw]
    indices = np.cumsum(trial_lengths[:-1])
    Xsep = np.split(Xsep, indices, axis=1)
    Xmatch = np.split(Xmatch, indices, axis=1)

    # Report status
    if verbosity >= 2:
        # Build end message
        message = header + "Signal separation finished" + footer
        message += " in {}".format(_pretty_timedelta(seconds=time.time() - t0))
        print(message, flush=True)

    return Xsep, Xmatch, Xmixmat, convergence


class Experiment:
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

    max_iter : int, default=20000
        Maximum number of iterations before timing out on an attempt.

        .. versionadded:: 1.0.0

    tol : float, default=1e-4
        Tolerance of the stopping condition.

        .. versionadded:: 1.0.0

    max_tries : int, default=1
        Maximum number of random initial states to try. Each random state will
        be optimized for `max_iter` iterations before timing out.

        .. versionadded:: 1.0.0

    ncores_preparation : int or None, default=-1
        The number of parallel subprocesses to use during the data
        preparation steps of :meth:`separation_prep`.
        These are ROI and neuropil subregion definitions, and extracting
        raw signals from TIFFs.

        If set to ``None`` or ``-1`` (default), the number of processes used
        will equal the number of threads on the machine.
        If this is set to ``-2``, the number of processes used will be one less
        than the number of threads on the machine; etc.
        Note that this behaviour can, especially for the data preparation step,
        be very memory-intensive.

    ncores_separation : int or None, default=-1
        The number of parallel subprocesses to use during the signal
        separation steps of :meth:`separate`.
        The separation steps requires less memory per subprocess than
        the preparation steps, and so can be often be set higher than
        `ncores_preparation`.

        If set to ``None`` or ``-1`` (default), the number of processes used
        will equal the number of threads on the machine.
        If this is set to ``-2``, the number of processes used will be one less
        than the number of threads on the machine; etc.

    method : "nmf" or "ica", default="nmf"
        Which blind source-separation method to use. Either ``"nmf"``
        for non-negative matrix factorization, or ``"ica"`` for
        independent component analysis. Default is ``"nmf"`` (recommended).

    lowmemory_mode : bool, optional
        If ``True``, FISSA will load TIFF files into memory frame-by-frame
        instead of holding the entire TIFF in memory at once. This
        option reduces the memory load, and may be necessary for very
        large inputs. Default is ``False``.

    datahandler : :class:`fissa.extraction.DataHandlerAbstract`, optional
        A custom datahandler object for handling ROIs and calcium data can
        be given here. See :mod:`fissa.extraction` for example datahandler
        classes. The default datahandler is
        :class:`~fissa.extraction.DataHandlerTifffile`.
        If `datahandler` is set, the `lowmemory_mode` parameter is
        ignored.

    verbosity : int, default=1
        How verbose the processing will be. The options are:

        - ``0``: No outputs.
        - ``1``: Progress bars and high level summary.
        - ``2``: Print intermediate progress steps.
        - ``3``: Print per-cell progress.

        .. versionadded:: 1.0.0

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

    means : list of `n_trials` :class:`numpy.ndarray`, each shaped ``(height, width)``
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
        of which is itself a :class:`numpy.ndarray` shaped ``(1, n_timepoint)``.

        The amount of change in fluorence relative to the baseline fluorence
        (Δf/f\ :sub:`0`).

        This field is only populated after :meth:`calc_deltaf` has been run;
        until then, it is set to ``None``.

        .. versionchanged:: 1.0.0
            The shape of the interior arrays changed from ``(n_timepoint, )``
            to ``(1, n_timepoint)``.

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

    def __init__(
        self,
        images,
        rois,
        folder=None,
        nRegions=4,
        expansion=1,
        alpha=0.1,
        max_iter=20000,
        tol=1e-4,
        max_tries=1,
        ncores_preparation=-1,
        ncores_separation=-1,
        method="nmf",
        lowmemory_mode=False,
        datahandler=None,
        verbosity=1,
    ):

        # Initialise internal variables
        self.clear(verbosity=0)

        if isinstance(images, basestring):
            self.images = sorted(glob.glob(os.path.join(images, "*.tif*")))
        elif isinstance(images, abc.Sequence):
            self.images = images
        else:
            raise ValueError("images should either be string or list")

        if isinstance(rois, basestring):
            if rois[-3:] == "zip":
                self.rois = [rois] * len(self.images)
            else:
                self.rois = sorted(glob.glob(os.path.join(rois, "*.zip")))
        elif isinstance(rois, abc.Sequence):
            self.rois = rois
            if len(rois) == 1:  # if only one roiset is specified
                self.rois *= len(self.images)
        else:
            raise ValueError("rois should either be string or list")

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
        self.max_iter = max_iter
        self.tol = tol
        self.max_tries = max_tries
        self.nTrials = len(self.images)  # number of trials
        self.ncores_preparation = ncores_preparation
        self.ncores_separation = ncores_separation
        self.method = method
        self.verbosity = verbosity

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
            "max_iter",
            "tol",
            "max_tries",
            "ncores_preparation",
            "ncores_separation",
            "method",
            "datahandler",
            "verbosity",
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
            "max_iter",
            "tol",
            "max_tries",
            "ncores_preparation",
            "ncores_separation",
            "method",
            "datahandler",
            "verbosity",
        ]
        repr_parts = [
            "{}={}".format(field, repr(getattr(self, field))) for field in fields
        ]
        return "{}.{}({})".format(
            __name__, self.__class__.__name__, ", ".join(repr_parts)
        )

    def clear(self, verbosity=None):
        r"""
        Clear prepared data, and all data downstream of prepared data.

        .. versionadded:: 1.0.0

        Parameters
        ----------
        verbosity : int, optional
            Whether to show the data fields which were cleared.
            By default, the object's :attr:`verbosity` attribute is used.
        """
        if verbosity is None:
            verbosity = self.verbosity - 1

        keys = ["means", "nCell", "raw", "roi_polys", "deltaf_raw"]
        # Wipe outputs
        keys_cleared = []
        for key in keys:
            if getattr(self, key, None) is not None:
                keys_cleared.append(key)
            setattr(self, key, None)

        if verbosity >= 1 and keys_cleared:
            print("Cleared {}".format(", ".join(repr(k) for k in keys_cleared)))

        # Wipe outputs of separate(), as they no longer match self.raw
        self.clear_separated(verbosity=verbosity)

    def clear_separated(self, verbosity=None):
        r"""
        Clear separated data, and all data downstream of separated data.

        .. versionadded:: 1.0.0

        Parameters
        ----------
        verbosity : int, optional
            Whether to show the data fields which were cleared.
            By default, the object's :attr:`verbosity` attribute is used.
        """
        if verbosity is None:
            verbosity = self.verbosity - 1

        keys = ["info", "mixmat", "sep", "result", "deltaf_result"]
        # Wipe outputs
        keys_cleared = []
        for key in keys:
            if getattr(self, key, None) is not None:
                keys_cleared.append(key)
            setattr(self, key, None)

        if verbosity >= 1 and keys_cleared:
            print("Cleared {}".format(", ".join(repr(k) for k in keys_cleared)))

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
        if self.verbosity >= 1:
            print("Reloading data from cache {}".format(path))
        cache = np.load(path, allow_pickle=True)
        for field in cache.files:
            value = cache[field]
            if np.array_equal(value, None):
                value = None
            elif value.ndim == 0:
                # Handle loading scalars
                value = value.item()
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
        as ``experiment.raw`` and ``experiment.rois``.
        ``experiment.raw`` is a list of arrays.
        ``experiment.raw[cell][trial]`` gives you the traces of a specific cell
        and trial, across cell and neuropil regions.
        ``experiment.roi_polys`` is a list of lists of arrays.
        ``experiment.roi_polys[cell][trial][region][0]`` gives you the
        polygon for the region for a specific cell, trial and region.
        ``region=0`` is the cell, and ``region>0`` gives the different neuropil
        regions.
        For separable masks, it is possible multiple outlines are
        found, which can be accessed as
        ``experiment.roi_polys[cell][trial][region][i]``,
        where ``i`` is the outline index.

        Parameters
        ----------
        redo : bool, optional
            If ``False``, we load previously prepared data when possible.
            If ``True``, we re-run the preparation, even if it has previously
            been run. Default is ``False``.
        """
        # Get the timestamp for program start
        t0 = time.time()

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
        n_trial = len(self.images)
        if self.verbosity >= 2:
            msg = "Doing region growing and data extraction for {} trials...".format(
                n_trial
            )
            msg += "\n  Images:"
            for image in self.images:
                if self.verbosity >= 3 or isinstance(image, basestring):
                    msg += "\n    {}".format(image)
                else:
                    msg += "\n    {}".format(image.__class__)
            msg += "\n  ROI sets:"
            for roiset in self.rois:
                if self.verbosity >= 3 or isinstance(roiset, basestring):
                    msg += "\n    {}".format(roiset)
                else:
                    msg += "\n    {}".format(roiset.__class__)
            print(msg, flush=True)

        # Make a handle to the extraction function with parameters configured
        _extract_cfg = functools.partial(
            extract,
            nRegions=self.nRegions,
            expansion=self.expansion,
            datahandler=self.datahandler,
            verbosity=self.verbosity - 1,
            total=n_trial,
        )

        # check whether we should show progress bars
        disable_progressbars = self.verbosity != 1

        # Check how many workers to spawn.
        # Map the behaviour of ncores=None to one job per CPU core, like for
        # multiprocessing.Pool(processes=None). With joblib, this is
        # joblib.Parallel(n_jobs=-1) instead.
        n_jobs = -1 if self.ncores_preparation is None else self.ncores_preparation

        if 0 <= n_jobs <= 1:
            # Don't use multiprocessing
            outputs = [
                _extract_cfg(image, rois, label=i)
                for i, (image, rois) in tqdm(
                    enumerate(zip(self.images, self.rois)),
                    total=self.nTrials,
                    desc="Extracting traces",
                    disable=disable_progressbars,
                )
            ]
        else:
            # Use multiprocessing
            outputs = Parallel(
                n_jobs=n_jobs, backend="threading", verbose=max(0, self.verbosity - 4)
            )(
                delayed(_extract_cfg)(image, rois, label=i)
                for i, (image, rois) in tqdm(
                    enumerate(zip(self.images, self.rois)),
                    total=self.nTrials,
                    desc="Extracting traces",
                    disable=disable_progressbars,
                )
            )

        # get number of cells
        nCell = len(outputs[0][1])

        # predefine data structures
        raw = np.empty((nCell, self.nTrials), dtype=object)
        roi_polys = np.empty_like(raw)

        # Set outputs
        if self.means is None:
            self.means = []
        for trial in range(self.nTrials):
            self.means.append(outputs[trial][2])
            for cell in range(nCell):
                raw[cell][trial] = outputs[trial][0][cell]
                roi_polys[cell][trial] = outputs[trial][1][cell]

        self.nCell = nCell  # number of cells
        self.raw = raw
        self.roi_polys = roi_polys

        if self.verbosity >= 1:
            print(
                "Finished extracting raw signals from {} ROIs across {} trials in {}.".format(
                    nCell,
                    n_trial,
                    _pretty_timedelta(seconds=time.time() - t0),
                ),
                flush=True,
            )

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
            Path to output file. The default destination is
            ``"preparation.npz"`` within the cache directory
            ``experiment.folder``.
        """
        fields = ["expansion", "means", "nCell", "nRegions", "raw", "roi_polys"]
        if destination is None:
            if self.folder is None:
                raise ValueError(
                    "The folder attribute must be declared in order to save"
                    " preparation outputs the cache."
                )
            destination = os.path.join(self.folder, "preparation.npz")
        if self.verbosity >= 1:
            print("Saving extracted traces to {}".format(destination), flush=True)
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
        # Get the timestamp for program start
        t0 = time.time()

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

        # Check size of the input arrays
        n_roi = len(self.raw)
        n_trial = len(self.raw[0])
        # Print what data will be analysed
        if self.verbosity >= 2:
            print(
                "Doing signal separation for {} ROIs over {} trials...".format(
                    n_roi, n_trial
                )
            )

        # Make a handle to the separation function with parameters configured
        _separate_cfg = functools.partial(
            separate_trials,
            alpha=self.alpha,
            max_iter=self.max_iter,
            tol=self.tol,
            max_tries=self.max_tries,
            method=self.method,
            verbosity=self.verbosity - 1,
            total=n_roi,
        )

        # check whether we should show progress bars
        disable_progressbars = self.verbosity != 1

        # Check how many workers to spawn.
        # Map the behaviour of ncores=None to one job per CPU core, like for
        # multiprocessing.Pool(processes=None). With joblib, this is
        # joblib.Parallel(n_jobs=-1) instead.
        n_jobs = -1 if self.ncores_separation is None else self.ncores_separation

        # Do the extraction
        if 0 <= n_jobs <= 1:
            # Don't use multiprocessing
            outputs = [
                _separate_cfg(X, label=i)
                for i, X in tqdm(
                    enumerate(self.raw),
                    total=self.nCell,
                    desc="Separating data",
                    disable=disable_progressbars,
                )
            ]
        else:
            # Use multiprocessing
            outputs = Parallel(
                n_jobs=n_jobs, backend="threading", verbose=max(0, self.verbosity - 4)
            )(
                delayed(_separate_cfg)(X, label=i)
                for i, X in tqdm(
                    enumerate(self.raw),
                    total=self.nCell,
                    desc="Separating data",
                    disable=disable_progressbars,
                )
            )

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

        # list non-converged cells
        non_converged_rois = [
            i_roi for i_roi, info_i in enumerate(info) if not info_i[0]["converged"]
        ]

        if self.verbosity >= 1:
            message = "Finished separating signals from {} ROIs across {} trials in {}".format(
                n_roi,
                n_trial,
                _pretty_timedelta(seconds=time.time() - t0),
            )
            if len(non_converged_rois) > 0:
                message += (
                    "\n"
                    "The following {} ROIs did not fully converge: {}."
                    " Consider increasing max_iter (currently set to {})"
                    " or other FISSA parameters if this happens often and/or"
                    " to a lot of cells.".format(
                        len(non_converged_rois), non_converged_rois, self.max_iter
                    )
                )
            print(message, flush=True)

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
            within the cache directory ``experiment.folder``.
        """
        fields = [
            "alpha",
            "deltaf_raw",
            "deltaf_result",
            "info",
            "max_iter",
            "max_tries",
            "method",
            "mixmat",
            "sep",
            "tol",
            "result",
        ]
        if destination is None:
            if self.folder is None:
                raise ValueError(
                    "The folder attribute must be declared in order to save"
                    " separation outputs to the cache."
                )
            destination = os.path.join(self.folder, "separated.npz")
        if self.verbosity >= 1:
            print("Saving results to {}".format(destination), flush=True)
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
        Calculate Δf/f0 for raw and result traces.

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
        # Get the timestamp for program start
        t0 = time.time()

        if self.verbosity >= 2:
            print("Calculating Δf/f0 for raw and result signals", flush=True)

        # Initialise output arrays
        deltaf_raw = np.empty_like(self.raw)
        deltaf_result = np.empty_like(self.result)

        # loop over cells
        disable_progressbars = self.verbosity < 1
        for cell in tqdm(
            range(self.nCell),
            total=self.nCell,
            desc="Calculating Δf/f0",
            disable=disable_progressbars,
        ):
            # if deltaf should be calculated across all trials
            if across_trials:
                # get concatenated traces
                raw_conc = np.concatenate(self.raw[cell], axis=1)[0, :]
                result_conc = np.concatenate(self.result[cell], axis=1)

                # calculate Δf/f0
                raw_f0 = deltaf.findBaselineF0(raw_conc, freq)
                raw_conc = (raw_conc - raw_f0) / raw_f0
                result_f0 = deltaf.findBaselineF0(result_conc, freq, 1).T[:, None]
                if use_raw_f0:
                    result_conc = (result_conc - result_f0) / raw_f0
                else:
                    result_conc = (result_conc - result_f0) / result_f0

                # store Δf/f0
                curTrial = 0
                for trial in range(self.nTrials):
                    nextTrial = curTrial + self.raw[cell][trial].shape[1]
                    signal = raw_conc[curTrial:nextTrial]
                    deltaf_raw[cell][trial] = np.expand_dims(signal, axis=0)
                    signal = result_conc[:, curTrial:nextTrial]
                    deltaf_result[cell][trial] = signal
                    curTrial = nextTrial
            else:
                # loop across trials
                for trial in range(self.nTrials):
                    # get current signals
                    raw_sig = self.raw[cell][trial][0, :]
                    result_sig = self.result[cell][trial]

                    # calculate Δf/fo
                    raw_f0 = deltaf.findBaselineF0(raw_sig, freq)
                    result_f0 = deltaf.findBaselineF0(result_sig, freq, 1).T[:, None]
                    result_f0[result_f0 < 0] = 0
                    raw_sig = (raw_sig - raw_f0) / raw_f0
                    if use_raw_f0:
                        result_sig = (result_sig - result_f0) / raw_f0
                    else:
                        result_sig = (result_sig - result_f0) / result_f0

                    # store Δf/f0
                    deltaf_raw[cell][trial] = np.expand_dims(raw_sig, axis=0)
                    deltaf_result[cell][trial] = result_sig

        self.deltaf_raw = deltaf_raw
        self.deltaf_result = deltaf_result

        if self.verbosity >= 1:
            print(
                "Finished calculating Δf/f0 for raw and result signals in {}".format(
                    _pretty_timedelta(seconds=time.time() - t0)
                ),
                flush=True,
            )

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
                    "fname must be provided if experiment folder is undefined"
                )
            fname = os.path.join(self.folder, "matlab.mat")

        if self.verbosity >= 1:
            print("Exporting results to matfile {}".format(fname), flush=True)

        # initialize dictionary to save
        M = collections.OrderedDict()

        def reformat_dict_for_matlab(orig_dict):
            new_dict = collections.OrderedDict()
            # loop over cells and trial
            for cell in range(self.nCell):
                # get current cell label
                c_lab = "cell" + str(cell)
                # update dictionary
                new_dict[c_lab] = collections.OrderedDict()
                for trial in range(self.nTrials):
                    # get current trial label
                    t_lab = "trial" + str(trial)
                    # update dictionary
                    new_dict[c_lab][t_lab] = orig_dict[cell][trial]
            return new_dict

        M["ROIs"] = reformat_dict_for_matlab(self.roi_polys)
        M["raw"] = reformat_dict_for_matlab(self.raw)
        M["result"] = reformat_dict_for_matlab(self.result)
        if getattr(self, "deltaf_raw", None) is not None:
            M["df_raw"] = reformat_dict_for_matlab(self.deltaf_raw)
        if getattr(self, "deltaf_result", None) is not None:
            M["df_result"] = reformat_dict_for_matlab(self.deltaf_result)

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
    # Calculate Δf/f0
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
