# -*- coding: utf-8 -*-
"""
Main FISSA user interface.

Authors:
    - Sander W Keemink <swkeemink@scimail.eu>
    - Scott C Lowe <scott.code.lowe@gmail.com>
"""

from __future__ import print_function
from past.builtins import basestring

import collections
import glob
from multiprocessing import Pool
import os.path
import sys
import warnings
try:
    from collections import abc
except ImportError:
    import collections as abc

import numpy as np
from scipy.io import savemat

from . import deltaf
from . import extraction
from . import neuropil as npil
from . import roitools


def extract_func(inputs):
    """Extract data using multiprocessing.

    Parameters
    ----------
    inputs : list
        list of inputs

        0. image array
        1. the rois
        2. number of neuropil regions
        3. how much larger neuropil region should be then central ROI

    Returns
    -------
    data : dict
        Data across cells.
    roi_polys : dict
        Polygons for each ROI.
    mean : np.ndarray
        Mean image.
    """
    image = inputs[0]
    rois = inputs[1]
    nNpil = inputs[2]
    expansion = inputs[3]
    datahandler = inputs[4]

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
        npil_masks = roitools.getmasks_npil(base_masks[cell], nNpil=nNpil,
                                            expansion=expansion)
        # add all current masks together
        masks = [base_masks[cell]] + npil_masks

        # extract traces
        data[cell] = datahandler.extracttraces(curdata, masks)

        # store ROI outlines
        roi_polys[cell] = [roitools.find_roi_edge(mask) for mask in masks]

    return data, roi_polys, mean


def separate_func(inputs):
    """Extraction function for multiprocessing.

    Parameters
    ----------
    inputs : list
        list of inputs

        0. Array with signals to separate
        1. Alpha input to npil.separate
        2. Method
        3. Current ROI number

    Returns
    -------
    Xsep : numpy.ndarray
        The raw separated traces.
    Xmatch : numpy.ndarray
        The separated traces matched to the primary signal.
    Xmixmat : numpy.ndarray
        Mixing matrix.
    convergence : dict
        Metadata for the convergence result.
    """
    X = inputs[0]
    alpha = inputs[1]
    method = inputs[2]

    Xsep, Xmatch, Xmixmat, convergence = npil.separate(
        X, method, maxiter=20000, tol=1e-4, maxtries=1, alpha=alpha
    )
    ROInum = inputs[3]
    print('Finished ROI number ' + str(ROInum))
    return Xsep, Xmatch, Xmixmat, convergence


class Experiment():
    """
    Does all the steps for FISSA.

    Parameters
    ----------
    images : str or list
        The raw recording data.
        Should be one of:

        - the path to a directory containing TIFF files (string),
        - an explicit list of TIFF files (list of strings),
        - a list of :term:`array_like` data already loaded into memory,
          each shaped ``(frames, y-coords, x-coords)``.

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

    folder : str or None, optional
        Output path to a directory in which the extracted data will
        be stored. If ``None`` (default), the data will not be cached.
    nRegions : int, optional
        Number of neuropil regions to draw. Use a higher number for
        densely labelled tissue. Default is ``4``.
    expansion : float, optional
        Expansion factor for the neuropil region, relative to the
        ROI area. Default is ``1``. The total neuropil area will be
        ``nRegions * expansion * area(ROI)``.
    alpha : float, optional
        Sparsity regularizaton weight for NMF algorithm. Set to zero to
        remove regularization. Default is ``0.1``.
        (Not used for ICA method.)
    ncores_preparation : int or None, optional
        Sets the number of subprocesses to be used during the data
        preparation steps (ROI and subregions definitions, data
        extraction from TIFFs, etc.).
        If set to ``None`` (default), there will be as many subprocesses
        as there are threads or cores on the machine. Note that this
        behaviour can, especially for the data preparation step,
        be very memory-intensive.
    ncores_separation : int or None, optional
        Same as `ncores_preparation`, but for the separation step.
        Note that this step requires less memory per subprocess, and
        hence can often be set higher than `ncores_preparation`.
    method : {'nmf', 'ica'}, optional
        Which blind source-separation method to use. Either ``'nmf'``
        for non-negative matrix factorization, or ``'ica'`` for
        independent component analysis. Default is ``'nmf'`` (recommended).
    lowmemory_mode : bool, optional
        If ``True``, FISSA will load TIFF files into memory frame-by-frame
        instead of holding the entire TIFF in memory at once. This
        option reduces the memory load, and may be necessary for very
        large inputs. Default is ``False``.
    datahandler : extraction.DataHandlerAbstract or None, optional
        A custom datahandler object for handling ROIs and calcium data can
        be given here. See :mod:`extraction` for example datahandler
        classes. The default datahandler is
        :class:`~extraction.DataHandlerTifffile`.
        Note: if `datahandler` is set, the `lowmemory_mode` parameter is
        ignored.
    """

    def __init__(self, images, rois, folder=None, nRegions=4,
                 expansion=1, alpha=0.1, ncores_preparation=None,
                 ncores_separation=None, method='nmf',
                 lowmemory_mode=False, datahandler=None):

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

        if datahandler is not None:
            self.datahandler = datahandler
        elif lowmemory_mode:
            self.datahandler = extraction.DataHandlerTifffileLazy()
        else:
            self.datahandler = extraction.DataHandlerTifffile()

        # define class variables
        self.folder = folder
        self.clear()
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

    def clear(self):
        """
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
        """
        Clear prepared data, and all data downstream of prepared data.

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
        """
        Load data from cache file in npz format.

        .. versionadded:: 1.0.0

        Parameters
        ----------
        path : str or None, optional
            Path to cache file (.npz format) or a directory containing
            ``"preparation.npz"`` and/or ``"separated.npz"`` files.
            If ``None`` (default), the `folder` parameter which was provided
            when the object was initialised is used (``self.folder``).
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
        """Prepare and extract the data to be separated.

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
        # define inputs
        inputs = [[]] * self.nTrials
        for trial in range(self.nTrials):
            inputs[trial] = [self.images[trial], self.rois[trial],
                             self.nRegions, self.expansion, self.datahandler]

        # Check whether we should use multiprocessing
        use_multiprocessing = (
            (self.ncores_preparation is None or self.ncores_preparation > 1)
        )
        # Do the extraction
        if use_multiprocessing and sys.version_info < (3, 0):
            # define pool
            pool = Pool(self.ncores_preparation)

            # run extraction
            results = pool.map(extract_func, inputs)
            pool.close()
            pool.join()

        elif use_multiprocessing:
            with Pool(self.ncores_preparation) as pool:
                # run extraction
                results = pool.map(extract_func, inputs)

        else:
            results = [extract_func(inputs[trial]) for trial in range(self.nTrials)]

        # get number of cells
        nCell = len(results[0][1])

        # predefine data structures
        raw = [[None for t in range(self.nTrials)] for c in range(nCell)]
        raw = np.asarray(raw)
        roi_polys = np.copy(raw)

        # Set outputs
        for trial in range(self.nTrials):
            self.means.append(results[trial][2])
            for cell in range(nCell):
                raw[cell][trial] = results[trial][0][cell]
                roi_polys[cell][trial] = results[trial][1][cell]

        self.nCell = nCell  # number of cells
        self.raw = raw
        self.roi_polys = roi_polys
        # Maybe save to cache file
        if self.folder is not None:
            self.save_prep()

    def save_prep(self, destination=None):
        """
        Save prepared raw signals, extracted from images, to an npz file.

        .. versionadded:: 1.0.0

        Parameters
        ----------
        destination : str or None, optional
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
        """
        Separate all the trials with FISSA algorithm.

        After running ``separate``, data can be found as follows:

        self.sep
            Raw separation output, without being matched. Signal ``i`` for
            a specific cell and trial can be found as
            ``self.sep[cell][trial][i,:]``.
        self.result
            Final output, in order of presence in cell ROI.
            Signal ``i`` for a specific cell and trial can be found at
            ``self.result[cell][trial][i, :]``.
            Note that the ordering is such that ``i = 0`` is the signal
            most strongly present in the ROI, and subsequent entries
            are in diminishing order.
        self.mixmat
            The mixing matrix, which maps how to from ``self.raw`` to
            ``self.separated``.
        self.info
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
        # predefine data structures
        sep = [[None for t in range(self.nTrials)]
               for c in range(self.nCell)]
        sep = np.asarray(sep)
        result = np.copy(sep)
        mixmat = np.copy(sep)
        info = np.copy(sep)

        # loop over cells to define function inputs
        inputs = [[]] * int(self.nCell)
        for cell in range(self.nCell):
            # initiate concatenated data
            X = np.concatenate(self.raw[cell], axis=1)

            # check for below 0 values
            if X.min() < 0:
                warnings.warn('Found values below zero in signal, ' +
                              'setting minimum to 0.')
                X -= X.min()

            # update inputs
            inputs[cell] = [X, self.alpha, self.method, cell]

        # Check whether we should use multiprocessing
        use_multiprocessing = (
            (self.ncores_separation is None or self.ncores_separation > 1)
        )
        # Do the extraction
        if use_multiprocessing and sys.version_info < (3, 0):
            # define pool
            pool = Pool(self.ncores_separation)

            # run separation
            results = pool.map(separate_func, inputs)
            pool.close()
            pool.join()

        elif use_multiprocessing:
            with Pool(self.ncores_separation) as pool:
                # run separation
                results = pool.map(separate_func, inputs)
        else:
            results = [separate_func(inputs[cell]) for cell in range(self.nCell)]

        # read results
        for cell in range(self.nCell):
            curTrial = 0
            Xsep, Xmatch, Xmixmat, convergence = results[cell]
            for trial in range(self.nTrials):
                nextTrial = curTrial + self.raw[cell][trial].shape[1]
                sep[cell][trial] = Xsep[:, curTrial:nextTrial]
                result[cell][trial] = Xmatch[:, curTrial:nextTrial]
                curTrial = nextTrial

                # store other info
                mixmat[cell][trial] = Xmixmat
                info[cell][trial] = convergence

        # Set outputs
        self.info = info
        self.mixmat = mixmat
        self.sep = sep
        self.result = result
        # Maybe save to cache file
        if self.folder is not None:
            self.save_separated()

    def save_separated(self, destination=None):
        """
        Save separated signals to an npz file.

        .. versionadded:: 1.0.0

        Parameters
        ----------
        destination : str or None, optional
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
        """
        Calculate deltaf/f0 for raw and result traces.

        The results can be accessed as ``self.deltaf_raw`` and
        ``self.deltaf_result``.

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
            and df/f0 value will be relative to the trial-specific f0.
            Default is ``True``.
        """
        deltaf_raw = [[None for t in range(self.nTrials)]
                      for c in range(self.nCell)]
        deltaf_raw = np.asarray(deltaf_raw)
        deltaf_result = np.copy(deltaf_raw)

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
        """Save the results to a MATLAB file.

        This will generate a .mat file which can be loaded into MATLAB to
        provide structs: ROIs, result, raw.

        If df/f0 was calculated, these will also be stored as ``df_result``
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
    """
    Functional interface to run FISSA.

    .. versionadded:: 1.0.0

    Parameters
    ----------
    images : str or list
        The raw recording data.
        Should be one of:

        - the path to a directory containing TIFF files (string),
        - an explicit list of TIFF files (list of strings),
        - a list of array-like data already loaded into memory, each shaped
          ``(frames, y-coords, x-coords)``.

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

    folder : str or None, optional
        Output path to a directory in which the extracted data will
        be stored. If ``None`` (default), the data will not be cached.
    freq : float or None, optional
        Imaging frequency, in Hz. Required if `return_deltaf` is ``True``.
    return_deltaf : bool, optional
        Whether to return df/f0. Otherwise, the decontaminated signal is
        returned scaled against the raw recording. Default is ``False``.
    deltaf_across_trials : bool, optional
        If ``True``, we estimate a single baseline f0 value across all
        trials when computing df/f0. If ``False``, each trial will have their
        own baseline f0, and df/f0 value will be relative to the trial-specific
        f0. Default is ``True``.
    export_to_matlab : bool or str or None, optional
        Whether to export the data to a MATLAB-compatible .mat file.
        If `export_to_matlab` is a string, it is used as the path to the output
        file. If `export_to_matlab` is ``True``, the matfile is saved to the
        default path of ``"matlab.mat"`` within the `folder` directory, and
        `folder` must be set. If this is ``None``, the matfile is exported to
        the default path if `folder` is set, and otherwise is not exported.
        Default is ``False``.
    **kwargs
        Additional keyword arguments as per `Experiment`.

    Returns
    -------
    result : 2d numpy.ndarray of 2d numpy.ndarrays of np.float64
        The vector `result[c, t][0, :]` is the trace from cell `c` in
        trial `t`. If `return_deltaf` is ``True``, this is df/f0.
        Otherwise, it is the decontaminated signal scaled as per the raw
        signal.
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
