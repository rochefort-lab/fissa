"""Main user interface for FISSA.

Authors:
    - Sander W Keemink (swkeemink@scimail.eu)
    - Scott C Lowe
"""

from __future__ import print_function

import collections
import glob
import os.path
import warnings

import numpy as np
from scipy.io import savemat

from . import datahandler
from . import deltaf
from . import neuropil as npil
from . import roitools

try:
    from multiprocessing import Pool
    has_multiprocessing = True
except BaseException:
    warnings.warn('Multiprocessing library is not installed, using single ' +
                  'core instead. To use multiprocessing install it by: ' +
                  'pip install multiprocessing')
    has_multiprocessing = False


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
    dict
        Data across cells.
    dict
        Polygons for each ROI.
    """
    image = inputs[0]
    rois = inputs[1]
    nNpil = inputs[2]
    expansion = inputs[3]

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
        roi_polys[cell] = [''] * len(masks)
        for i in range(len(masks)):
            roi_polys[cell][i] = roitools.find_roi_edge(masks[i])

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
    numpy.ndarray
        The raw separated traces.
    numpy.ndarray
        The separated traces matched to the primary signal.
    numpy.ndarray
        Mixing matrix.
    dict
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
    """Does all the steps for FISSA."""

    def __init__(self, images, rois, folder, nRegions=4,
                 expansion=1, alpha=0.1, ncores_preparation=None,
                 ncores_separation=None, method='nmf',
                 lowmemory_mode=False, datahandler_custom=None,
                 **kwargs):
        """Initialisation. Set the parameters for your Fissa instance.

        Parameters
        ----------
        images : str or list
            The raw recording data.
            Should be one of:

            - the path to a directory containing TIFF files (string),
            - an explicit list of TIFF files (list of strings),
            - a list of array_like data already loaded into memory, each shaped
              `(frames, y-coords, x-coords)`.

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

        folder : str
            Output path to a directory in which the extracted data will
            be stored.
        nRegions : int, optional
            Number of neuropil regions to draw. Use a higher number for
            densely labelled tissue. Default is 4.
        expansion : float, optional
            Expansion factor for the neuropil region, relative to the
            ROI area. Default is 1. The total neuropil area will be
            `nRegions * expansion * area(ROI)`.
        alpha : float, optional
            Sparsity regularizaton weight for NMF algorithm. Set to zero to
            remove regularization. Default is 0.1. (Not used for ICA method.)
        ncores_preparation : int, optional (default: None)
            Sets the number of subprocesses to be used during the data
            preparation steps (ROI and subregions definitions, data
            extraction from tifs, etc.).
            If set to `None` (default), there will be as many subprocesses
            as there are threads or cores on the machine. Note that this
            behaviour can, especially for the data preparation step,
            be very memory-intensive.
        ncores_separation : int, optional (default: None)
            Same as `ncores_preparation`, but for the separation step.
            Note that this step requires less memory per subprocess, and
            hence can often be set higher than `ncores_preparation`.
        method : {'nmf', 'ica'}, optional
            Which blind source-separation method to use. Either `'nmf'`
            for non-negative matrix factorization, or `'ica'` for
            independent component analysis. Default (recommended) is
            `'nmf'`.
        lowmemory_mode : bool, optional
            If `True`, FISSA will load TIFF files into memory frame-by-frame
            instead of holding the entire TIFF in memory at once. This
            option reduces the memory load, and may be necessary for very
            large inputs. Default is `False`.
        datahandler_custom : object, optional
            A custom datahandler for handling ROIs and calcium data can
            be given here. See datahandler.py (the default handler) for
            an example.

        """
        if isinstance(images, str):
            self.images = sorted(glob.glob(images + '/*.tif*'))
        elif isinstance(images, list):
            self.images = images
        else:
            raise ValueError('images should either be string or list')

        if isinstance(rois, str):
            if rois[-3:] == 'zip':
                self.rois = [rois] * len(self.images)
            else:
                self.rois = sorted(glob.glob(rois + '/*.zip'))
        elif isinstance(rois, list):
            self.rois = rois
            if len(rois) == 1:  # if only one roiset is specified
                self.rois *= len(self.images)
        else:
            raise ValueError('rois should either be string or list')
        global datahandler
        if lowmemory_mode:
            from . import datahandler_framebyframe as datahandler
        if datahandler_custom is not None:
            datahandler = datahandler_custom

        # define class variables
        self.folder = folder
        self.raw = None
        self.sep = None
        self.result = None
        self.nRegions = nRegions
        self.expansion = expansion
        self.alpha = alpha
        self.nTrials = len(self.images)  # number of trials
        self.means = []
        self.ncores_preparation = ncores_preparation
        self.ncores_separation = ncores_separation
        self.method = method

        # check if any data already exists
        if not os.path.exists(folder):
            os.makedirs(folder)
        if os.path.isfile(folder + '/preparation.npy'):
            if os.path.isfile(folder + '/separated.npy'):
                self.separate()
            else:
                self.separation_prep()

    def separation_prep(self, redo=False):
        """Prepare and extract the data to be separated.

        For each trial, performs the following steps:

        - Load in data as arrays
        - Load in ROIs as masks
        - Grow and seaparate ROIs to define neuropil regions
        - Using neuropil and original ROI regions, extract traces from data

        After running this you can access the raw data (i.e. pre-separation)
        as `self.raw` and `self.rois`. self.raw is a list of arrays.
        `self.raw[cell][trial]` gives you the traces of a specific cell and
        trial, across cell and neuropil regions. `self.roi_polys` is a list of
        lists of arrays. `self.roi_polys[cell][trial][region][0]` gives you the
        polygon for the region for a specific cell, trial and region. `region=0`
        is the cell, and `region>0` gives the different neuropil regions.
        For separateable masks, it is possible multiple outlines are found,
        which can be accessed as `self.roi_polys[cell][trial][region][i]`,
        where `i` is the outline index.

        Parameters
        ----------
        redo : bool, optional
            If `False`, we load previously prepared data when possible.
            If `True`, we re-run the preparation, even if it has previously
            been run. Default is `False`.

        """
        # define filename where data will be present
        fname = self.folder + '/preparation.npy'

        # try to load data from filename
        if not redo:
            try:
                nCell, raw, roi_polys = np.load(fname)
                print('Reloading previously prepared data...')
            except BaseException:
                redo = True

        if redo:
            print('Doing region growing and data extraction....')
            # define inputs
            inputs = [0] * self.nTrials
            for trial in range(self.nTrials):
                inputs[trial] = [self.images[trial], self.rois[trial],
                                 self.nRegions, self.expansion]

            # Do the extraction
            if has_multiprocessing:
                # define pool
                pool = Pool(self.ncores_separation)

                # run extraction
                results = pool.map(extract_func, inputs)
                pool.close()
            else:
                results = [0] * self.nTrials
                for trial in range(self.nTrials):
                    results[trial] = extract_func(inputs[trial])

            # get number of cells
            nCell = len(results[0][1])

            # predefine data structures
            raw = [[None for t in range(self.nTrials)] for c in range(nCell)]
            roi_polys = np.copy(raw)

            # store results
            for trial in range(self.nTrials):
                self.means += [results[trial][2]]
                for cell in range(nCell):
                    raw[cell][trial] = results[trial][0][cell]
                    roi_polys[cell][trial] = results[trial][1][cell]

            # save
            np.save(fname, (nCell, raw, roi_polys))

        # store relevant info
        self.nCell = nCell  # number of cells
        self.raw = raw
        self.roi_polys = roi_polys

    def separate(self, redo_prep=False, redo_sep=False):
        """Separate all the trials with FISSA algorithm.

        After running `separate`, data can be found as follows:

        self.sep
            Raw separation output, without being matched. Signal `i` for
            a specific cell and trial can be found as
            `self.sep[cell][trial][i,:]`.
        self.result
            Final output, in order of presence in cell ROI.
            Signal `i` for a specific cell and trial can be found at
            `self.result[cell][trial][i, :]`.
            Note that the ordering is such that `i = 0` is the signal
            most strongly present in the ROI, and subsequent entries
            are in diminishing order.
        self.mixmat
            The mixing matrix (how to go between `self.separated` and
            `self.raw` from the `separation_prep()` function).
        self.info
            Information about separation routine, iterations needed, etc.

        Parameters
        ----------
        redo_prep : bool, optional
            Whether to redo the preparation. Default is `False.` Note that
            if this is true, we set `redo_sep = True` as well.
        redo_sep : bool, optional
            Whether to redo the separation. Default is `False`. Note that
            this parameter is ignored if `redo_prep` is set to `True`.

        """
        # Do data preparation
        self.separation_prep(redo_prep)
        if redo_prep:
            redo_sep = True

        # Define filename to store data in
        fname = self.folder + '/separated.npy'
        if not redo_sep:
            try:
                info, mixmat, sep, result = np.load(fname)
                print('Reloading previously separated data...')
            except BaseException:
                redo_sep = True

        # separate data, if necessary
        if redo_sep:
            print('Doing signal separation....')
            # predefine data structures
            sep = [[None for t in range(self.nTrials)]
                   for c in range(self.nCell)]
            result = np.copy(sep)
            mixmat = np.copy(sep)
            info = np.copy(sep)

            # loop over cells to define function inputs
            inputs = [0] * self.nCell
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

            if has_multiprocessing:
                # define pool
                pool = Pool()

                # run separation
                results = pool.map(separate_func, inputs)
                pool.close()
            else:
                results = [0] * self.nCell
                for cell in range(self.nCell):
                    results[cell] = separate_func(inputs[cell])

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

            # save
            np.save(fname, (info, mixmat, sep, result))

        # store
        self.info = info
        self.mixmat = mixmat
        self.sep = sep
        self.result = result

    def calc_deltaf(self, freq, use_raw_f0=True, across_trials=True):
        """Calculate deltaf/f0 for raw and result traces.

        The results can be accessed as self.deltaf_raw and self.deltaf_result.
        self.deltaf_raw is only the ROI trace instead of the traces across all
        subregions.

        Parameters
        ----------
        freq : float
            Imaging frequency, in Hz.
        use_raw_f0 : bool, optional
            If `True` (default), use an f0 estimate from the raw ROI trace
            for both raw and result traces. If `False`, use individual f0
            estimates for each of the traces.
        across_trials : bool, optional
            If `True`, we estimate a single baseline f0 value across all
            trials. If `False`, each trial will have their own baseline f0,
            and df/f0 value will be relative to the trial-specific f0.
            Default is `True`.

        """
        deltaf_raw = [[None for t in range(self.nTrials)]
                      for c in range(self.nCell)]
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

    def save_to_matlab(self):
        """Save the results to a matlab file.

        Can be found in `folder/matlab.mat`.

        This will give you a filename.mat file which if loaded in Matlab gives
        the following structs: ROIs, result, raw.

        If df/f0 was calculated, these will also be stored as `df_result`
        and `df_raw`, which will have the same format as result and raw.

        These can be interfaced with as follows, for cell 0, trial 0:

        - `ROIs.cell0.trial0{1}` polygon for the ROI
        - `ROIs.cell0.trial0{2}` polygon for first neuropil region
        - `result.cell0.trial0(1,:)` final extracted cell signal
        - `result.cell0.trial0(2,:)` contaminating signal
        - `raw.cell0.trial0(1,:)` raw measured celll signal
        - `raw.cell0.trial0(2,:)` raw signal from first neuropil region
        """
        # define filename
        fname = self.folder + '/matlab.mat'

        # initialize dictionary to save
        M = collections.OrderedDict()
        M['ROIs'] = collections.OrderedDict()
        M['raw'] = collections.OrderedDict()
        M['result'] = collections.OrderedDict()

        # loop over cells and trial
        for cell in range(self.nCell):
            # get current cell label
            c_lab = 'cell' + str(cell)
            # update dictionary
            M['ROIs'][c_lab] = collections.OrderedDict()
            M['raw'][c_lab] = collections.OrderedDict()
            M['result'][c_lab] = collections.OrderedDict()
            for trial in range(self.nTrials):
                # get current trial label
                t_lab = 'trial' + str(trial)
                # update dictionary
                M['ROIs'][c_lab][t_lab] = self.roi_polys[cell][trial]
                M['raw'][c_lab][t_lab] = self.raw[cell][trial]
                M['result'][c_lab][t_lab] = self.result[cell][trial]

        savemat(fname, M)
