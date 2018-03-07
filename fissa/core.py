"""Main user interface for FISSA.

Authors: Sander Keemink (swkeemink@scimail.eu) and Scott Lowe
"""
import datahandler
import collections
import roitools
import glob
import warnings
import os.path
import numpy as np
import neuropil as npil
from scipy.io import savemat
import deltaf
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
        [0] : image array
        [1] : the rois
        [2] : number of neuropil regions
        [3] : how much larger neuropil region should be then central ROI

    Returns
    -------
    dictionary
        dictionary containing data across cells
    dictionary
        dictionary containing polygons for each ROI
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
        [0] : Array with signals to separate
        [1] : Alpha input to npil.separate
        [2] : current ROI number

    Returns
    -------
    some output

    """
    X = inputs[0]
    alpha = inputs[1]
    method = inputs[2]
    Xsep, Xmatch, Xmixmat, convergence = npil.separate(
        X, method, maxiter=20000, tol=1e-4, maxtries=1, alpha=alpha
    )
    ROInum = inputs[3]
    print 'Finished ROI number ' + str(ROInum)
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
        images : string or list
            The raw images data.
            Should be the path to a folder with tiffs,
            an explicit list of tiff locations (strings),
            or a list of already loaded in arrays.
            Each tiff/array is seen as a single trial.
            Non tiff data should be formatted as (frames,y-coords,x-coords)
        rois : string or list
            The roi definitions.
            Should be the path of a folder with imagej zips, the explicit path
            of a single imagej zip, an explicit list of imagej zip locations,
            a list of arrays encoding roi polygons, or a list of lists of
            binary arrays representing roi masks.
            Should be either a single roiset for all trials, or a different
            roiset for each trial.
        folder : string
            Where to store the extracted data.
        nRegions : int, optional (default: 4)
            Number of neuropil regions to draw. Use higher number for densely
            labelled tissue.
        expansion : float, optional (default: 1)
            How much larger to make each neuropil subregion than ROI area.
            Full neuropil area will be nRegions*expansion*ROI area
        alpha : float, optional (default: 0.1)
            Sparsity constraint for NMF
        ncores_preparation : int, optional (default: None)
            Sets the number of processes to be used for data preparation
            (ROI and subregions definitions, data extraction from tifs,
            etc.)
            By default FISSA uses all the available processing threads.
            This can, especially for the data preparation step,
            quickly fill up your memory.
        ncores_separation : int, optional (default: None)
            Same as ncores_preparation, but for the separation step.
            As a rule, this can be set higher than ncores_preparation, as
            the separation step takes much less memory.
        method : string, optional
            Either 'nmf' for non-negative matrix factorization, or 'ica' for
            independent component analysis. 'nmf' option recommended.
        lowmemory_mode : bool, optional
            If True, FISSA will load tiff file frame by frame instead of
            whole files at once. This will reduce the memory load.
        datahandler_custom : object, optional
            A custom datahandler for handling ROIs and calcium data can
            optionally be given. See datahandler.py for an example.

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
            import datahandler_framebyframe as datahandler
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

        Per trial:
        * Load in data as arrays
        * load in ROIs as masks
        * grow and seaparate ROIs to get neuropil regions
        * using neuropil and original regions, extract traces from data

        After running this you can access the raw data (i.e. pre separation)
        as self.raw and self.rois. self.raw is a list of arrays.
        self.raw[cell][trial] gives you the traces of a specific cell and
        trial, across cell and neuropil regions. self.roi_polys is a list of
        lists of arrays. self.roi_polys[cell][trial][region][0] gives you the
        polygon for the region for a specific cell, trial and region. region=0
        is the cell, and region>0 gives the different neuropil regions.
        For separateable masks, it is possible multiple outlines are found,
        which can be accessed as self.roi_polys[cell][trial][region][i],
        where 'i' is the outline index.

        Parameters
        ----------
        redo : bool
            Redo the preparation even if done before

        """
        # define filename where data will be present
        fname = self.folder + '/preparation.npy'

        # try to load data from filename
        if not redo:
            try:
                nCell, raw, roi_polys = np.load(fname)
                print 'Reloading previously prepared data...'
            except BaseException:
                redo = True

        if redo:
            print 'Doing region growing and data extraction....'
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

        After running separate, data can be found as follows:

        self.sep
            Raw separation output, without being matched. Signal 'i' for
            a specific cell and trial can be found as
            self.sep[cell][trial][i,:]
        self.result
            Final output, in order of presence in cell ROI.
            Signal 'i' for a specific cell and trial can be found as
            self.result[cell][trial][i,:]
            i = 0 is most strongly present signal
            i = 1 less so, etc.
        self.mixmat
            The mixing matrix (how to go between self.separated and
            self.raw from the separation_prep function)
        self.info
            Info about separation algorithm, iterations needed, etc.

        Parameters
        ----------
        redo_prep : bool, optional
            Whether to redo the preparation. This will always set
            redo_sep = True as well.
        redo_sep : bool, optional
            Whether to redo the separation

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
                print 'Reloading previously separated data...'
            except BaseException:
                redo_sep = True

        # separate data, if necessary
        if redo_sep:
            print 'Doing signal separation....'
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
            Imaging frequency in Hz.
        use_raw_f0 : bool
            If True (default), use the f0 from the raw ROI trace for both
            raw and result traces. If False, use the f0 of each trace.
        across_trials : bool
            Whether to do calculate the deltaf/f0 across all trials (default),
            or per trial.

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

        Can be found in 'folder/matlab.mat'.

        This will give you a filename.mat file which if loaded in Matlab gives
        the following structs: ROIs, result, raw.

        If df/f0 was calculated, these will also be stored as df_result
        and df_raw, which will have the same format as result and raw.

        These can be interfaced with as follows, for cell 0, trial 0:
            ROIs.cell0.trial0{1} % polygon for the ROI
            ROIs.cell0.trial0{2} % polygon for first neuropil region
            result.cell0.trial0(1,:) % final extracted cell signal
            result.cell0.trial0(2,:) % contaminating signal
            raw.cell0.trial0(1,:) % raw measured celll signal
            raw.cell0.trial0(2,:) % raw signal from first neuropil region
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


def run_fissa(*args, **kwargs):
    """Shorthand for making Fissa instance and running it on your dataset."""
    # clive = Fissa(*args, **kwargs)
    raise NotImplementedError()
