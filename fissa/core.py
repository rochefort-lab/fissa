"""Main user interface for FISSA.

Authors: Sander Keemink (swkeemink@scimail.eu) and Scott Lowe
"""

import datahandler
import roitools
import glob
import warnings
import os.path
import numpy as np
import neuropil as npil
from scipy.io import savemat
try:
    from multiprocessing import Pool
    has_multiprocessing = True
except:
    warnings.warn('Multiprocessing library is not installed, using single ' +
                  'core instead. To use multiprocessing install it by: ' +
                  'pip install multiprocessing')
    has_multiprocessing = False


def extract_func(inputs):
    """Extraction function for multiprocessing.

    Parameters
    ----------
    inputs : list
        list of inputs
        [0] : image array
        [1] : the rois

    Returns
    -------
    dictionary
        dictionary containing data across cells
    dictionary
        dictionary containing polygons for each ROI
    """
    image = inputs[0]
    rois = inputs[1]
    # get data as arrays and rois as masks
    curdata = datahandler.image2array(image)
    base_masks = datahandler.rois2masks(rois, curdata.shape[1:])

    # get the mean image
    mean = curdata.mean(axis=0)

    # predefine dictionaries
    data = {}
    roi_polys = {}

    # get neuropil masks and extract signals
    for cell in range(len(base_masks)):
        # neuropil masks
        npil_masks = roitools.getmasks_npil(base_masks[cell], nNpil=4,
                                            expansion=6)
        # add all current masks together
        masks = [base_masks[cell]]+npil_masks

        # extract traces
        data[cell] = datahandler.extracttraces(curdata, masks)

        # store ROI outlines
        roi_polys[cell] = ['']*len(masks)
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
        [1] : current ROI number

    Returns
    -------
    some output

    """
    X = inputs[0]
    Xsep, Xmatch, Xmixmat, convergence = npil.separate(
                X, 'nmf', maxiter=20000, tol=1e-4, maxtries=1)
    ROInum = inputs[1]
    print 'Finished ROI number ' + str(ROInum)
    return Xsep, Xmatch, Xmixmat, convergence


class Experiment():
    """Does all the steps for FISSA in one swoop."""

    def __init__(self, images, rois, folder, nRegions=4, **params):
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
            Should be the path of a folder with imagej zips,
            an explicit list of imagej zip locations, a list of arrays encoding
            roi polygons, or a list of lists of binary arrays representing roi
            masks.
            Should be either a single roiset for all trials, or a different
            roiset for each trial.
        folder : string
            Where to store the extracted data.
        nRegions : int, optional (default: 4)
            Number of neuropil regions to draw. Use higher number for densely
            labelled tissue.

        TOOD:
        * inputs such as imaging frequency, number of neuropil regions,
        general FISSA options, etc
        """
        if isinstance(images, str):
            self.images = sorted(glob.glob(images+'/*.tif*'))
        elif isinstance(images, list):
            self.images = images
        else:
            raise ValueError('images should either be string or list')

        if isinstance(rois, str):
            self.rois = sorted(glob.glob(images+'/*.zip'))
        elif isinstance(rois, list):
            self.rois = rois
            if len(rois) == 1:  # if only one roiset is specified
                self.rois *= len(self.images)
        else:
            raise ValueError('rois should either be string or list')

        # define class variables
        self.folder = folder
        self.raw = None
        self.sep = None
        self.result = None
        self.nRegions = nRegions
        self.nTrials = len(self.images)  # number of trials
        self.means = []
        self.ncores_preparation = None
        self.ncores_separation = None

        # check if any data already exists
        if not os.path.exists(folder):
            os.makedirs(folder)
        if os.path.isfile(folder + '/preparation.npy'):
            if os.path.isfile(folder+'/separated.npy'):
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
        as self.raw and self.rois. self.raw is a dictionary of arrays.
        self.raw[cell,trial] gives you the traces of a specific cell and trial,
        across cell and neuropil regions. self.roi_polys is a dictioary of
        lists of arrays. self.roi_polys[cell,trial][region][0] gives you the
        polygon for the region for a specific cell, trial and region. region=0
        is the cell, and region>0 gives the different neuropil regions.
        For separateable masks, it is possible multiple outlines are found,
        which can be accessed as self.roi_polys[cell,trial][region][i],
        where 'i' is the outline index.

        Parameters
        ----------
        redo : bool
            Redo the preparation even if done before
        """
        # define filename where data will be present
        fname = self.folder+'/preparation.npy'

        # try to load data from filename
        if not redo:
            try:
                nCell, data, roi_polys = np.load(fname)
                print 'Reloading previously prepared data...'
            except:
                redo = True

        if redo:
            print 'Doing region growing and data extraction....'
            # predefine data structures
            data = {}
            roi_polys = {}

            # define inputs
            inputs = [0]*self.nTrials
            for trial in range(self.nTrials):
                inputs[trial] = [self.images[trial], self.rois[trial]]

            # Do the extraction
            if has_multiprocessing:
                # define pool
                pool = Pool(self.ncores_separation)

                # run extraction
                results = pool.map(extract_func, inputs)
                pool.close()
            else:
                results = [0]*self.nTrials
                for trial in range(self.nTrials):
                    results[trial] = extract_func(inputs[trial])

            # get number of cells
            nCell = len(results[0][1])

            # store results
            for trial in range(self.nTrials):
                self.means += [results[trial][2]]
                for cell in range(nCell):
                    data[cell, trial] = results[trial][0][cell]
                    roi_polys[cell, trial] = results[trial][1][cell]

            # save
            np.save(fname, (nCell, data, roi_polys))

        # store relevant info
        self.nCell = nCell  # number of cells
        self.raw = data
        self.roi_polys = roi_polys

    def separate(self, redo_prep=False, redo_sep=False):
        """Separate all the trials with FISSA algorithm.

        After running separate, data can be found as follows:

        self.sep
            Raw separation output, without being matched. Signal 'i' for
            a specific cell and trial can be found as
            self.sep[cell,trial][i,:]
        self.result
            Final output, in order of presence in cell ROI.
            Signal 'i' for a specific cell and trial can be found as
            self.result[cell,trial][i,:]
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
            except:
                redo_sep = True

        # separate data, if necessary
        if redo_sep:
            print 'Doing signal separation....'
            # predefine data structures
            sep = {}
            result = {}
            mixmat = {}
            info = {}
            trial_lens = np.zeros(len(self.images), dtype=int)  # trial lengths

            # loop over cells to define function inputs
            inputs = [0]*self.nCell
            for cell in range(self.nCell):
                # get first trial data
                cur_signal = self.raw[cell, 0]

                # initiate concatenated data
                X = cur_signal
                trial_lens[0] = cur_signal.shape[1]

                # concatenate all trials
                for trial in range(1, self.nTrials):
                    # get current trial data
                    cur_signal = self.raw[cell, trial]

                    # concatenate
                    X = np.concatenate((X, cur_signal), axis=1)

                    trial_lens[trial] = cur_signal.shape[1]

                # check for below 0 values
                if X.min() < 0:
                    X -= X.min()

                # update inputs
                inputs[cell] = [X, cell]

            if has_multiprocessing:
                # define pool
                pool = Pool()

                # run separation
                results = pool.map(separate_func, inputs)
                pool.close()
            else:
                results = [0]*self.nCell
                for cell in range(self.nCell):
                    results[cell] = separate_func(inputs[cell])

            # read results
            for cell in range(self.nCell):
                curTrial = 0
                Xsep, Xmatch, Xmixmat, convergence = results[cell]
                for trial in range(self.nTrials):
                    nextTrial = curTrial+trial_lens[trial]
                    sep[cell, trial] = Xsep[:, curTrial:nextTrial]
                    result[cell, trial] = Xmatch[:, curTrial:nextTrial]
                    curTrial = nextTrial

                    # store other info
                    mixmat[cell, trial] = Xmixmat
                    info[cell, trial] = convergence

            # save
            np.save(fname, (info, mixmat, sep, result))

        # store
        self.info = info
        self.mixmat = mixmat
        self.sep = sep
        self.result = result

    def save_to_matlab(self):
        """Save the results to a matlab file.

        Can be found in 'folder/matlab.mat'.

        This will give you a filename.mat file which if loaded in Matlab gives
        the following structs: ROIs, fissa, raw.

        These can be interfaced with as follows, for cell 0, trial 0:
            ROIs.cell0.trial0{1} % polygon for the ROI
            ROIs.cell0.trial0{2} % polygon for first neuropil region
            fissa.cell0.trial0(1,:) % final extracted cell signal
            fissa.cell0.trial0(2,:) % contaminating signal
            raw.cell0.trial0(1,:) % raw measured celll signal
            raw.cell0.trial0(2,:) % raw signal from first neuropil region
        """
        # define filename
        fname = self.folder + '/matlab.mat'

        # initialize dictionary to save
        M = {}
        M['ROIs'] = {}
        M['raw'] = {}
        M['fissa'] = {}

        # loop over cells and trial
        for cell in range(self.nCell):
            # get current cell label
            c_lab = 'cell'+str(cell)
            # update dictionary
            M['ROIs'][c_lab] = {}
            M['raw'][c_lab] = {}
            M['fissa'][c_lab] = {}
            for trial in range(self.nTrials):
                # get current trial label
                t_lab = 'trial'+str(trial)
                # update dictionary
                M['ROIs'][c_lab][t_lab] = self.roi_polys[cell, trial]
                M['raw'][c_lab][t_lab] = self.raw[cell, trial]
                M['fissa'][c_lab][t_lab] = self.result[cell, trial]

        savemat(fname, M)


def run_fissa(*args, **kwargs):
    """Shorthand for making Fissa instance and running it on your dataset."""
    # clive = Fissa(*args, **kwargs)
    raise NotImplementedError()
