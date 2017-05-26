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
    multiprocessing = True
except:
    warnings.warn('Multiprocessing library is not installed, using single ' +
                  'core instead. To use multiprocessing install it by: ' +
                  'pip install multiprocessing')
    multiprocessing = False


def extract_func(lst):
    """Extraction function for multiprocessing.

    Parameters
    ----------
    lst : list
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
    image = lst[0]
    rois = lst[1]

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


def separate_func(lst):
    """Extraction function for multiprocessing.

    Parameters
    ----------
    lst : list
        list of inputs
        [0] : Array with signals to separate
        [1] : current ROI number

    Returns
    -------
    some output

    """
    X = lst[0]
    Xsep, Xmatch, Xmixmat, convergence = npil.separate(
                X, 'nmf', maxiter=20000, tol=1e-4, maxtries=1)
    ROInum = lst[1]
    print 'Finished ROI number ' + str(ROInum)
    return Xsep, Xmatch, Xmixmat, convergence


class Experiment():
    """Does all the steps for FISSA in one swoop."""

    def __init__(self, images, rois, filename, nRegions=4, **params):
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
        filename : string
            Where to store the extracted data.
            Should be of form or 'folder/file' without an extension.
        nRegions : int, optional (default: 4)
            Number of neuropil regions to draw. Use higher number for densely
            labelled tissue.

        TOOD:
        * inputs such as imaging frequency, number of neuropil regions,
        general FISSA options, etc
        """
        if type(images) == str:
            self.images = sorted(glob.glob(images+'/*.tif'))
        elif type(images) == list:
            self.images = images
        else:
            raise ValueError('images should either be string or list')

        if type(rois) == str:
            self.rois = sorted(glob.glob(images+'/*.zip'))
        elif type(rois) == list:
            self.rois = rois
            if len(rois) == 1:  # if only one roiset is specified
                self.rois *= len(self.images)
        else:
            raise ValueError('rois should either be string or list')

        if os.path.isfile(filename):
            self.separate()

        self.raw = None
        self.sep = None
        self.matched = None
        self.nRegions = nRegions
        self.nTrials = len(self.images)  # number of trials
        self.means = []

    def separation_prep(self, filename='default.npy', redo=False):
        """This will prepare the data to be separated in the following steps,
        per trial:
        * load in data as arrays
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
        filename : string, optional
            Where to store the extracted data.
            Should be of form 'folder/file.npy'.
        redo : bool, optional
            Whether to redo the extraction, i.e. replace the filename file.
        """
        # try to load data from filename
        if not redo:
            try:
                nCell, data, roi_polys = np.load(filename)
            except:
                print filename + ' does not exist yet, doing extraction...'
                redo = True
        if redo:
            print 'Doing region growing and data extraction....'
            # predefine data structures
            data = {}
            roi_polys = {}

            # across trials
            if multiprocessing:
                # define pool
                pool = Pool()

                # run extraction
                inputs = []
                for trial in range(self.nTrials):
                    inputs += [[self.images[trial], self.rois[trial]]]
                results = pool.map(extract_func, inputs)

                # get number of cells
                nCell = len(results[0][1])

                # store results
                for trial in range(self.nTrials):
                    for cell in range(nCell):
                        data[cell, trial] = results[trial][0][cell]
                        roi_polys[cell, trial] = results[trial][1][cell]
                pool.close()
            else:
                for trial in range(self.nTrials):
                    # get data as arrays and rois as maks
                    curdata = datahandler.image2array(self.images[trial])
                    base_masks = datahandler.rois2masks(self.rois[trial],
                                                        curdata.shape[1:])
                    curdata = curdata

                    # get neuropil masks and extract signals
                    for cell in range(len(base_masks)):
                        # neuropil masks
                        npil_masks = roitools.getmasks_npil(
                                                        base_masks[cell],
                                                        nNpil=self.nRegions,
                                                        expansion=5)
                        # add all current masks together
                        masks = [base_masks[cell]]+npil_masks

                        # extract traces
                        data[cell, trial] = datahandler.extracttraces(curdata,
                                                                      masks)

                        # store ROI outlines
                        roi_polys[cell, trial] = ['']*len(masks)
                        for i in range(len(masks)):
                            roi_polys[cell, trial][i] = roitools.find_roi_edge(
                                                                      masks[i])

                nCell = len(base_masks)

            # save
            np.save(filename, (nCell, data, roi_polys))

        # store relevant info
        self.nCell = nCell  # number of cells
        self.raw = data
        self.roi_polys = roi_polys

    def separate(self, filename='default.npy', redo=False):
        """Separate all the trials with FISSA algorithm.

        After running separate, data can be found as follows:

        self.sepa
            Raw separation output, without being matched. Signal 'i' for
            a specific cell and trial can be found as
            self.sep[cell,trial][i,:]
        self.matched
            Matched output, in order of presence in cell ROI
            Signal 'i' for a specific cell and trial can be found as
            self.matched[cell,trial][i,:]
        self.mixmat
            The mixing matrix (how to go between self.separated and
            self.raw from the separation_prep function)
        self.info
            Info about separation algorithm, iterations needed, etc.

        Parameters
        ----------
        filename : string, optional
            Where to store the extracted data.
            Should be of form 'folder/file.npy'.
        redo : bool, optional
            Whether to redo the extraction, i.e. replace the filename file.
        """
        # do separation prep (if necessary)
        if self.raw is None or redo:
            self.separation_prep(filename, redo)

        print 'Doing signal separation....'
        # predefine data structures
        sep = {}
        matched = {}
        mixmat = {}
        info = {}
        trial_lens = np.zeros(len(self.images), dtype=int)  # trial lengths

        if multiprocessing:
            # define pool
            pool = Pool()

            # loop over cells to define multiproc function inputs
            inputs = []
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
                inputs += [[X, cell]]

            # run separation
            results = pool.map(separate_func, inputs)

            # read results
            for cell in range(self.nCell):
                curTrial = 0
                Xsep, Xmatch, Xmixmat, convergence = results[cell]
                for trial in range(self.nTrials):
                    nextTrial = curTrial+trial_lens[trial]
                    sep[cell, trial] = Xsep[:, curTrial:nextTrial]
                    matched[cell, trial] = Xmatch[:, curTrial:nextTrial]
                    curTrial = nextTrial

                    # store other info
                    mixmat[cell, trial] = Xmixmat
                    info[cell, trial] = convergence
        else:
            # loop over cells
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

                # do FISSA
                Xsep, Xmatch, Xmixmat, convergence = npil.separate(
                            X, 'nmf', maxiter=20000, tol=1e-4, maxtries=1)

                # separate by trial again and store
                curTrial = 0  # trial count
                for trial in range(self.nTrials):
                    nextTrial = curTrial+trial_lens[trial]
                    sep[cell, trial] = Xsep[:, curTrial:nextTrial]
                    matched[cell, trial] = Xmatch[:, curTrial:nextTrial]
                    curTrial = nextTrial

                    # store other info
                    mixmat[cell, trial] = Xmixmat
                    info[cell, trial] = convergence

        # store
        self.info = info
        self.mixmat = mixmat
        self.sep = sep
        self.matched = matched

    def save_to_matlab(self, filename='default.mat'):
        """Save the results to a matlab file.

        Can be accessed as...

        """
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
                M['fissa'][c_lab][t_lab] = self.matched[cell, trial]
        savemat(filename, M)


def run_fissa(*args, **kwargs):
    """Shorthand for making Fissa instance and running it on your dataset."""
    # clive = Fissa(*args, **kwargs)
    raise NotImplementedError()
