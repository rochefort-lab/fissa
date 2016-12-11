'''
Main user interface for FISSA.
'''

import datahandler
import roitools
import glob
import numpy as np
import neuropil as npil

class Experiment():
    '''
    Does all the steps for FISSA in one swoop.
    '''

    def __init__(self, images, rois, **params):
        '''
        Initialisation. Set the parameters for your Fissa instance.
        You can set all the parameters for all the functions

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
            roi polygons, or a list of lists of binary arrays representing roi masks.
            Should be either a single roiset for all trials, or a different
            roiset for each trial.

        TOOD:
        * inputs such as imaging frequency, number of neuropil regions,
        general FISSA options, etc
        '''
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
            if len(rois) == 1: # if only one roiset is specified
                self.rois *= len(self.images)
        else:
            raise ValueError('rois should either be string or list')
            
        self.raw = None
        self.sep = None
        self.matched = None
        self.nTrials = len(self.images)  # number of trials

    def separation_prep(self):
        ''' This will prepare the data to be separated in the following steps,
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
        '''
        print 'Doing region growing and data extraction....'        
        # predefine data structures
        data = {}
        roi_polys = {}

        # across trials
        for trial in range(self.nTrials):
            # get data as arrays and rois as maks
            curdata = datahandler.image2array(self.images[trial])
            base_masks = datahandler.rois2masks(self.rois[trial], curdata.shape[1:])

            # get neuropil masks and extract signals
            for cell in range(len(base_masks)):
                # neuropil masks                
                npil_masks = roitools.getmasks_npil(base_masks[cell], nNpil=4,
                                                    iterations=15)
                # add all current masks together
                masks = [base_masks[cell]]+npil_masks
                
                # extract traces
                data[cell,trial] = datahandler.extracttraces(curdata, masks)
                
                # store ROI outlines
                roi_polys[cell, trial] = ['']*len(masks)
                for i in range(len(masks)):
                    roi_polys[cell, trial][i] = roitools.find_roi_edge(masks[i])

        # store relevant info
        self.nCell = len(base_masks)  # number of cells
        self.raw = data
        self.roi_polys = roi_polys

    def separate(self):
        ''' Separate all the trials with FISSA algorithm.

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

        '''
        # do separation prep (if necessary)
        if self.raw is None:
            self.separation_prep()
        
        print 'Doing signal separation for trial....' 
        # predefine data structures
        sep = {}
        matched = {}
        mixmat = {}
        info = {}
        trial_lens = np.zeros(len(self.images),dtype=int)  # trial lengths
        
        # loop over cells
        for cell in range(self.nCell):
            # get first trial data
            cur_signal = self.raw[cell,0]
            
            # low pass filter
            cur_signal = npil.lowPassFilter(cur_signal.T,fs=40,fw_base=5).T
            
            # initiate concatenated data
            X = cur_signal
            trial_lens[0] = cur_signal.shape[1]
            
            # concatenate all trials
            for trial in range(1,self.nTrials):
                # get current trial data
                cur_signal = self.raw[cell,trial]
                
                # low pass filter
                cur_signal = npil.lowPassFilter(cur_signal.T,fs=40,fw_base=5).T
                
                # concatenate
                X = np.concatenate((X,cur_signal),axis=1)
                
                trial_lens[trial] = cur_signal.shape[1]
                
            # check for below 0 values
            if X.min() < 0:
                X -= X.min()
            
            # do FISSA
            Xsep, Xmatch, Xmixmat, convergence = npil.separate(X,
            'nmf_sklearn',maxiter=20000,tol=1e-4,maxtries=1)
            
            # separate by trial again and store
            curTrial = 0  # trial count
            for trial in range(self.nTrials):
                nextTrial = curTrial+trial_lens[trial]
                sep[cell,trial] = Xsep[:,curTrial:nextTrial]
                matched[cell,trial] = Xmatch[:,curTrial:nextTrial]                
                curTrial = nextTrial
            
            # store other info
            mixmat[cell,trial] = Xmixmat
            info[cell,trial] = convergence
            
        # store
        self.info = info
        self.mixmat = mixmat
        self.sep = sep
        self.matched = matched
        
def run_fissa(*args, **kwargs):
    '''
    Shorthand for making Fissa instance and running it on
    your dataset.
    '''
    clive = Fissa(*args, **kwargs)
