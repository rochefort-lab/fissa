'''
Main user interface for FISSA.
'''

import datahandler
import roitools
import glob


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
        
        data = {}
        roi_polys = {}

        # across trials
        for trial in range(len(self.images)):
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
                roi_polys[cell, trial] = ['']*5
                for i in range(5):
                    roi_polys[cell, trial][i] = roitools.find_roi_edge(masks[i])

        self.raw = data
        self.roi_polys = roi_polys

    def fit(self, data, rois, which_rois=None):
        '''
        Do the computation required to go from inital input to final output.

        Parameters
        ----------
        data : maybe this is the path to a tiff?
            maybe this is a numpy array?
        rois : maybe this is the path to a zip file containing imagej stuff?
            maybe this is a numpy-array of masks?
            maybe this is a list of co-ordinates for the ROI?
        which_rois : list or an int (default all)

        '''
        pass

    def transform(self):
        pass

    def fit_and_transform(self):
        pass

    def main(self, *args, **kwargs):
        return self.fit_and_transform(*args, **kwargs)


def run_fissa(*args, **kwargs):
    '''
    Shorthand for making Fissa instance and running it on
    your dataset.
    '''
    clive = Fissa(*args, **kwargs)
