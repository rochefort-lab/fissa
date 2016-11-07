'''
Main user interface for FISSA.
'''

import neuropil


class Fissa():
    '''
    Does all the steps for FISSA in one swoop.
    '''
    def __init__(self, ):
        '''
        Initialisation. Set the parameters for your Fissa instance.
        You can set all the parameters for all the functions
        '''
        pass

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
