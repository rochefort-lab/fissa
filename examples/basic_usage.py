"""Basic FISSA usage.

This file explains step-by-step how to use the FISSA toolbox.

See Basic usage.ipynb and Basic usage.html for a move verbose version.
"""
# FISSA toolbox import
import fissa

# data location
rois = 'exampleData/20150429.zip'
images = 'exampleData/20150529'

# extraction location
folder = 'fissa_example'
# make sure you use a different folder for each experiment!

# experiment definition
experiment = fissa.Experiment(images, rois, folder)

# do separation
experiment.separate(redo_prep=True)

# (optional) export to matlab
experiment.save_to_matlab()
