"""Basic FISSA usage."""
# FISSA toolbox import
import fissa.core as fissa

# data location
rois = ['../exampleData/20150429.zip']
images = '../exampleData/20150529'

# extraction location
folder = 'fissa_example'
# make sure you use a different folder for each experiment!

# experiment definition
experiment = fissa.Experiment(images, rois, folder)

# do separation
experiment.separate()

# (optional) export to matlab
experiment.save_to_matlab()
