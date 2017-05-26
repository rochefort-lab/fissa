# FISSA toolbox import
import fissa.core as fissa

# data location
rois = ['../exampleData/MC_20150429_A01.zip']
images = '../exampleData/20150529_mini'

# experiment definition
experiment = fissa.Experiment(images, rois)

# do separation
filename = 'example_experiment'
# make sure you use a different filename for each experiment!
experiment.separate(filename)

# (optional) export to matlab
filename = 'extracted.mat'
experiment.save_to_matlab(filename)
