#!/usr/bin/env python

"""
Basic FISSA usage example.

This file contains a step-by-step example workflow for using the FISSA toolbox
with a class-based/object-oriented interface.

An example notebook is provided here:
https://github.com/rochefort-lab/fissa/blob/master/examples/Basic%20usage.ipynb
"""

import fissa

# On Windows, it is necessary to wrap the script in a __name__ check, so
# that multiprocessing works correctly. Multiprocessing is triggered by the
# experiment.separate() step.
if __name__ == "__main__":
    # Define the data to extract
    rois = "exampleData/20150429.zip"
    images = "exampleData/20150529"

    # Define the name of the experiment extraction location
    output_dir = "fissa_example"
    # Make sure you use a different output path for each experiment you run.

    # Instantiate a fissa experiment object
    experiment = fissa.Experiment(images, rois, output_dir)

    # Run the FISSA separation algorithm
    experiment.separate()

    # Export to a .mat file which can be opened with MATLAB (optional)
    experiment.save_to_matlab()
