#!/usr/bin/env python

"""
Basic FISSA usage example.

This file contains a step-by-step example workflow for using the FISSA toolbox
with a functional interface.

An example notebook is provided here:
https://github.com/rochefort-lab/fissa/blob/master/examples/Basic%20usage%20-%20Functional.ipynb
"""

import fissa

# Define the data to extract
rois = "exampleData/20150429.zip"
images = "exampleData/20150529"

# Define the name of the experiment extraction location
# Make sure you use a different output path for each experiment you run.
output_dir = "fissa_example"

# Run FISSA on this data
result = fissa.run_fissa(images, rois, output_dir, export_to_matlab=True)
