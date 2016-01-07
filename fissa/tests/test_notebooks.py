"""
Unit tests for iPython Notebooks
"""
from __future__ import division

import sys
import os
import subprocess
import unittest
import numpy as np

from .base_test import BaseTestCase


def individual_notebook_check(filename):
    # Need to call ipython2/python2 or ipython3/python3 on command line, so
    # we need to know which version of python we are currently running.
    ver = str(sys.version_info.major)
    # Make sure the file does exist
    assert os.path.isfile(filename)
    # Convert the .ipynb file into a .py script
    subprocess.check_call(['ipython'+ver, 'nbconvert', '--to=script', filename])
    # The .py file is made into the current directory
    destination, scriptname = os.path.split(filename)
    scriptname = os.path.splitext(scriptname)[0]+'.py'
    # Move the script into the correct folder
    os.rename(scriptname, os.path.join(destination, scriptname))
    # Check our current location
    old_dir = os.getcwd()
    try:
        # Move ourselves to this folder. The iPython script expects to be run
        # from there.
        os.chdir(destination)
        # Try to run the notebook file. Must be done in iPython - not Python -
        # so the magic can flow freely.
        subprocess.check_call(['ipython'+ver, scriptname])
    finally:
        # Move back to the original location
        os.chdir(old_dir)

class TestNotebooks(BaseTestCase):

    def test_example_workflow(self):
        individual_notebook_check('doc/Example Workflow.ipynb')

    def test_tutorial(self):
        individual_notebook_check('doc/FISSA Tutorial.ipynb')
