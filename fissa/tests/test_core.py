'''Unit tests for core.py.'''

from __future__ import division

from datetime import datetime
import shutil
import os, os.path
import unittest

import numpy as np

from .base_test import BaseTestCase
from .. import core


class TestExperimentA(BaseTestCase):
    '''Test Experiment class and its methods.'''

    def __init__(self, *args, **kw):
        super(TestExperimentA, self).__init__(*args, **kw)

        self.resources_dir = os.path.join(self.test_directory, 'resources', 'a')
        self.output_dir = os.path.join(
            self.resources_dir,
            'out-' + datetime.now().strftime('%H%M%S%f')
        )
        self.images_dir = 'images'
        self.image_names = ['AVG_A01_R1_small.tif']
        self.roi_zip_path = 'rois.zip'
        self.roi_paths = [os.path.join('rois', r) for r in ['01.roi']]

        self.expected_00 = np.array([
           [11.25423074,  0.        ,  0.        ,  7.55432252, 19.11182766,
             0.        ,  6.37473238,  0.        ,  0.        ,  0.        ,
             0.        ,  1.58567319,  2.28185467,  0.        , 16.70204514,
            17.55112746, 17.23642459,  0.        ,  0.        ,  0.        ,
             0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
             0.        ,  0.        ,  0.        , 14.75392227],
           [89.75326173, 81.33290066, 88.77502093, 80.71108594, 85.5315738 ,
            78.42423771, 80.3659251 , 84.46124736, 78.04229961, 81.48360449,
            82.12879963, 83.11862592, 83.09085808, 91.22418523, 86.42399606,
            81.05860567, 86.15497276, 81.53903092, 80.53875696, 83.41061814,
            80.59332446, 81.64495893, 86.26057223, 82.47622273, 83.28735277,
            84.00697623, 83.68517083, 83.19829805, 82.06518458],
           [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
             0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
             0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
             0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
             0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
             0.        ,  0.        ,  0.        ,  0.        ],
           [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
             0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
             0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
             0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
             0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
             0.        ,  0.        ,  0.        ,  0.        ],
           [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
             0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
             0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
             0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
             0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
             0.        ,  0.        ,  0.        ,  0.        ],
        ])

    def setUp(self):
        if os.path.isdir(self.output_dir):
            self.tearDown()
        os.makedirs(self.output_dir)

    def tearDown(self):
        if os.path.isdir(self.output_dir):
            shutil.rmtree(self.output_dir)

    def test_imagedir_roizip(self):
        image_path = os.path.join(self.resources_dir, self.images_dir)
        roi_path = os.path.join(self.resources_dir, self.roi_zip_path)
        exp = core.Experiment(image_path, roi_path, self.output_dir)
        exp.separate()
        actual = exp.result
        self.assert_equal(len(actual), 1)
        self.assert_equal(len(actual[0]), 1)
        self.assert_allclose(actual[0][0], self.expected_00)

    def test_imagelist_roizip(self):
        image_paths = [
            os.path.join(self.resources_dir, self.images_dir, img)
            for img in self.image_names
        ]
        roi_path = os.path.join(self.resources_dir, self.roi_zip_path)
        exp = core.Experiment(image_paths, roi_path, self.output_dir)
        exp.separate()
        actual = exp.result
        self.assert_equal(len(actual), 1)
        self.assert_equal(len(actual[0]), 1)
        self.assert_allclose(actual[0][0], self.expected_00)

    @unittest.expectedFailure
    def test_imagedir_roilistpath(self):
        image_path = os.path.join(self.resources_dir, self.images_dir)
        roi_paths = [
            os.path.join(self.resources_dir, r)
            for r in self.roi_paths
        ]
        print(roi_paths)
        exp = core.Experiment(image_path, roi_paths, self.output_dir)
        exp.separate()
        actual = exp.result
        self.assert_equal(len(actual), 1)
        self.assert_equal(len(actual[0]), 1)
        self.assert_allclose(actual[0][0], self.expected_00)

    @unittest.expectedFailure
    def test_imagelist_roilistpath(self):
        image_paths = [
            os.path.join(self.resources_dir, self.images_dir, img)
            for img in self.image_names
        ]
        roi_paths = [
            os.path.join(self.resources_dir, r)
            for r in self.roi_paths
        ]
        exp = core.Experiment(image_paths, roi_paths, self.output_dir)
        exp.separate()
        actual = exp.result
        self.assert_equal(len(actual), 1)
        self.assert_equal(len(actual[0]), 1)
        self.assert_allclose(actual[0][0], self.expected_00)

    def test_ncores_preparation_1(self):
        image_path = os.path.join(self.resources_dir, self.images_dir)
        roi_path = os.path.join(self.resources_dir, self.roi_zip_path)
        exp = core.Experiment(image_path, roi_path, self.output_dir,
                              ncores_preparation=1)
        exp.separate()
        actual = exp.result
        self.assert_equal(len(actual), 1)
        self.assert_equal(len(actual[0]), 1)
        self.assert_allclose(actual[0][0], self.expected_00)

    def test_ncores_preparation_2(self):
        image_path = os.path.join(self.resources_dir, self.images_dir)
        roi_path = os.path.join(self.resources_dir, self.roi_zip_path)
        exp = core.Experiment(image_path, roi_path, self.output_dir,
                              ncores_preparation=2)
        exp.separate()
        actual = exp.result
        self.assert_equal(len(actual), 1)
        self.assert_equal(len(actual[0]), 1)
        self.assert_allclose(actual[0][0], self.expected_00)

    def test_ncores_separate_1(self):
        image_path = os.path.join(self.resources_dir, self.images_dir)
        roi_path = os.path.join(self.resources_dir, self.roi_zip_path)
        exp = core.Experiment(image_path, roi_path, self.output_dir,
                              ncores_separation=1)
        exp.separate()
        actual = exp.result
        self.assert_equal(len(actual), 1)
        self.assert_equal(len(actual[0]), 1)
        self.assert_allclose(actual[0][0], self.expected_00)

    def test_ncores_separate_2(self):
        image_path = os.path.join(self.resources_dir, self.images_dir)
        roi_path = os.path.join(self.resources_dir, self.roi_zip_path)
        exp = core.Experiment(image_path, roi_path, self.output_dir,
                              ncores_separation=2)
        exp.separate()
        actual = exp.result
        self.assert_equal(len(actual), 1)
        self.assert_equal(len(actual[0]), 1)
        self.assert_allclose(actual[0][0], self.expected_00)

    def test_lowmemorymode(self):
        image_path = os.path.join(self.resources_dir, self.images_dir)
        roi_path = os.path.join(self.resources_dir, self.roi_zip_path)
        exp = core.Experiment(image_path, roi_path, self.output_dir,
                              lowmemory_mode=True)
        exp.separate()
        actual = exp.result
        self.assert_equal(len(actual), 1)
        self.assert_equal(len(actual[0]), 1)
        self.assert_allclose(actual[0][0], self.expected_00)

    def test_manualhandler(self):
        image_path = os.path.join(self.resources_dir, self.images_dir)
        roi_path = os.path.join(self.resources_dir, self.roi_zip_path)
        from .. import datahandler
        exp = core.Experiment(image_path, roi_path, self.output_dir,
                              datahandler_custom=datahandler)
        exp.separate()
        actual = exp.result
        self.assert_equal(len(actual), 1)
        self.assert_equal(len(actual[0]), 1)
        self.assert_allclose(actual[0][0], self.expected_00)

    def test_prepfirst(self):
        image_path = os.path.join(self.resources_dir, self.images_dir)
        roi_path = os.path.join(self.resources_dir, self.roi_zip_path)
        exp = core.Experiment(image_path, roi_path, self.output_dir)
        exp.separation_prep()
        exp.separate()
        actual = exp.result
        self.assert_equal(len(actual), 1)
        self.assert_equal(len(actual[0]), 1)
        self.assert_allclose(actual[0][0], self.expected_00)

    def test_redo(self):
        image_path = os.path.join(self.resources_dir, self.images_dir)
        roi_path = os.path.join(self.resources_dir, self.roi_zip_path)
        exp = core.Experiment(image_path, roi_path, self.output_dir)
        exp.separate()
        exp.separate(redo_prep=True, redo_sep=True)
        actual = exp.result
        self.assert_equal(len(actual), 1)
        self.assert_equal(len(actual[0]), 1)
        self.assert_allclose(actual[0][0], self.expected_00)

    def test_calcdeltaf(self):
        image_path = os.path.join(self.resources_dir, self.images_dir)
        roi_path = os.path.join(self.resources_dir, self.roi_zip_path)
        exp = core.Experiment(image_path, roi_path, self.output_dir)
        exp.separate()
        exp.calc_deltaf(4)
        actual = exp.deltaf_result
        self.assert_equal(len(actual), 1)
        self.assert_equal(len(actual[0]), 1)
        #TODO: Check contents of exp.deltaf_result

    def test_matlab(self):
        image_path = os.path.join(self.resources_dir, self.images_dir)
        roi_path = os.path.join(self.resources_dir, self.roi_zip_path)
        exp = core.Experiment(image_path, roi_path, self.output_dir)
        exp.separate()
        exp.save_to_matlab()
        #TODO: Check contents of the .mat file

    def test_matlab_deltaf(self):
        image_path = os.path.join(self.resources_dir, self.images_dir)
        roi_path = os.path.join(self.resources_dir, self.roi_zip_path)
        exp = core.Experiment(image_path, roi_path, self.output_dir)
        exp.separate()
        exp.save_to_matlab()
        exp.calc_deltaf(4)
        exp.save_to_matlab()
        #TODO: Check contents of the .mat file
