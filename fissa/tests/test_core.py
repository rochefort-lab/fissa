'''Unit tests for core.py.'''

from __future__ import division

import os, os.path
import shutil
import sys
import types
import unittest

import numpy as np
from scipy.io import loadmat

from .base_test import BaseTestCase
from .. import core
from .. import extraction


class ExperimentTestMixin:
    """Base tests for Experiment class."""

    def __init__(self, *args, **kwargs):
        self.output_dir = self.tempdir

    def setUp(self):
        self.tearDown()

    def tearDown(self):
        if os.path.isdir(self.output_dir):
            shutil.rmtree(self.output_dir)

    def compare_result(self, actual):
        """
        Compare experiment result against self.expected["result"].
        """
        # Check sizes are correct
        expected_shape = len(self.roi_paths), len(self.image_names)
        self.assert_equal(np.shape(actual), expected_shape)
        # Check contents are correct
        self.assert_allclose_ragged(actual, self.expected["result"])

    def compare_deltaf_result(self, actual):
        """
        Compare experiment result against self.expected["deltaf_result"].
        """
        # Check sizes are correct
        expected_shape = len(self.roi_paths), len(self.image_names)
        self.assert_equal(np.shape(actual), expected_shape)
        self.assert_allclose_ragged(actual, self.expected["deltaf_result"])

    def compare_output(self, actual, separated=True, compare_deltaf=None):
        """
        Compare actual output against expected from self.expected.

        Parameters
        ----------
        actual : fissa.Experiment
            Actual experiment.
        separated : bool
            Whether to compare results of :meth:`fissa.Experiment.separate`.
            Default is ``True``.
        compare_deltaf : bool or None
            Whether to compare ``actual.deltaf_raw`` and
            ``actual.deltaf_result`` against their targets.
            If ``None`` (default), this is automatically determined based on
            whether ``actual.deltaf_result`` (if ``separated=True``) or
            ``actual.deltaf_raw`` (otherwise) is not ``None``.
        """
        if compare_deltaf is None:
            if separated:
                compare_deltaf = actual.deltaf_result is not None
            else:
                compare_deltaf = actual.deltaf_raw is not None
        # Check sizes are correct
        expected_shape = len(self.roi_paths), len(self.image_names)
        self.assert_equal(np.shape(actual.raw), expected_shape)
        self.assert_equal(len(actual.means), len(self.image_names))
        # Check contents are correct
        self.assert_equal(actual.means, self.expected["means"])
        self.assert_allclose_ragged(actual.roi_polys, self.expected["roi_polys"])
        if separated:
            # Check sizes are correct
            self.assert_equal(np.shape(actual.sep), expected_shape)
            # Check result is correct
            self.compare_result(actual.result)
            # Check contents are correct
            self.assert_allclose_ragged(actual.sep, self.expected["sep"])
            self.assert_allclose_ragged(actual.mixmat, self.expected["mixmat"])
        if compare_deltaf:
            self.assert_allclose_ragged(actual.deltaf_raw, self.expected["deltaf_raw"])
        if compare_deltaf and separated:
            self.compare_deltaf_result(actual.deltaf_result)

    def compare_experiments(
            self,
            actual,
            expected,
            folder=True,
            prepared=True,
            separated=True,
        ):
        """
        Compare attributes of two experiments.

        Parameters
        ----------
        actual : fissa.Experiment
            Actual experiment.
        expected : fissa.Experiment
            Expected experiment.
        folder : bool
            Whether to compare the folder values. Default is ``True``.
        prepared : bool
            Whether to compare results of :meth:`fissa.Experiment.separation_prep`.
            Default is ``True``.
        separated : bool
            Whether to compare results of :meth:`fissa.Experiment.separate`.
            Default is ``True``.
        """
        # We do all these comparisons explicitly one-by-one instead of in a
        # for loop so you can see which one is failing.

        # Check the parameters are the same
        self.assert_equal(actual.images, expected.images)
        self.assert_equal(actual.rois, expected.rois)
        if folder:
            self.assert_equal(actual.folder, expected.folder)
        self.assert_equal(actual.nRegions, expected.nRegions)
        self.assert_equal(actual.expansion, expected.expansion)
        self.assert_equal(actual.alpha, expected.alpha)
        self.assert_equal(actual.ncores_preparation, expected.ncores_preparation)
        self.assert_equal(actual.ncores_separation, expected.ncores_separation)
        self.assert_equal(actual.method, expected.method)
        # self.assert_equal(actual.datahandler, expected.datahandler)

        if prepared:
            if expected.raw is None:
                self.assertIs(actual.raw, expected.raw)
            else:
                self.assert_allclose_ragged(actual.raw, expected.raw)
            self.assert_allclose_ragged(actual.means, expected.means)
            self.assert_allclose_ragged(actual.roi_polys, expected.roi_polys)
        if separated:
            self.assert_allclose_ragged(actual.result, expected.result)
            self.assert_allclose_ragged(actual.sep, expected.sep)
            self.assert_allclose_ragged(actual.mixmat, expected.mixmat)
            self.assert_equal(actual.info, expected.info)

    def compare_matlab(self, fname, experiment, compare_deltaf=None):
        """
        Compare matfile contents against an experiment.

        Parameters
        ----------
        fname : str
            Path to .mat file to test.
        experiment : fissa.Experiment
            Experiment with expected values to compare against.
        compare_deltaf : bool or None
            Whether to compare ``experiment.deltaf_raw`` against
            ``experiment.raw`` and ``experiment.deltaf_result`` against
            ``experiment.result``.
            If ``None`` (default), this is automatically determined based on
            whether ``experiment.deltaf_result`` is not ``None``.
        """
        if compare_deltaf is None:
            compare_deltaf = experiment.deltaf_result is not None
        self.assertTrue(os.path.isfile(fname))
        # Check contents of the .mat file
        M = loadmat(fname)
        self.assert_allclose(M["raw"][0, 0][0][0, 0][0], experiment.raw[0, 0])
        self.assert_allclose(M["result"][0, 0][0][0, 0][0], experiment.result[0, 0])
        self.assert_allclose(
            M["ROIs"][0, 0][0][0, 0][0][0][0],
            experiment.roi_polys[0, 0][0][0],
        )
        if compare_deltaf:
            self.assert_allclose(
                M["df_result"][0, 0][0][0, 0][0],
                experiment.deltaf_result[0, 0],
            )
            # Row and column vectors on MATLAB are 2d instead of 1d, and df_raw
            # is a vector, not a matrix, so has an extra dimension.
            # N.B. This extra dimension is in the wrong place as it doesn't align
            # with the other attributes.
            self.assert_allclose(
                M["df_raw"][0, 0][0][0, 0][0][0, :],
                experiment.deltaf_raw[0, 0],
            )

    def compare_matlab_expected(self, fname, compare_deltaf=True):
        """
        Compare matfile contents against expected from test attributes.

        Parameters
        ----------
        fname : str
            Path to .mat file to test.
        compare_deltaf : bool or None
            Whether to compare ``deltaf_raw`` and ``deltaf_result``.
            Default is ``True``.
        """
        self.assertTrue(os.path.isfile(fname))
        # Check contents of the .mat file
        M = loadmat(fname)
        self.assert_allclose(M["raw"][0, 0][0][0, 0][0], self.expected["raw"][0, 0])
        self.assert_allclose(
            M["result"][0, 0][0][0, 0][0],
            self.expected["result"][0, 0],
        )
        self.assert_allclose(
            M["ROIs"][0, 0][0][0, 0][0][0][0],
            self.expected["roi_polys"][0, 0][0][0],
        )
        if compare_deltaf:
            self.assert_allclose(
                M["df_result"][0, 0][0][0, 0][0],
                self.expected["deltaf_result"][0, 0],
            )
            # Row and column vectors on MATLAB are 2d instead of 1d, and df_raw
            # is a vector, not a matrix, so has an extra dimension.
            # N.B. This extra dimension is in the wrong place as it doesn't align
            # with the other attributes.
            self.assert_allclose(
                M["df_raw"][0, 0][0][0, 0][0][0, :],
                self.expected["deltaf_raw"][0, 0],
            )

    def compare_str_repr_contents(self, actual, params=None):
        print("ACTUAL: {}".format(actual))
        self.assert_starts_with(actual, "fissa.core.Experiment(")
        self.assertTrue(actual[-1] == ")")
        self.assertTrue("images=" in actual)
        self.assertTrue("rois=" in actual)
        if not params:
            return
        for param, value in params.items():
            expected = "{}={},".format(param, repr(value))
            print("Testing presence of ~ {} ~".format(expected))
            self.assertTrue(expected in actual)

    def test_repr_class(self):
        exp = core.Experiment(self.images_dir, self.roi_zip_path)
        self.compare_str_repr_contents(repr(exp))

    def test_str_class(self):
        exp = core.Experiment(self.images_dir, self.roi_zip_path)
        self.compare_str_repr_contents(str(exp))

    def test_str_contains_stuff(self):
        params = {
            "nRegions": 7,
            "expansion": 0.813962,
            "alpha": 0.212827,
            "ncores_preparation": 1,
            "ncores_separation": None,
            "method": "nmf",
        }
        exp = core.Experiment(self.images_dir, self.roi_zip_path, **params)
        self.compare_str_repr_contents(str(exp), params)

    def test_repr_contains_stuff(self):
        params = {
            "nRegions": 7,
            "expansion": 0.813962,
            "alpha": 0.212827,
            "ncores_preparation": 1,
            "ncores_separation": None,
            "method": "nmf",
        }
        exp = core.Experiment(self.images_dir, self.roi_zip_path, **params)
        self.compare_str_repr_contents(repr(exp), params)

    def test_repr_eval(self):
        params = {
            "nRegions": 7,
            "expansion": 0.813962,
            "alpha": 0.212827,
            "ncores_preparation": 1,
            "ncores_separation": None,
            "method": "nmf",
        }
        exp = core.Experiment(self.images_dir, self.roi_zip_path, **params)
        actual = repr(exp)
        # We've done relative imports to test the current version of the code,
        # but the repr string expects to be able to find packages at fissa.
        # To solve this, we make a dummy package named fissa and put aliases
        # to core and extraction on it, so eval can find them.
        fissa = types.ModuleType("DummyFissa")
        fissa.core = core
        fissa.extraction = extraction
        print("Evaluating: {}".format(actual))
        exp2 = eval(actual)
        self.compare_experiments(exp, exp2)


    def test_imagedir_roizip(self):
        exp = core.Experiment(self.images_dir, self.roi_zip_path)
        exp.separate()
        self.compare_output(exp)
        self.compare_str_repr_contents(str(exp))
        self.compare_str_repr_contents(repr(exp))

    def test_imagelist_roizip(self):
        image_paths = [
            os.path.join(self.images_dir, img)
            for img in self.image_names
        ]
        exp = core.Experiment(image_paths, self.roi_zip_path)
        exp.separate()
        self.compare_output(exp)
        self.compare_str_repr_contents(str(exp))
        self.compare_str_repr_contents(repr(exp))

    def test_imagelistloaded_roizip(self):
        image_paths = [
            os.path.join(self.images_dir, img)
            for img in self.image_names
        ]
        datahandler = extraction.DataHandlerTifffile()
        images = [datahandler.image2array(pth) for pth in image_paths]
        exp = core.Experiment(images, self.roi_zip_path)
        exp.separate()
        self.compare_output(exp)
        self.compare_str_repr_contents(str(exp))
        self.compare_str_repr_contents(repr(exp))

    @unittest.expectedFailure
    def test_imagedir_roilistpath(self):
        roi_paths = [
            os.path.join(self.resources_dir, r)
            for r in self.roi_paths
        ]
        exp = core.Experiment(self.images_dir, roi_paths)
        exp.separate()
        self.compare_output(exp)
        self.compare_str_repr_contents(str(exp))
        self.compare_str_repr_contents(repr(exp))

    @unittest.expectedFailure
    def test_imagelist_roilistpath(self):
        image_paths = [
            os.path.join(self.images_dir, img)
            for img in self.image_names
        ]
        roi_paths = [
            os.path.join(self.resources_dir, r)
            for r in self.roi_paths
        ]
        exp = core.Experiment(image_paths, roi_paths)
        exp.separate()
        self.compare_output(exp)
        self.compare_str_repr_contents(str(exp))
        self.compare_str_repr_contents(repr(exp))

    def test_nocache(self):
        exp = core.Experiment(self.images_dir, self.roi_zip_path)
        exp.separate()
        self.compare_output(exp)

    def test_ncores_preparation_1(self):
        exp = core.Experiment(
            self.images_dir,
            self.roi_zip_path,
            ncores_preparation=1,
        )
        exp.separate()
        self.compare_output(exp)

    def test_ncores_preparation_2(self):
        exp = core.Experiment(
            self.images_dir,
            self.roi_zip_path,
            ncores_preparation=2,
        )
        exp.separate()
        self.compare_output(exp)

    def test_ncores_separate_1(self):
        exp = core.Experiment(
            self.images_dir,
            self.roi_zip_path,
            ncores_separation=1,
        )
        exp.separate()
        self.compare_output(exp)

    def test_ncores_separate_2(self):
        exp = core.Experiment(
            self.images_dir,
            self.roi_zip_path,
            ncores_separation=2,
        )
        exp.separate()
        self.compare_output(exp)

    def test_lowmemorymode(self):
        exp = core.Experiment(
            self.images_dir,
            self.roi_zip_path,
            lowmemory_mode=True,
        )
        exp.separate()
        self.compare_output(exp)

    def test_lowmemorymode_datahandler(self):
        with self.assertRaises(ValueError):
            exp = core.Experiment(
                self.images_dir,
                self.roi_zip_path,
                lowmemory_mode=True,
                datahandler=extraction.DataHandlerTifffile(),
            )

    def test_manualhandler_Tifffile(self):
        exp = core.Experiment(
            self.images_dir,
            self.roi_zip_path,
            self.output_dir,
            datahandler=extraction.DataHandlerTifffile(),
        )
        exp.separate()
        self.compare_output(exp)

    def test_manualhandler_TifffileLazy(self):
        exp = core.Experiment(
            self.images_dir,
            self.roi_zip_path,
            self.output_dir,
            datahandler=extraction.DataHandlerTifffileLazy(),
        )
        exp.separate()
        self.compare_output(exp)

    def test_manualhandler_Pillow(self):
        exp = core.Experiment(
            self.images_dir,
            self.roi_zip_path,
            self.output_dir,
            datahandler=extraction.DataHandlerPillow(),
        )
        exp.separate()
        self.compare_output(exp)

    def test_caching(self):
        exp = core.Experiment(self.images_dir, self.roi_zip_path, self.output_dir)
        exp.separate()

    def test_prefolder(self):
        os.makedirs(self.output_dir)
        exp = core.Experiment(self.images_dir, self.roi_zip_path, self.output_dir)
        exp.separate()

    def test_cache_pwd_explict(self):
        """Check we can use pwd as the cache folder"""
        prevdir = os.getcwd()
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        try:
            os.chdir(self.output_dir)
            exp = core.Experiment(self.images_dir, self.roi_zip_path, ".")
            exp.separate()
        finally:
            os.chdir(prevdir)

    def test_cache_pwd_implicit(self):
        """Check we can use pwd as the cache folder"""
        prevdir = os.getcwd()
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        try:
            os.chdir(self.output_dir)
            exp = core.Experiment(self.images_dir, self.roi_zip_path, "")
            exp.separate()
        finally:
            os.chdir(prevdir)

    def test_subfolder(self):
        """Check we can write to a subfolder"""
        output_dir = os.path.join(self.output_dir, "a", "b", "c")
        exp = core.Experiment(self.images_dir, self.roi_zip_path, output_dir)
        exp.separate()

    def test_folder_deleted_before_call(self):
        """Check we can write to a folder that is deleted in the middle"""
        exp = core.Experiment(self.images_dir, self.roi_zip_path, self.output_dir)
        # Delete the folder between instantiating Experiment and separate()
        self.tearDown()
        exp.separate()

    def test_folder_deleted_between_prep_sep(self):
        """Check we can write to a folder that is deleted in the middle"""
        exp = core.Experiment(self.images_dir, self.roi_zip_path, self.output_dir)
        # Delete the folder between separation_prep() and separate()
        exp.separation_prep()
        self.tearDown()
        exp.separate()

    def test_prepfirst(self):
        exp = core.Experiment(self.images_dir, self.roi_zip_path, self.output_dir)
        exp.separation_prep()
        exp.separate()
        self.compare_output(exp)

    def test_redo(self):
        """Test whether experiment redoes work when requested."""
        exp = core.Experiment(self.images_dir, self.roi_zip_path, self.output_dir)
        capture_pre = self.capsys.readouterr()  # Clear stdout
        exp.separate()
        capture_post = self.recapsys(capture_pre)
        self.assert_starts_with(capture_post.out, "Doing")
        capture_pre = self.capsys.readouterr()  # Clear stdout
        exp.separate(redo_prep=True, redo_sep=True)
        capture_post = self.recapsys(capture_pre)
        self.assert_starts_with(capture_post.out, "Doing")
        self.compare_output(exp)

    def test_load_cache(self):
        """Test whether cached output is loaded during init."""
        image_path = self.images_dir
        roi_path = self.roi_zip_path
        # Run an experiment to generate the cache
        exp1 = core.Experiment(image_path, roi_path, self.output_dir)
        exp1.separate()
        # Make a new experiment we will test
        exp = core.Experiment(image_path, roi_path, self.output_dir)
        # Cache should be loaded without calling separate
        self.compare_output(exp)

    def test_load_cache_piecemeal(self):
        """
        Test whether cached output is loaded during individual method calls.
        """
        image_path = self.images_dir
        roi_path = self.roi_zip_path
        # Run an experiment to generate the cache
        exp1 = core.Experiment(image_path, roi_path, self.output_dir)
        exp1.separate()
        # Make a new experiment we will test; this should load the cache
        capture_pre = self.capsys.readouterr()  # Clear stdout
        exp = core.Experiment(image_path, roi_path, self.output_dir)
        capture_post = self.recapsys(capture_pre)  # Capture and then re-output
        self.assert_starts_with(capture_post.out, "Reloading data")
        # Ensure previous cache is loaded again when we run separation_prep
        capture_pre = self.capsys.readouterr()  # Clear stdout
        exp.separation_prep()
        capture_post = self.recapsys(capture_pre)
        self.assert_starts_with(capture_post.out, "Reloading data")
        # Ensure previous cache is loaded again when we run separate
        capture_pre = self.capsys.readouterr()  # Clear stdout
        exp.separate()
        capture_post = self.recapsys(capture_pre)
        self.assert_starts_with(capture_post.out, "Reloading data")
        # Check the contents loaded from cache
        self.compare_output(exp)

    def test_load_cached_prep(self):
        """
        With prep cached, test prep loads and separate waits for us to call it.
        """
        image_path = self.images_dir
        roi_path = self.roi_zip_path
        # Run an experiment to generate the cache
        exp1 = core.Experiment(image_path, roi_path, self.output_dir)
        exp1.separation_prep()
        # Make a new experiment we will test; this should load the cache
        capture_pre = self.capsys.readouterr()  # Clear stdout
        exp = core.Experiment(image_path, roi_path, self.output_dir)
        capture_post = self.recapsys(capture_pre)  # Capture and then re-output
        self.assert_starts_with(capture_post.out, "Reloading data")
        # Ensure previous cache is loaded again when we run separation_prep
        capture_pre = self.capsys.readouterr()  # Clear stdout
        exp.separation_prep()
        capture_post = self.recapsys(capture_pre)  # Capture and then re-output
        self.assert_starts_with(capture_post.out, "Reloading data")
        # Since we did not run and cache separate, this needs to run now
        capture_pre = self.capsys.readouterr()  # Clear stdout
        exp.separate()
        capture_post = self.recapsys(capture_pre)
        self.assert_starts_with(capture_post.out, "Doing signal separation")
        # Check the contents loaded from cache
        self.compare_output(exp)

    def test_load_manual_prep(self):
        """Loading prep results from a different folder."""
        image_path = self.images_dir
        roi_path = self.roi_zip_path
        prev_folder = os.path.join(self.output_dir, "a")
        # Run an experiment to generate the cache
        exp1 = core.Experiment(image_path, roi_path, prev_folder)
        exp1.separation_prep()
        # Make a new experiment we will test
        new_folder = os.path.join(self.output_dir, "b")
        exp = core.Experiment(image_path, roi_path, new_folder)
        exp.load(os.path.join(prev_folder, "preparation.npz"))
        # Cached prep should now be loaded correctly
        self.compare_experiments(exp, exp1, folder=False)

    def test_load_manual_sep(self):
        """Loading prep results from a different folder."""
        image_path = self.images_dir
        roi_path = self.roi_zip_path
        prev_folder = os.path.join(self.output_dir, "a")
        # Run an experiment to generate the cache
        exp1 = core.Experiment(image_path, roi_path, prev_folder)
        exp1.separate()
        # Make a new experiment we will test
        new_folder = os.path.join(self.output_dir, "b")
        exp = core.Experiment(image_path, roi_path, new_folder)
        exp.load(os.path.join(prev_folder, "separated.npz"))
        # Cached results should now be loaded correctly
        self.compare_experiments(exp, exp1, folder=False, prepared=False)

    def test_load_manual_directory(self):
        """Loading results from a different folder."""
        image_path = self.images_dir
        roi_path = self.roi_zip_path
        prev_folder = os.path.join(self.output_dir, "a")
        # Run an experiment to generate the cache
        exp1 = core.Experiment(image_path, roi_path, prev_folder)
        exp1.separate()
        # Make a new experiment we will test
        new_folder = os.path.join(self.output_dir, "b")
        exp = core.Experiment(image_path, roi_path, new_folder)
        exp.load(prev_folder)
        # Cache should now be loaded correctly
        self.compare_experiments(exp, exp1, folder=False)

    def test_load_manual(self):
        """Loading results from a different folder."""
        image_path = self.images_dir
        roi_path = self.roi_zip_path
        prev_folder = os.path.join(self.output_dir, "a")
        # Run an experiment to generate the cache
        exp1 = core.Experiment(image_path, roi_path, prev_folder)
        exp1.separate()
        # Make a new experiment we will test
        new_folder = os.path.join(self.output_dir, "b")
        exp = core.Experiment(image_path, roi_path, new_folder)
        # Copy the contents from the old cache to the new cache
        shutil.rmtree(new_folder)
        shutil.copytree(prev_folder, new_folder)
        # Manually trigger loading the new cache
        exp.load()
        # Cache should now be loaded correctly
        self.compare_experiments(exp, exp1, folder=False)

    def test_load_empty_prep(self):
        """Behaviour when loading a prep cache that is empty."""
        exp = core.Experiment(self.images_dir, self.roi_zip_path, self.output_dir)
        # Make an empty prep save file
        np.savez_compressed(os.path.join(self.output_dir, "preparation.npz"))
        exp.separation_prep()
        self.compare_output(exp, separated=False)

    def test_load_empty_sep(self):
        """Behaviour when loading a separated cache that is empty."""
        exp = core.Experiment(self.images_dir, self.roi_zip_path, self.output_dir)
        # Make an empty separated save file
        np.savez_compressed(os.path.join(self.output_dir, "separated.npz"))
        exp.separate()
        self.compare_output(exp)

    def test_load_none(self):
        """Behaviour when loading a cache containing None."""
        fields = ["raw", "result", "deltaf_result"]
        exp = core.Experiment(self.images_dir, self.roi_zip_path, self.output_dir)
        # Set the fields to be something other than `None`
        for field in fields:
            setattr(exp, field, 42)
        # Make a save file which contains values set to `None`
        fname = os.path.join(self.output_dir, "dummy.npz")
        np.savez_compressed(fname, **{field: None for field in fields})
        # Load the file and check the data appears as None, not np.array(None)
        exp.load(fname)
        for field in fields:
            self.assertIs(getattr(exp, field), None)

    @unittest.expectedFailure
    def test_badprepcache_init1(self):
        """
        With a faulty prep cache, test prep hits an error during init and then stops.
        """
        image_path = self.images_dir
        roi_path = self.roi_zip_path
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        # Make a bad cache
        with open(os.path.join(self.output_dir, "preparation.npz"), "w") as f:
            f.write("badfilecontents")

        capture_pre = self.capsys.readouterr()  # Clear stdout
        exp = core.Experiment(image_path, roi_path, self.output_dir)
        capture_post = self.recapsys(capture_pre)  # Capture and then re-output
        self.assertTrue("An error occurred" in capture_post.out)

        self.assertIs(exp.raw, None)

    @unittest.expectedFailure
    def test_badprepcache_init2(self):
        """
        With a faulty prep cache, test prep initially errors but then runs when called.
        """
        image_path = self.images_dir
        roi_path = self.roi_zip_path
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        # Make a bad cache
        with open(os.path.join(self.output_dir, "preparation.npz"), "w") as f:
            f.write("badfilecontents")

        capture_pre = self.capsys.readouterr()  # Clear stdout
        exp = core.Experiment(image_path, roi_path, self.output_dir)
        capture_post = self.recapsys(capture_pre)  # Capture and then re-output
        self.assertTrue("An error occurred" in capture_post.out)

        capture_pre = self.capsys.readouterr()  # Clear stdout
        exp.separation_prep()
        capture_post = self.recapsys(capture_pre)  # Capture and then re-output
        self.assert_starts_with(capture_post.out, "Doing region growing")

    def test_badprepcache(self):
        """
        With a faulty prep cache, test prep catches error and runs when called.
        """
        image_path = self.images_dir
        roi_path = self.roi_zip_path
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        exp = core.Experiment(image_path, roi_path, self.output_dir)
        # Make a bad cache
        with open(os.path.join(self.output_dir, "preparation.npz"), "w") as f:
            f.write("badfilecontents")

        capture_pre = self.capsys.readouterr()  # Clear stdout
        exp.separation_prep()
        capture_post = self.recapsys(capture_pre)  # Capture and then re-output
        self.assertTrue("An error occurred" in capture_post.out)

        capture_pre = self.capsys.readouterr()  # Clear stdout
        exp.separate()
        capture_post = self.recapsys(capture_pre)  # Capture and then re-output
        self.assert_starts_with(capture_post.out, "Doing signal separation")

        self.compare_output(exp)

    def test_badsepcache(self):
        """
        With a faulty separated cache, test separate catches error and runs when called.
        """
        image_path = self.images_dir
        roi_path = self.roi_zip_path
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        exp = core.Experiment(image_path, roi_path, self.output_dir)
        exp.separation_prep()
        # Make a bad cache
        with open(os.path.join(self.output_dir, "separated.npz"), "w") as f:
            f.write("badfilecontents")

        capture_pre = self.capsys.readouterr()  # Clear stdout
        exp.separate()
        capture_post = self.recapsys(capture_pre)  # Capture and then re-output
        self.assertTrue("An error occurred" in capture_post.out)

        self.compare_output(exp)

    def test_manual_save_prep(self):
        """Saving prep results with manually specified filename."""
        destination = os.path.join(self.output_dir, "m", ".test_output.npz")
        os.makedirs(os.path.dirname(destination))
        exp = core.Experiment(self.images_dir, self.roi_zip_path)
        exp.separation_prep()
        exp.save_prep(destination=destination)
        self.assertTrue(os.path.isfile(destination))

    def test_manual_save_sep(self):
        """Saving sep results with manually specified filename."""
        destination = os.path.join(self.output_dir, "m", ".test_output.npz")
        os.makedirs(os.path.dirname(destination))
        exp = core.Experiment(self.images_dir, self.roi_zip_path)
        exp.separate()
        exp.save_separated(destination=destination)
        self.assertTrue(os.path.isfile(destination))

    def test_manual_save_sep_undefined(self):
        """Saving prep results without specifying a filename."""
        exp = core.Experiment(self.images_dir, self.roi_zip_path)
        exp.separation_prep()
        with self.assertRaises(ValueError):
            exp.save_prep()

    def test_manual_save_prep_undefined(self):
        """Saving sep results without specifying a filename."""
        exp = core.Experiment(self.images_dir, self.roi_zip_path)
        exp.separate()
        with self.assertRaises(ValueError):
            exp.save_separated()

    def test_load_manual_undefined(self):
        """Loading results without specifying a filename or cache folder."""
        exp = core.Experiment(self.images_dir, self.roi_zip_path)
        with self.assertRaises(ValueError):
            exp.load()

    def test_calcdeltaf(self):
        exp = core.Experiment(self.images_dir, self.roi_zip_path)
        exp.separate()
        exp.calc_deltaf(self.fs)
        self.compare_output(exp, compare_deltaf=True)

    def test_calcdeltaf_cache(self):
        exp = core.Experiment(self.images_dir, self.roi_zip_path, self.output_dir)
        exp.separate()
        exp.calc_deltaf(self.fs)
        self.compare_output(exp, compare_deltaf=True)

    def test_calcdeltaf_notrawf0(self):
        exp = core.Experiment(self.images_dir, self.roi_zip_path)
        exp.separate()
        exp.calc_deltaf(self.fs, use_raw_f0=False)
        # We did not use this setting to generate the expected values, so can't
        # compare the output against the target.
        self.compare_output(exp, compare_deltaf=False)
        # Check sizes are correct
        expected_shape = len(self.roi_paths), len(self.image_names)
        self.assert_equal(np.shape(exp.deltaf_result), expected_shape)

    def test_calcdeltaf_notacrosstrials(self):
        exp = core.Experiment(self.images_dir, self.roi_zip_path)
        exp.separate()
        exp.calc_deltaf(self.fs, across_trials=False)
        # We did not use this setting to generate the expected values, so can't
        # compare the output against the target.
        self.compare_output(exp, compare_deltaf=False)
        # Check sizes are correct
        expected_shape = len(self.roi_paths), len(self.image_names)
        self.assert_equal(np.shape(exp.deltaf_result), expected_shape)

    def test_calcdeltaf_notrawf0_notacrosstrials(self):
        exp = core.Experiment(self.images_dir, self.roi_zip_path)
        exp.separate()
        exp.calc_deltaf(self.fs, use_raw_f0=False, across_trials=False)
        # We did not use this setting to generate the expected values, so can't
        # compare the output against the target.
        self.compare_output(exp, compare_deltaf=False)
        # Check sizes are correct
        expected_shape = len(self.roi_paths), len(self.image_names)
        self.assert_equal(np.shape(exp.deltaf_result), expected_shape)

    def test_matlab(self):
        exp = core.Experiment(self.images_dir, self.roi_zip_path, self.output_dir)
        exp.separate()
        exp.save_to_matlab()
        fname = os.path.join(self.output_dir, "matlab.mat")
        # Check contents of the .mat file
        self.compare_matlab(fname, exp)

    def test_matlab_custom_fname(self):
        exp = core.Experiment(self.images_dir, self.roi_zip_path, self.output_dir)
        exp.separate()
        fname = os.path.join(self.output_dir, "test_output.mat")
        exp.save_to_matlab(fname)
        # Check contents of the .mat file
        self.compare_matlab(fname, exp)

    def test_matlab_no_cache_no_fname(self):
        exp = core.Experiment(self.images_dir, self.roi_zip_path)
        exp.separate()
        self.assertRaises(ValueError, exp.save_to_matlab)

    def test_matlab_from_cache(self):
        """Save to matfile after loading from cache."""
        # Run an experiment to generate the cache
        exp1 = core.Experiment(self.images_dir, self.roi_zip_path, self.output_dir)
        exp1.separate()
        # Make a new experiment we will test
        exp = core.Experiment(self.images_dir, self.roi_zip_path, self.output_dir)
        # Cache should be loaded without calling separate
        exp.save_to_matlab()
        fname = os.path.join(self.output_dir, "matlab.mat")
        self.compare_matlab(fname, exp)

    def test_matlab_deltaf(self):
        exp = core.Experiment(self.images_dir, self.roi_zip_path, self.output_dir)
        exp.separate()
        exp.calc_deltaf(self.fs)
        exp.save_to_matlab()
        fname = os.path.join(self.output_dir, "matlab.mat")
        # Check contents of the .mat file
        self.compare_matlab(fname, exp, compare_deltaf=True)

    def test_func(self):
        image_path = os.path.join(self.resources_dir, self.images_dir)
        roi_path = os.path.join(self.resources_dir, self.roi_zip_path)
        actual = core.run_fissa(
            image_path, roi_path, self.output_dir, export_to_matlab=None
        )
        self.compare_result(actual)
        # Check contents of the .mat file
        expected_file = os.path.join(self.output_dir, "matlab.mat")
        self.compare_matlab_expected(expected_file, compare_deltaf=False)

    def test_func_explict_matlab(self):
        image_path = os.path.join(self.resources_dir, self.images_dir)
        roi_path = os.path.join(self.resources_dir, self.roi_zip_path)
        actual = core.run_fissa(
            image_path, roi_path, self.output_dir, export_to_matlab=True
        )
        self.compare_result(actual)
        expected_file = os.path.join(self.output_dir, "matlab.mat")
        self.compare_matlab_expected(expected_file, compare_deltaf=False)

    def test_func_explict_nomatlab(self):
        image_path = os.path.join(self.resources_dir, self.images_dir)
        roi_path = os.path.join(self.resources_dir, self.roi_zip_path)
        actual = core.run_fissa(
            image_path, roi_path, self.output_dir, export_to_matlab=False
        )
        self.compare_result(actual)
        expected_file = os.path.join(self.output_dir, "matlab.mat")
        self.assertFalse(os.path.isfile(expected_file))

    def test_func_manual_matlab(self):
        image_path = os.path.join(self.resources_dir, self.images_dir)
        roi_path = os.path.join(self.resources_dir, self.roi_zip_path)
        fname = os.path.join(self.output_dir, "test_output.mat")
        actual = core.run_fissa(
            image_path, roi_path, self.output_dir, export_to_matlab=fname
        )
        self.compare_result(actual)
        self.compare_matlab_expected(fname, compare_deltaf=False)

    def test_func_nocache(self):
        image_path = os.path.join(self.resources_dir, self.images_dir)
        roi_path = os.path.join(self.resources_dir, self.roi_zip_path)
        actual = core.run_fissa(image_path, roi_path)
        self.compare_result(actual)

    def test_func_deltaf(self):
        image_path = os.path.join(self.resources_dir, self.images_dir)
        roi_path = os.path.join(self.resources_dir, self.roi_zip_path)
        actual = core.run_fissa(
            image_path, roi_path, self.output_dir, freq=self.fs, return_deltaf=True
        )
        self.compare_deltaf_result(actual)

    def test_func_deltaf_nofreq(self):
        image_path = os.path.join(self.resources_dir, self.images_dir)
        roi_path = os.path.join(self.resources_dir, self.roi_zip_path)
        with self.assertRaises(ValueError):
            core.run_fissa(image_path, roi_path, self.output_dir, return_deltaf=True)


class TestExperimentB(BaseTestCase, ExperimentTestMixin):
    """Test core on Experiment B, which has 2 ROIs and 3 TIFFs."""

    def __init__(self, *args, **kwargs):
        super(TestExperimentB, self).__init__(*args, **kwargs)
        ExperimentTestMixin.__init__(self)

        self.resources_dir = os.path.join(self.test_directory, "resources", "b")
        self.images_dir = os.path.join(self.resources_dir, "images")
        self.image_names = ["AVG_A01.tif", "AVG_A02.tif", "AVG_A03.tif"]
        self.image_shape = (29, 21)
        self.fs = 1
        self.roi_zip_path = os.path.join(self.resources_dir, "rois.zip")
        self.roi_paths = [os.path.join("rois", "{:02d}.roi") for r in range(3, 5)]

        self.expected = np.load(
            os.path.join(
                self.resources_dir,
                "expected_py{}.npz".format(sys.version_info.major),
            ),
            allow_pickle=True,
        )


class TestSeparateTrials(BaseTestCase):
    """Tests for the separate_trials helper function."""

    def __init__(self, *args, **kwargs):
        super(TestSeparateTrials, self).__init__(*args, **kwargs)

        # Load cached data
        self.resources_dir = os.path.join(self.test_directory, "resources", "b")
        cache = np.load(
            os.path.join(
                self.resources_dir,
                "expected_py{}.npz".format(sys.version_info.major),
            ),
            allow_pickle=True,
        )
        # Test against saved data for the first ROI
        self.raw = cache["raw"][0]
        self.expected_sep = list(cache["sep"][0])
        self.expected_match = list(cache["result"][0])
        self.expected_mixmat = cache["mixmat"][0][0]
        self.expected_convergence = cache["info"][0][0]
        # We can't require the number of iterations to be the same across
        # all versions of sklearn.
        self.expected_convergence.pop("iterations")

    def compare_outputs(self, outputs):
        Xsep, Xmatch, Xmixmat, convergence = outputs
        self.assert_allclose_ragged(Xsep, self.expected_sep)
        self.assert_allclose_ragged(Xmatch, self.expected_match)
        self.assert_allclose(Xmixmat, self.expected_mixmat)
        convergence.pop("iterations")
        self.assert_equal(convergence, self.expected_convergence)

    def test_separate_trials(self):
        outputs = core.separate_trials(self.raw)
        self.compare_outputs(outputs)

    def test_separate_trials_label_int(self):
        label = 239457
        capture_pre = self.capsys.readouterr()  # Clear stdout
        outputs = core.separate_trials(self.raw, roi_label=label)
        capture_post = self.recapsys(capture_pre)  # Capture and then re-output
        self.assertTrue(str(label) in capture_post.out)
        self.compare_outputs(outputs)

    def test_separate_trials_label_str(self):
        label = "awesome_roi"
        capture_pre = self.capsys.readouterr()  # Clear stdout
        outputs = core.separate_trials(self.raw, roi_label=label)
        capture_post = self.recapsys(capture_pre)  # Capture and then re-output
        self.assertTrue(label in capture_post.out)
        self.compare_outputs(outputs)

    @unittest.skipIf(sys.version_info < (3, 2), "assertWarnsRegex only on Python>=3.3")
    def test_separate_trials_negative(self):
        with self.assertWarnsRegex(UserWarning, ".*values below zero.*"):
            outputs = core.separate_trials(self.raw - 1e6)

    @unittest.skipIf(sys.version_info < (3, 2), "assertWarnsRegex only on Python>=3.3")
    def test_separate_trials_negative_labelled(self):
        label = "awesome_roi"
        with self.assertWarnsRegex(UserWarning, ".*values below zero.*" + label + ".*"):
            outputs = core.separate_trials(self.raw - 1e6, roi_label=label)
