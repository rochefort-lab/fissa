"""
Provides a base test class for other test classes to inherit from.

Includes the numpy testing functions as methods.
"""

import contextlib
import datetime
import os.path
import random
import shutil
import string
import sys
import tempfile
import unittest
from inspect import getsourcefile

try:
    from collections import abc
except ImportError:
    import collections as abc

import numpy as np
import pytest
from numpy.testing import (
    assert_allclose,
    assert_almost_equal,
    assert_array_equal,
    assert_equal,
)

# Check where the test directory is located, to be used when fetching
# test resource files
TEST_DIRECTORY = os.path.dirname(os.path.abspath(getsourcefile(lambda: 0)))


def assert_allclose_ragged(actual, desired):
    assert_equal(
        np.array(actual, dtype=object).shape,
        np.array(desired, dtype=object).shape,
    )
    for desired_i, actual_i in zip(desired, actual):
        if (
            getattr(desired, "dtype", None) == object
            or np.asarray(desired).dtype == object
        ):
            assert_allclose_ragged(actual_i, desired_i)
        else:
            assert_allclose(actual_i, desired_i)


def assert_equal_list_of_array_perm_inv(actual, desired):
    assert_equal(len(actual), len(desired))
    for desired_i in desired:
        n_matches = 0
        for actual_j in actual:
            if np.equal(actual_j, desired_i).all():
                n_matches += 1
        assert n_matches >= 0


def assert_equal_dict_of_array(actual, desired):
    assert_equal(actual.keys(), desired.keys())
    for k in desired.keys():
        assert_equal(actual[k], desired[k])


def assert_starts_with(actual, desired):
    """
    Check that a string starts with a certain substring.

    Parameters
    ----------
    desired : str
        Desired initial string.
    actual : str-like
        Actual string or string-like object.
    """
    try:
        assert len(actual) >= len(desired)
    except BaseException:
        print(
            "Actual string too short ({} < {} characters)".format(
                len(actual), len(desired)
            )
        )
        print("ACTUAL: {}".format(actual))
        raise
    try:
        return assert_equal(str(actual)[: len(desired)], desired)
    except BaseException as err:
        msg = "ACTUAL: {}".format(actual)
        if isinstance(getattr(err, "args", None), str):
            err.args += "\n" + msg
        elif isinstance(getattr(err, "args", None), tuple):
            if len(err.args) == 1:
                err.args = (err.args[0] + "\n" + msg,)
            else:
                err.args += (msg,)
        else:
            print(msg)
        raise


class BaseTestCase(unittest.TestCase):
    """
    Superclass for all the FISSA test cases.
    """

    # Have the test directory as an attribute to the class as well as
    # a top-level variable
    test_directory = TEST_DIRECTORY

    def __init__(self, *args, **kwargs):
        """Instance initialisation."""
        # First do the __init__ associated with parent class
        # super(self).__init__(*args, **kw)  # Only works on Python3
        super(BaseTestCase, self).__init__(*args, **kwargs)  # Works on Python2
        # Add a test to automatically use when comparing objects of
        # type numpy ndarray. This will be used for self.assertEqual().
        self.addTypeEqualityFunc(np.ndarray, self.assert_allclose)
        self.tempdir = os.path.join(
            tempfile.gettempdir(),
            "out-" + self.generate_temp_name(),
        )

    def generate_temp_name(self, n_character=12):
        """
        Generate a random string to use as a temporary output path.
        """
        population = string.ascii_uppercase + string.digits
        if hasattr(random, "choices"):
            # Python 3.6+
            rstr = "".join(random.choices(population, k=n_character))
        else:
            # Python 2.7/3.5
            rstr = "".join(random.choice(population) for _ in range(n_character))
        return "{}-{}".format(datetime.datetime.now().strftime("%M%S%f"), rstr)

    def tearDown(self):
        # If it was created, delete the randomly generated temporary directory
        if os.path.isdir(self.tempdir):
            shutil.rmtree(self.tempdir)

    @contextlib.contextmanager
    def subTest(self, *args, **kwargs):
        # For backwards compatability with Python < 3.4
        if hasattr(super(BaseTestCase, self), "subTest"):
            yield super(BaseTestCase, self).subTest(*args, **kwargs)
        else:
            yield None

    @pytest.fixture(autouse=True)
    def capsys(self, capsys):
        self.capsys = capsys

    def recapsys(self, *captures):
        """
        Capture stdout and stderr, then write them back to stdout and stderr.

        Capture is done using the `pytest.capsys` fixture.

        Parameters
        ----------
        *captures : pytest.CaptureResult, optional
            A series of extra captures to output. For each `capture` in
            `captures`, `capture.out` and `capture.err` are written to stdout
            and stderr.

        Returns
        -------
        capture : pytest.CaptureResult
            `capture.out` and `capture.err` contain all the outputs to stdout
            and stderr since the previous capture with `~pytest.capsys`.
        """
        capture_now = self.capsys.readouterr()
        for capture in captures:
            sys.stdout.write(capture.out)
            sys.stderr.write(capture.err)
        sys.stdout.write(capture_now.out)
        sys.stderr.write(capture_now.err)
        return capture_now

    def assert_almost_equal(self, actual, desired, *args, **kwargs):
        return assert_almost_equal(actual, desired, *args, **kwargs)

    def assert_array_equal(self, actual, desired, *args, **kwargs):
        return assert_array_equal(actual, desired, *args, **kwargs)

    def assert_allclose(self, actual, desired, *args, **kwargs):
        # Handle msg argument, which is passed from assertEqual, established
        # with addTypeEqualityFunc in __init__
        kwargs.pop("msg", None)
        return assert_allclose(actual, desired, *args, **kwargs)

    def assert_equal(self, actual, desired, *args, **kwargs):
        return assert_equal(actual, desired, *args, **kwargs)

    def assert_allclose_ragged(self, actual, desired):
        if desired is None:
            return self.assertIs(actual, desired)
        return assert_allclose_ragged(actual, desired)

    def assert_equal_list_of_array_perm_inv(self, actual, desired):
        return assert_equal_list_of_array_perm_inv(actual, desired)

    def assert_equal_dict_of_array(self, actual, desired):
        return assert_equal_dict_of_array(actual, desired)

    def assert_starts_with(self, actual, desired):
        return assert_starts_with(actual, desired)
