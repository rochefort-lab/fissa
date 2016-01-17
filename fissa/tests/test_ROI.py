"""
The tests below were taken from the sima package
http://www.losonczylab.org/sima
version 1.3.0.

License
-------
This file is Copyright (C) 2014  The Trustees of Columbia University in the
City of New York.

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
from builtins import object
# Unit tests for sima/ROI.py
# Tests follow conventions for NumPy/SciPy available at
# https://github.com/numpy/numpy/blob/master/doc/TESTS.rst.txt

# use assert_() and related functions over the built in assert to ensure tests
# run properly, regardless of how python is started.
from numpy.testing import (
    assert_,
    assert_equal,
    assert_almost_equal,
    assert_array_almost_equal,
    assert_raises,
    assert_array_equal,
    dec,
    TestCase,
    run_module_suite,
    assert_allclose)

from ..sima_borrowed import ROI
import numpy as np


def setup():
    return


def teardown():
    return


def test_poly2mask():
    poly1 = [[0, 0], [0, 1], [1, 1], [1, 0]]
    poly2 = [[0, 1], [0, 2], [2, 2], [2, 1]]
    masks = ROI.poly2mask([poly1, poly2], (3, 3))
    assert_equal(
        masks[0].todense(),
        np.matrix([[True, False, False],
                   [True, True, False],
                   [False, False, False]], dtype=bool))


if __name__ == "__main__":
    run_module_suite()
