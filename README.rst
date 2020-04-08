|Binder| |Gitter| |PyPI badge| |Travis| |AppVeyor| |Documentation| |Codecov| |Coveralls| |Downloads|

FISSA
=====

FISSA (Fast Image Signal Separation Analysis) is a Python library for
decontaminating somatic signals from two-photon calcium imaging data. It
can read images in tiff format and ROIs in zips as exported by ImageJ;
or operate with numpy arrays directly, which can be produced by
importing files stored in other formats.

For details of the algorithm, please see our `companion
paper <https://www.doi.org/10.1038/s41598-018-21640-2>`__ published in
Scientific Reports. For the code used to generate the simulated data
in the companion paper, see the
`SimCalc repository <https://github.com/rochefort-lab/SimCalc/>`__.

FISSA is compatible with both Python 2.7 and Python 3.5+. Using Python 3
is strongly encouraged, as Python 2 will no longer be `maintained
starting January 2020 <https://python3statement.org/>`__.

FISSA has been tested on Ubuntu 17.04 and on Windows Windows 10 with the
`Anaconda <https://www.anaconda.com/download/#linux>`__ distribution.

Documentation, including the full API, is available online at
`<https://fissa.readthedocs.io>`_.

If you encounter a specific problem please `open a new
issue <https://github.com/rochefort-lab/fissa/issues/new>`__. For
general discussion and help with installation or setup, please see the
`Gitter chat <https://gitter.im/rochefort-lab/fissa>`__.

Usage
-----

A general tutorial on the use of FISSA can be found here:
`[HTML] <https://rochefort-lab.github.io/fissa/examples/Basic%20usage.html>`__
`[Binder] <https://mybinder.org/v2/gh/rochefort-lab/fissa/master?filepath=examples/Basic%20usage.ipynb>`__
`[Source] <https://github.com/rochefort-lab/fissa/blob/master/examples/Basic%20usage.ipynb>`__
`[Raw] <https://raw.githubusercontent.com/rochefort-lab/fissa/master/examples/Basic%20usage.ipynb>`__.

An example workflow with another Python toolbox (SIMA):
`[HTML] <https://rochefort-lab.github.io/fissa/examples/SIMA%20example.html>`__
`[Binder] <https://mybinder.org/v2/gh/rochefort-lab/fissa/master?filepath=examples/SIMA%20example.ipynb>`__
`[Source] <https://github.com/rochefort-lab/fissa/blob/master/examples/SIMA%20example.ipynb>`__
`[Raw] <https://raw.githubusercontent.com/rochefort-lab/fissa/master/examples/SIMA%20example.ipynb>`__.

An example workflow importing data exported from a MATLAB toolbox (cNMF):
`[HTML] <https://rochefort-lab.github.io/fissa/examples/cNMF%20example.html>`__
`[Binder] <https://mybinder.org/v2/gh/rochefort-lab/fissa/master?filepath=examples/cNMF%20example.ipynb>`__
`[Source] <https://github.com/rochefort-lab/fissa/blob/master/examples/cNMF%20example.ipynb>`__
`[Raw] <https://raw.githubusercontent.com/rochefort-lab/fissa/master/examples/cNMF%20example.ipynb>`__.

You can try out each of the example notebooks interactively on `Binder <https://mybinder.org/v2/gh/rochefort-lab/fissa/master?filepath=examples>`__.

These notebooks can also be run on your own machine.
To do so, you will need to:

1.  Install fissa with its plotting dependencies :code:`pip install fissa[plotting]`.
2.  If you want to run the sima notebook, you will also have to install sima with :code:`pip install sima`.
    Note that sima only supports python<=3.6.
3.  Download `a copy of the repository <https://github.com/rochefort-lab/fissa/archive/master.zip>`__,
    unzip it and browse to the `examples <examples>`__ directory.
4.  Start up a jupyter notebook server to run our notebooks :code:`jupyter notebook`.

If you're new to jupyter notebooks, an approachable tutorial can be found at
`<https://www.datacamp.com/community/tutorials/tutorial-jupyter-notebook>`_.

Installation
------------

Installation on Windows
~~~~~~~~~~~~~~~~~~~~~~~

Basic prerequisites
^^^^^^^^^^^^^^^^^^^

Download and install, in the following order:

-  (for Python 2.7 only) Microsoft Visual C++ Compiler for Python 2.7:
   https://www.microsoft.com/en-us/download/details.aspx?id=44266

-  Python 2.7 or 3.5+ (recommended) Anaconda as the Python environment,
   available from https://www.anaconda.com/download/.

Installing FISSA
^^^^^^^^^^^^^^^^

Open ``Anaconda Prompt.exe``, which can be found through the Windows
start menu or search, and type or copy-paste (by right clicking) the
following:

::

    conda install -c conda-forge shapely tifffile

Then, install FISSA by running the command

::

    pip install fissa

You can check to see if FISSA is installed by running the command

::

    python -c "import fissa; print(fissa.__version__)"

You will see your FISSA version number printed in the terminal.

If you want to use the interactive plotting from the notebooks, you
should also install the HoloViews plotting toolbox, as follows

::

    conda install -c ioam holoviews

See `usage <#usage>`__ above for details on how to use FISSA.

Installation on Linux
~~~~~~~~~~~~~~~~~~~~~

Before installing FISSA, you will need to make sure you have all of its
dependencies (and the dependencies of its dependencies) installed.

Here we will outline how to do all of these steps, assuming you already
have both Python and pip installed. It is highly likely that your Linux
distribution ships with these.

Dependencies of dependencies
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

-  `scipy <https://pypi.python.org/pypi/scipy/>`__ requires a `Fortran
   compiler and
   BLAS/LAPACK/ATLAS <http://www.scipy.org/scipylib/building/linux.html#installation-from-source>`__.

-  `shapely <https://pypi.python.org/pypi/Shapely>`__ requires GEOS.

-  `Pillow <https://pypi.org/project/Pillow/>`__>=3.0.0 effectively
   requires a JPEG library.

These packages can be installed on *Debian/Ubuntu* with the following
shell commands.

.. code:: bash

    sudo apt-get update
    sudo apt-get install gfortran libopenblas-dev liblapack-dev libatlas-dev libatlas-base-dev
    sudo apt-get install libgeos-dev
    sudo apt-get install libjpeg-dev

.. installing-fissa-1:

Installing FISSA
^^^^^^^^^^^^^^^^

For normal usage of FISSA, you can install the latest release version on
PyPI using pip:

::

    pip install fissa

To also install fissa along with the dependencies required to run our
sample notebooks (which include plots rendered with holoviews) you
should run the following command:

::

    pip install fissa['plotting']

You can check to see if FISSA is installed by running the command

::

    python -c "import fissa; print(fissa.__version__)"

You will see your FISSA version number printed in the terminal.


Folder Structure
----------------

A clone of this repository will contain directories detailed below.

docs/
~~~~~

Contains the source for the documentation, which is available online at
`<https://fissa.readthedocs.io>`_.
You can build a local copy of the documentation by running the command

::

    make -C docs html

examples/
~~~~~~~~~

Contains example code. You can load the notebooks as .ipynb directly in
GitHub, or on your system if you know how to use jupyter notebooks.
The example notebooks can also be run interactively on `Binder <https://mybinder.org/v2/gh/rochefort-lab/fissa/master?filepath=examples>`__.

examples/exampleData/
~~~~~~~~~~~~~~~~~~~~~

Contains example data. It a zipfile with region of interests from
ImageJ. It also contains three tiff stacks, which have been downsampled
and cropped from full data from the Rochefort lab.

.. fissa-1:

fissa/
~~~~~~

Contains the toolbox.

fissa/tests/
~~~~~~~~~~~~

Contains tests for the toolbox, which are run to ensure it will work as
expected.

.ci/
~~~~

Contains files necessary for deploying tests on continuous integration
servers. Users should ignore this directory.

Citing FISSA
------------

If you use FISSA for your research, please cite the following paper in
any resulting publications:

S. W. Keemink, S. C. Lowe, J. M. P. Pakan, E. Dylda, M. C. W. van
Rossum, and N. L. Rochefort. FISSA: A neuropil decontamination toolbox
for calcium imaging signals, *Scientific Reports*, **8**\ (1):3493,
2018.
`doi: 10.1038/s41598-018-21640-2 <https://www.doi.org/10.1038/s41598-018-21640-2>`__.

For your convenience, the FISSA package ships with a copy of this
citation in bibtex format, available at
`citation.bib <https://raw.githubusercontent.com/rochefort-lab/fissa/master/citation.bib>`__.

License
-------

Unless otherwise stated in individual files, all code is Copyright (c)
2015, Sander Keemink, Scott Lowe, and Nathalie Rochefort. All rights
reserved.

This program is free software; you can redistribute it and/or modify it
under the terms of the GNU General Public License as published by the
Free Software Foundation; either version 3 of the License, or (at your
option) any later version.

This program is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General
Public License for more details.

You should have received a copy of the GNU General Public License along
with this program. If not, see http://www.gnu.org/licenses/.

.. |Gitter| image:: https://badges.gitter.im/Join%20Chat.svg
   :target: https://gitter.im/rochefort-lab/fissa
   :alt: Join the FISSA chat
.. |PyPI badge| image:: https://img.shields.io/pypi/v/fissa.svg
   :target: https://pypi.org/project/fissa
   :alt: Latest PyPI release
.. |Travis| image:: https://travis-ci.org/rochefort-lab/fissa.svg?branch=master
   :target: https://travis-ci.org/rochefort-lab/fissa
   :alt: Travis Build Status
.. |AppVeyor| image:: https://ci.appveyor.com/api/projects/status/n694frm31qcv29j0/branch/master?svg=true
   :target: https://ci.appveyor.com/project/scottclowe/rochefort-lab-fissa/branch/master
   :alt: AppVeyor Build Status
.. |Documentation| image:: https://readthedocs.org/projects/fissa/badge/?version=latest
   :target: https://fissa.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status
.. |Codecov| image:: https://codecov.io/gh/rochefort-lab/fissa/branch/master/graph/badge.svg
   :target: https://codecov.io/gh/rochefort-lab/fissa
   :alt: Codecov Coverage
.. |Coveralls| image:: https://coveralls.io/repos/github/rochefort-lab/fissa/badge.svg?branch=master
   :target: https://coveralls.io/github/rochefort-lab/fissa?branch=master
   :alt: Coveralls Coverage
.. |Downloads| image:: https://pepy.tech/badge/fissa
   :target: https://pepy.tech/project/fissa
   :alt: Download Counter
.. |Binder| image:: https://mybinder.org/badge_logo.svg
   :target: https://mybinder.org/v2/gh/rochefort-lab/fissa/master?filepath=examples
   :alt: Launch Notebooks in Binder
