[![Join the FISSA chat](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/rochefort-lab/fissa)
[![Travis Build Status](https://travis-ci.org/rochefort-lab/fissa.svg?branch=master)](https://travis-ci.org/rochefort-lab/fissa)


FISSA
=====

FISSA (Fast Image Signal Separation Analysis) is a Python library for decontaminating somatic signals from two-photon calcium imaging data.
It can read images in tiff format and ROIs in zips as exported by ImageJ; or operate with numpy arrays directly, which can be produced by importing files stored in other formats.

For details of the algorithm, please see our [companion paper](https://www.doi.org/10.1038/s41598-018-21640-2) published in Scientific Reports.

FISSA is compatible with both Python 2.7 and Python 3.5+.
It has been tested on Ubuntu 17.04 and on Windows 7 with the
[Anaconda](https://www.anaconda.com/download/#linux) distribution.

If you encounter a specific problem please
[open a new issue](https://github.com/rochefort-lab/fissa/issues/new).
For general discussion and help with installation or setup, please see the
[Gitter chat](https://gitter.im/rochefort-lab/fissa).


Usage
-----

A general tutorial on the use of FISSA can be found at:
<https://rochefort-lab.github.io/fissa/examples/Basic%20usage.html>

An example workflow with another Python toolbox (SIMA):
<https://rochefort-lab.github.io/fissa/examples/SIMA%20example.html>

An example workflow importing data exported from a MATLAB toolbox (cNMF):
<https://rochefort-lab.github.io/fissa/examples/cNMF%20example.html>

These notebooks can also be run on your own machine.
To do so, you will need to [download a copy of the repository](https://github.com/rochefort-lab/fissa/archive/master.zip), unzip it and browse to the [examples](examples) directory.
Then, start up a jupyter notebook server to run our notebooks.
If you're new to jupyter notebooks, an approachable tutorial can be found at <https://www.datacamp.com/community/tutorials/tutorial-jupyter-notebook>.


Installation
------------

### Installation on Windows
#### Basic prerequisites
Download and install, in the following order:
* Microsoft Visual C++ Compiler for Python 2.7: <https://www.microsoft.com/en-us/download/details.aspx?id=44266>

* Python 2.7 Anaconda as the Python environment, available from
<https://www.anaconda.com/download/>.


#### Installing FISSA
Open `Anaconda Prompt.exe`, which can be found through the Windows
start menu or search, and type or copy-paste (by right clicking) the following:

```
conda install -c conda-forge shapely tifffile
```

Then, install FISSA by running the command

```
pip install fissa
```

To test if FISSA has been installed, enter the command
```
python
```
to go into the Python environment. Then type
```
import fissa
```
If no errors show up, FISSA is now installed.
You can leave Python by typing `exit()`.

If you want to use the interactive plotting from the notebooks, you should
also install the HoloViews plotting toolbox, as follows

```
conda install -c ioam holoviews
```

See [usage](#usage) above for details on how to use FISSA.


### Installation on Linux

Before installing FISSA, you will need to make sure you have all of its dependencies
(and the dependencies of its dependencies) installed.

Here we will outline how to do all of these steps, assuming you already have both
Python 2.7 and pip installed. It is highly likely that your Linux distribution ships with these.

#### Dependencies of dependencies
* [scipy](https://pypi.python.org/pypi/scipy/) requires a
  [Fortran compiler and BLAS/LAPACK/ATLAS](http://www.scipy.org/scipylib/building/linux.html#installation-from-source).

* [shapely](https://pypi.python.org/pypi/Shapely) requires GEOS.

* [Pillow](https://pypi.org/project/Pillow/)>=3.0.0 effectively requires a JPEG library.

These packages can be installed on *Debian/Ubuntu* with the following shell
commands.

```bash
sudo apt-get update
sudo apt-get install gfortran libopenblas-dev liblapack-dev libatlas-dev libatlas-base-dev
sudo apt-get install libgeos-dev
sudo apt-get install libjpeg-dev
```

#### Installing FISSA

For normal usage of FISSA, you can install the latest release version on PyPI
using pip:
```
pip install fissa
```

To also install fissa along with the dependencies required to run our sample
notebooks (which include plots rendered with holoviews) you should run the
following command:
```
pip install fissa['plotting']
```

Afterwards, you can test to see if FISSA is install by running the command
```
python
```
to start an interactive python session. Then run
```
import fissa
```
at the python command prompt.

If no errors show up, FISSA is now installed.
You can leave the interactive python session with the `exit()` command, or CTRL+D.


Folder Structure
----------------

### continuous_integration/
Contains files necessary for deploying tests on continuous integration servers.
Users should ignore this directory.

### examples/
Contains example code. You can load the notebooks as .ipynb directly in GitHub,
or on your system if you know how to use jupyter notebooks.

### examples/exampleData/
Contains example data. It a zipfile with region of interests from ImageJ.
It also contains three tiff stacks, which have been downsampled and cropped
from full data from the Rochefort lab.

### fissa/
Contains the toolbox.

### fissa/tests/
Contains tests for the toolbox, which are run to ensure it will work as expected.


Citing FISSA
------------

If you use FISSA for your research, please cite the following paper
in any resulting publications:

S. W. Keemink, S. C. Lowe, J. M. P. Pakan, E. Dylda, M. C. W. van Rossum, and N. L. Rochefort. FISSA: A neuropil decontamination toolbox for calcium imaging signals, *Scientific Reports*, **8**(1):3493, 2018. [DOI:10.1038/s41598-018-21640-2](https://www.doi.org/10.1038/s41598-018-21640-2).

For your convenience, the FISSA package ships with a copy of this citation in bibtex format, available at [citation.bib](citation.bib).


License
-------

Unless otherwise stated in individual files, all code is
Copyright (c) 2015, Sander Keemink, Scott Lowe, and Nathalie Rochefort.
All rights reserved.

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
