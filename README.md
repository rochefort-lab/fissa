[![Join the FISSA chat](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/rochefort-lab/fissa)
[![Shippable](https://img.shields.io/shippable/56391d7a1895ca4474227917.svg)](https://app.shippable.com/projects/56391d7a1895ca4474227917)

FISSA
=====

FISSA (Fast Image Signal Separation Analysis) is a Python library for decontaminating
somatic signals from 2-photon calcium imaging data.
It can read images in tiff format and ROIs in zips as exported by ImageJ, or Numpy arrays
in general after importing from different formats.

Currently, FISSA is only available for Python 2.7, and has been tested on
Ubuntu 17.04 and on Windows 7 with the
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

These notebooks can also be downloaded from this repository and run on your own machine, by downloading the examples folder.

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
conda install -c conda-forge shapely 
```

Install FISSA as follows (from the folder above fissa).

```
pip install fissa
```
To test if FISSA installed correctly type
```
python
```
and then
```
import fissa
```
If no errors show up, FISSA installed correctly.

If you want use the interactive plotting from the notebooks also install
the HoloViews plotting toolbox

```
conda install -c holoviews
```


### Installation on Linux

Before installing FISSA, you will need to make sure you have all of its dependencies
(and the dependencies of its dependencies) installed.

Here we will outline how to do all of these steps, assuming you already have both
Python 2.7 and pip installed. It is highly likely that your Linux distribution ships with these.

#### Dependencies of dependencies
* [scipy](https://pypi.python.org/pypi/scipy/) requires a
  [Fortran compiler and BLAS/LAPACK/ATLAS](http://www.scipy.org/scipylib/building/linux.html#installation-from-source).

* [shapely](https://pypi.python.org/pypi/Shapely) requires GEOS.

These packages can be installed on *Debian/Ubuntu* with the following shell
commands.

```bash
sudo apt-get update
sudo apt-get install gfortran libopenblas-dev liblapack-dev libatlas-dev libatlas-base-dev
sudo apt-get install libgeos-dev
```

#### Installing FISSA
Install basic FISSA as:

```
pip install fissa
```

With the libraries required to run the notebooks with plots:

```
pip install fissa['plotting']
```

To test if FISSA installed correctly type
```
python
```
and then
```
import fissa
```
If no errors show up, FISSA installed correctly.

Folder Structure
----------------

### examples
Contains example code. You can load the notebooks as .ipynb directly in GitHub,
or on your system if you know how to use jupyter notebooks
(for a tutorial see https://www.datacamp.com/community/tutorials/tutorial-jupyter-notebook).

For a basic tutorial of using FISSA see ```Basic usage.ipynb``` or ```Basic usage.html```. An example workflow is shown in ```basic_usage.py```.

### exampleData
Contains example data. It a zipfile with region of interests from ImageJ.
It also contains three tiff stacks, which have been downsampled and cropped
from full data from the Rochefort lab.

### fissa
Contains the toolbox.


Citing FISSA
------------

If you use FISSA for your research, please cite the following paper
in any resulting publications:

_Paper in preparation_


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
