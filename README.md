FISSA
=====

FISSA (Fast Image Signal Source Analysis) is a Python library for extracting
somatic signals from 2-photon calcium imaging data.
It requires images in tiff format as well as predefined ROIs around somas. 

It offers the use of ICA and several NMF algorithms to do so, as well as 
ROI manipulation routines for generating neuropil ROIs. 


Installation
------------

FISSA is currently only available for Python 2.7, and has only been tested on
Ubuntu 15.04.

You can download the package source from GitHub, and then install FISSA and its 
dependencies as follows

```unix
    git clone https://github.com/rochefort-lab/fissa.git
    pip install -r fissa/require_first.txt
    pip install -r fissa/requirements.txt
    pip install -e fissa
```

To generate the plots in the iPython Notebooks, you will also need to install
the optional dependencies:

```unix
    pip install -r fissa/optional.txt
```

If you wish, you can install FISSA and its dependencies into a virtual
environment.

### Notes on dependencies of dependencies

* `scipy` requires a fortran compiler and LAPACK, which on Ubuntu can be
  installed with `sudo apt-get install gfortran libopenblas-dev`

* `shapely` requires GEOS, which on Ubuntu can be installed
  `sudo apt-get install libgeos-dev`

* If you want to use version 3.0.0 of Pillow, you will need to install a JPEG
  library with `sudo apt-get install libjpeg-dev`. Alternatively, you can
  install a version 2.9.0 of Pillow.


Folder Structure
----------------

### doc
Contains example code. You can load the notebooks as .ipynb directly in Github, 
or on your system if you know how to use ipython notebooks 
(http://ipython.org/ipython-doc/stable/notebook/index.html). 
You can also read the .html pages instead. 

### exampleData
Contains example data. It has two zips with region of interests from ImageJ. 
It also contains three tiff stacks, which have been downsampled and cropped 
from full data from the Rochefort lab. 

### fissa
Contains the toolbox. See the tutorial in doc for how to use it, and the
comments in the different modules inside FISSA.


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

This program is currently closed source; you can not redistribute it unless
under the express permission of one of the copyright holders.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
