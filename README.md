FISSA
=====

FISSA (Fast Image Signal Source Analysis) is a Python library for extracting
somatic signals from 2-photon data, given predefined ROIs around somas. 

It offers the use of ICA and several NMF algorithms to do so, as well as 
ROI manipulation routines for generating neuropil ROIs. 


Installation
------------

FISSA is currently only available for Python 2.7, and has only been tested on
Linux.

Having downloaded the package source from GitHub, you can install FISSA and its 
dependencies as follows:

```unix
    pip install -r require_first.txt
    pip install -r requirements.txt
    pip install -e fissa
```

To generate the plots in the iPython Notebooks, you will also need to install
the optional dependencies:

```unix
    pip install -r optional.txt
```

If you wish, you can install FISSA and its dependecies into a virtual
environment.


License
-------

Unless otherwise stated in individual files, all code is
Copyright (c) 2015, Sander Keemink and Scott Lowe.
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
