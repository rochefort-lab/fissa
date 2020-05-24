Changelog
=========

All notable changes to FISSA will be documented here.

The format is based on `Keep a Changelog`_, and this project adheres to
`Semantic Versioning`_.

.. _Keep a Changelog: https://keepachangelog.com/en/1.0.0/
.. _Semantic Versioning: https://semver.org/spec/v2.0.0.html

Categories for changes are: Added, Changed, Deprecated, Removed, Fixed,
Security.


Unreleased
----------

`Full commit changelog <https://github.com/rochefort-lab/fissa/compare/0.7.2...master>`__.


Version `0.7.2 <https://github.com/rochefort-lab/fissa/tree/0.7.2>`__
---------------------------------------------------------------------

Release date: 2020-05-24.
`Full commit changelog <https://github.com/rochefort-lab/fissa/compare/0.7.1...0.7.2>`__.

.. _v0.7.2 Fixed:

Fixed
~~~~~

-   Loading ovals and ellipses which are partially offscreen (to the top or left of the image).
    (`#140 <https://github.com/rochefort-lab/fissa/pull/140>`__)

.. _v0.7.2 Changed:

Changed
~~~~~~~

-   Attempting to load any type of ROI which is fully offscreen to the top or left of the image now produces an error.
    (`#140 <https://github.com/rochefort-lab/fissa/pull/140>`__)


Version `0.7.1 <https://github.com/rochefort-lab/fissa/tree/0.7.1>`__
---------------------------------------------------------------------

Release date: 2020-05-22.
`Full commit changelog <https://github.com/rochefort-lab/fissa/compare/0.7.0...0.7.1>`__.

.. _v0.7.1 Fixed:

Fixed
~~~~~

-   Loading oval, ellipse, brush/freehand, freeline, and polyline ImageJ ROIs on Python 3.
    (`#135 <https://github.com/rochefort-lab/fissa/pull/135>`__)

.. _v0.7.1 Added:

Added
~~~~~

-   Support for rotated rectangle and multipoint ROIs on Python 3.
    (`#135 <https://github.com/rochefort-lab/fissa/pull/135>`__)


Version `0.7.0 <https://github.com/rochefort-lab/fissa/tree/0.7.0>`__
---------------------------------------------------------------------

Release date: 2020-05-04.
`Full commit changelog <https://github.com/rochefort-lab/fissa/compare/0.6.4...0.7.0>`__.

.. _v0.7.0 Security:

Security
~~~~~~~~

-   **Caution:** This release knowingly exposes a new security vulnerability.
    In numpy 1.16, the default behaviour of
    `numpy.load <https://numpy.org/doc/stable/reference/generated/numpy.load.html>`__
    changed to stop loading files saved with pickle compression by default,
    due to potential security problems. However, the default behaviour of
    `numpy.save <https://numpy.org/doc/stable/reference/generated/numpy.save.html>`__
    is still to save with pickling enabled. In order to preserve our
    user-experience and backward compatibility with existing fissa cache files,
    we have changed our behaviour to allow numpy to load from pickled files.
    (`#111 <https://github.com/rochefort-lab/fissa/pull/111>`__)

.. _v0.7.0 Changed:

Changed
~~~~~~~

-   Officially drop support for Python 3.3 and 3.4.
    Add `python_requires` to package metadata, specifying Python 2.7 or >=3.5 is required.
    (`#114 <https://github.com/rochefort-lab/fissa/pull/114>`__)
-   Allow tuples and other sequences to be image and roi inputs to FISSA, not just lists.
    (`#73 <https://github.com/rochefort-lab/fissa/pull/73>`__)
-   Multiprocessing is no longer used when the number of cores is specified as 1.
    (`#74 <https://github.com/rochefort-lab/fissa/pull/74>`__)
-   Changed default ``axis`` argument to internal function ``fissa.roitools.shift_2d_array`` from ``None`` to ``0``.
    (`#54 <https://github.com/rochefort-lab/fissa/pull/54>`__)
-   Documentation updates.
    (`#112 <https://github.com/rochefort-lab/fissa/pull/112>`__,
    `#115 <https://github.com/rochefort-lab/fissa/pull/115>`__,
    `#119 <https://github.com/rochefort-lab/fissa/pull/119>`__,
    `#120 <https://github.com/rochefort-lab/fissa/pull/120>`__,
    `#121 <https://github.com/rochefort-lab/fissa/pull/121>`__)

.. _v0.7.0 Fixed:

Fixed
~~~~~

-   Allow loading from pickled numpy saved files.
    (`#111 <https://github.com/rochefort-lab/fissa/pull/111>`__)
-   Problems reading ints correctly from ImageJ rois on Windows; fixed for Python 3 but not Python 2.
    This problem does not affect Unix, which was already working correctly on both Python 2 and 3.
    (`#90 <https://github.com/rochefort-lab/fissa/pull/90>`__)
-   Reject unsupported ``axis`` argument to internal function ``fissa.roitools.shift_2d_array``.
    (`#54 <https://github.com/rochefort-lab/fissa/pull/54>`__)
-   Don't round number of npil segments down to 0 in ``fissa.roitools.split_npil`` when using ``adaptive_num=True``.
    (`#54 <https://github.com/rochefort-lab/fissa/pull/54>`__)
-   Handling float ``num_slices`` in ``fissa.roitools.split_npil``, for when ``adaptive_num=True``, which was causing problems on Python 3.
    (`#54 <https://github.com/rochefort-lab/fissa/pull/54>`__)

.. _v0.7.0 Added:

Added
~~~~~

-   Test suite additions.
    (`#54 <https://github.com/rochefort-lab/fissa/pull/54>`__,
    `#99 <https://github.com/rochefort-lab/fissa/pull/99>`__)


Version `0.6.4 <https://github.com/rochefort-lab/fissa/tree/0.6.4>`__
---------------------------------------------------------------------

Release date: 2020-04-07.
`Full commit changelog <https://github.com/rochefort-lab/fissa/compare/0.6.3...0.6.4>`__.

This version fully supports Python 3.8, but unfortunately this information was not noted correctly in the PyPI metadata for the release.

.. _v0.6.4 Fixed:

Fixed
~~~~~

-   Fix multiprocessing pool closure on Python 3.8.
    (`#105 <https://github.com/rochefort-lab/fissa/pull/105>`__)


Version `0.6.3 <https://github.com/rochefort-lab/fissa/tree/0.6.3>`__
---------------------------------------------------------------------

Release date: 2020-04-03.
`Full commit changelog <https://github.com/rochefort-lab/fissa/compare/0.6.2...0.6.3>`__.

.. _v0.6.3 Fixed:

Fixed
~~~~~

-   Specify a maximum version for the panel dependency of holoviews on
    Python <3.6, which allows us to continue supporting Python 3.5, otherwise
    dependencies fail to install.
    (`#101 <https://github.com/rochefort-lab/fissa/pull/101>`__)
-   Save deltaf to MATLAB compatible output.
    (`#70 <https://github.com/rochefort-lab/fissa/pull/70>`__)
-   Wipe downstream data stored in the experiment object if upstream data
    changes, so data that is present is always consistent with each other.
    (`#93 <https://github.com/rochefort-lab/fissa/pull/93>`__)
-   Prevent slashes in paths from doubling up if the input path has a trailing
    slash.
    (`#71 <https://github.com/rochefort-lab/fissa/pull/71>`__)
-   Documentation updates.
    (`#91 <https://github.com/rochefort-lab/fissa/pull/91>`__,
    `#88 <https://github.com/rochefort-lab/fissa/pull/88>`__,
    `#97 <https://github.com/rochefort-lab/fissa/pull/97>`__,
    `#89 <https://github.com/rochefort-lab/fissa/pull/89>`__)


Version `0.6.2 <https://github.com/rochefort-lab/fissa/tree/0.6.2>`__
---------------------------------------------------------------------

Release date: 2020-03-11.
`Full commit changelog <https://github.com/rochefort-lab/fissa/compare/0.6.1...0.6.2>`__.

.. _v0.6.2 Fixed:

Fixed
~~~~~

-   Specify a maximum version for tifffile dependency on Python <3.6, which
    allows us to continue supporting Python 2.7 and 3.5, which otherwise
    fail to import dependencies correctly.
    (`#87 <https://github.com/rochefort-lab/fissa/pull/87>`__)
-   Documentation fixes and updates.
    (`#64 <https://github.com/rochefort-lab/fissa/pull/64>`__,
    `#65 <https://github.com/rochefort-lab/fissa/pull/65>`__,
    `#67 <https://github.com/rochefort-lab/fissa/pull/67>`__,
    `#76 <https://github.com/rochefort-lab/fissa/pull/76>`__,
    `#77 <https://github.com/rochefort-lab/fissa/pull/77>`__,
    `#78 <https://github.com/rochefort-lab/fissa/pull/78>`__,
    `#79 <https://github.com/rochefort-lab/fissa/pull/79>`__,
    `#92 <https://github.com/rochefort-lab/fissa/pull/92>`__)


Version `0.6.1 <https://github.com/rochefort-lab/fissa/tree/0.6.1>`__
---------------------------------------------------------------------

Release date: 2019-03-11.
`Full commit changelog <https://github.com/rochefort-lab/fissa/compare/0.6.0...0.6.1>`__.

.. _v0.6.1 Fixed:

Fixed
~~~~~

-   Allow ``deltaf.findBaselineF0`` to run with fewer than 90 samples, by reducing the pad-length if necessary.
    (`#62 <https://github.com/rochefort-lab/fissa/pull/62>`__)
-   Basic usage notebook wasn't supplying the correct ``datahandler_custom`` argument for the custom datahandler (it was using ``datahandler`` instead, which is incorrect; this was silently ignored previously but will now trigger an error).
    (`#62 <https://github.com/rochefort-lab/fissa/pull/62>`__)
-   Use ``ncores_preparation`` for perparation step, not ``ncores_separation``.
    (`#59 <https://github.com/rochefort-lab/fissa/pull/59>`__)
-   Only use ``ncores_separation`` for separation step, not all cores.
    (`#59 <https://github.com/rochefort-lab/fissa/pull/59>`__)
-   Allow both byte strings and unicode strings to be arguments of functions which require strings.
    Previously, byte strings were required on Python 2.7 and unicode strings on Python 3.
    (`#60 <https://github.com/rochefort-lab/fissa/pull/60>`__)


Version `0.6.0 <https://github.com/rochefort-lab/fissa/tree/0.6.0>`__
---------------------------------------------------------------------

Release date: 2019-02-26.
`Full commit changelog <https://github.com/rochefort-lab/fissa/compare/0.5.3...0.6.0>`__.

.. _v0.6.0 Added:

Added
~~~~~

-  Python 3 compatibility.
   (`#33 <https://github.com/rochefort-lab/fissa/pull/33>`__)
-  Documentation generation, with Sphinx, Sphinx-autodoc, and Napoleon.
   (`#38 <https://github.com/rochefort-lab/fissa/pull/38>`__)


Version `0.5.3 <https://github.com/rochefort-lab/fissa/tree/0.5.3>`__
---------------------------------------------------------------------

Release date: 2019-02-18.
`Full commit changelog <https://github.com/rochefort-lab/fissa/compare/0.5.2...0.5.3>`__.

.. _v0.5.3 Fixed:

Fixed
~~~~~

-  Fix f0 detection with low sampling rates.
   (`#27 <https://github.com/rochefort-lab/fissa/pull/27>`__)


Version `0.5.2 <https://github.com/rochefort-lab/fissa/tree/0.5.2>`__
---------------------------------------------------------------------

Release date: 2018-03-07.
`Full commit changelog <https://github.com/rochefort-lab/fissa/compare/0.5.1...0.5.2>`__.

.. _v0.5.2 Changed:

Changed
~~~~~~~

-  The default alpha value was changed from 0.2 to 0.1.
   (`#20 <https://github.com/rochefort-lab/fissa/pull/20>`__)


Version `0.5.1 <https://github.com/rochefort-lab/fissa/tree/0.5.1>`__
---------------------------------------------------------------------

Release date: 2018-01-10.
`Full commit changelog <https://github.com/rochefort-lab/fissa/compare/0.5.0...0.5.1>`__.

.. _v0.5.1 Added:

Added
~~~~~

-  Possibility to define custom datahandler script for other formats
-  Added low memory mode option to load larger tiffs frame-by-frame
   (`#14 <https://github.com/rochefort-lab/fissa/pull/14>`__)
-  Added option to use ICA instead of NMF (not recommended, but is a lot
   faster).
-  Added the option for users to define a custom data and ROI loading
   script.
   (`#13 <https://github.com/rochefort-lab/fissa/pull/13>`__)

.. _v0.5.1 Fixed:

Fixed
~~~~~

-  Fixed custom datahandler usage.
   (`#14 <https://github.com/rochefort-lab/fissa/pull/14>`__)
-  Documentation fixes.
   (`#12 <https://github.com/rochefort-lab/fissa/pull/12>`__)

Version `0.5.0 <https://github.com/rochefort-lab/fissa/tree/0.5.0>`__
---------------------------------------------------------------------

Release date: 2017-10-05

Initial release
