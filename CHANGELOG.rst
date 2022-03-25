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

`Full commit changelog <https://github.com/rochefort-lab/fissa/compare/1.0.0...master>`__.


Version `1.0.0 <https://github.com/rochefort-lab/fissa/tree/1.0.0>`__
---------------------------------------------------------------------

Release date: 2022-03-25.
`Full commit changelog <https://github.com/rochefort-lab/fissa/compare/0.7.2...1.0.0>`__.
This is a major release which offers a serious update to the interface,
documentation, and backend of FISSA.

Please note that some of the changes to the codebase are
backward-incompatible changes. For the most part, the only breaking
changes which users will need to be concerned with are listed below. We
also recommend looking through our updated `tutorials and
examples <https://github.com/rochefort-lab/fissa#usage>`__.

-  The format of the cache used by the new release is different to
   previous versions. The new version is not capable of restoring
   previous results saved in the old cache format. If it is run with
   ``folder`` set to a directory containing a cache in the old format,
   it will ignore the old cache and run from scratch, saving the new
   cache in the same directory with a different file name and format.
-  The shape of ``experiment.deltaf_raw`` has been changed to be a
   ``numpy.ndarray`` of 2d numpy arrays shaped ``(1, time)`` to match
   the shape of the other outputs.

Although we have noted some other breaking changes, end users are very
unlikely to be affected by these. These other changes will only affect
users that have written their own custom datahandler, or who are
directly calling backend tools such as the ``ROI`` module, or the
functions ``fissa.core.separate_func`` and ``fissa.core.extract_func``,
all of which which were either removed or refactored and renamed in this
release.

The most important addition is ``fissa.run_fissa``, a new high-level
function-based interface to FISSA. This function does the same
operations as ``fissa.Experiment``, but has a more streamlined
interface.

.. _v1.0.0 Breaking:

Breaking changes
~~~~~~~~~~~~~~~~

-  The names of the cache files have changed from ``"preparation.npy"``
   and ``"separated.npy"`` to ``"prepared.npz"`` and
   ``"separated.npz"``, and the structure of the contents was changed.
   FISSA output caches generated with version 0.7.x will not no longer
   be loaded when using version 1.0.0. The new version stores analysis
   parameters and TIFF means along with the raw and decontaminated
   traces.
   (`#177 <https://github.com/rochefort-lab/fissa/pull/177>`__,
   `#223 <https://github.com/rochefort-lab/fissa/pull/223>`__,
   `#245 <https://github.com/rochefort-lab/fissa/pull/245>`__,
   `#246 <https://github.com/rochefort-lab/fissa/pull/246>`__)
-  The shape of ``experiment.deltaf_raw`` was changed from a
   ``numpy.ndarray`` shaped ``(cells, trials)``, each containing a
   ``numpy.ndarray`` shaped ``(time, )``, to ``numpy.ndarray`` shaped
   ``(cells, trials)``, each element of which is shaped ``(1, time)``.
   The new output shape matches that of ``experiment.raw`` and
   ``experiment.deltaf_result``.
   (`#198 <https://github.com/rochefort-lab/fissa/pull/198>`__)
-  The way data handlers were defined was reworked to use classes
   instead. The ``datahandler`` and ``datahandler_framebyframe`` modules
   were dropped, and replaced with an ``extraction`` module containing
   both of these data handlers as classes instead. Custom data handlers
   will need to be rewritten to be a class inheriting from
   ``fissa.extraction.DataHandlerAbstract`` instead of a custom module.
-  The ``ROI`` module was renamed to ``polygons``.
   (`#219 <https://github.com/rochefort-lab/fissa/pull/219>`__)
-  In ``neuropil.separate``, the keyword arguments ``maxiter`` and
   ``maxtries`` were renamed to be ``max_iter`` and ``max_tries``. This
   change was made so that the parameter name ``max_iter`` is the same
   as the parameter name used by ``sklearn.decomposition.NMF``.
   (`#230 <https://github.com/rochefort-lab/fissa/pull/230>`__)
-  The internal functions ``fissa.core.extract_func`` and
   ``fissa.core.separate_func`` were removed. New functions which are
   comparable but actually have user-friendly interfaces,
   ``fissa.core.extract`` and ``fissa.core.separate_trials``, were added
   in their place.

.. _v1.0.0 Changed:

Changed
~~~~~~~

-  The outputs ``experiment.raw``, ``experiment.sep``, and
   ``experiment.deltaf_raw`` were changed from a list of lists of 2d
   numpy arrays, to a 2d numpy array of 2d numpy arrays. Other outputs,
   such as ``experiment.result`` were already a 2d numpy array of 2d
   numpy arrays.
   (`#164 <https://github.com/rochefort-lab/fissa/pull/164>`__)
-  Output arrays (``experiment.result``, etc.) are now initialized as
   empty arrays instead of lists of lists of ``None`` objects.
   (`#212 <https://github.com/rochefort-lab/fissa/pull/212>`__)
-  The multiprocessing back-end now uses joblib instead of the
   multiprocessing module.
   (`#227 <https://github.com/rochefort-lab/fissa/pull/227>`__)
-  Unnecessary warnings about ragged numpy.ndarrays were suppressed.
   (`#243 <https://github.com/rochefort-lab/fissa/pull/243>`__,
   `#247 <https://github.com/rochefort-lab/fissa/pull/247>`__)
-  Output properties are now automatically wiped when parameter
   attributes are changes.
   (`#254 <https://github.com/rochefort-lab/fissa/pull/254>`__)
-  The set of extra requirements named ``"dev"`` which specified the
   requirements needed to run the test suite was renamed to ``"test"``.
   This can be installed with ``pip install fissa[test]``. There is
   still a ``"dev"`` set of extras, but these are now development tools
   and no longer include the requirements needed to run the unit tests.
   (`#185 <https://github.com/rochefort-lab/fissa/pull/185>`__)

.. _v1.0.0 Fixed:

Fixed
~~~~~

-  The preparation and separation steps are no longer needlessly re-run
   (`#171 <https://github.com/rochefort-lab/fissa/pull/171>`__,
   `#172 <https://github.com/rochefort-lab/fissa/pull/172>`__)
-  Mean images are saved with float64 precision regardless of the
   precision of the source TIFFs file.
   (`#176 <https://github.com/rochefort-lab/fissa/pull/176>`__)
-  Various bugs in the Suite2p workflow. (
   `#181 <https://github.com/rochefort-lab/fissa/pull/181>`__,
   `#257 <https://github.com/rochefort-lab/fissa/pull/257>`__)
-  Variables set to ``None`` are no longer saved as ``numpy.ndarray``.
   (`#199 <https://github.com/rochefort-lab/fissa/pull/199>`__)
-  An error is now raised when both lowmemory mode and a manual
   datahandler are provided.
   (`#206 <https://github.com/rochefort-lab/fissa/pull/206>`__)
-  Mismatches between the number of rois/trials/etc. and the array
   shapes (which can occur when the data in ``experiment.raw`` is
   overwritten) are resolved by determining these attributes
   dynamically.
   (`#244 <https://github.com/rochefort-lab/fissa/pull/244>`__)
-  Use ``np.array`` instead of the deprecated ``np.matrix``.
   (`#174 <https://github.com/rochefort-lab/fissa/pull/174>`__)
-  Use ``np.float64`` instead of the deprecated ``np.float``.
   (`#213 <https://github.com/rochefort-lab/fissa/pull/213>`__)
-  Iterate over elements in ``shapely.geometry.MultiPolygon`` by using
   the ``geoms`` attribute instead treating the whole object as an
   iterable (which is now deprecated).
   (`#272 <https://github.com/rochefort-lab/fissa/pull/272>`__)

.. _v1.0.0 Added:

Added
~~~~~

-  Added ``fissa.run_fissa``, a high-level function-based interface to
   FISSA. This does the same operations as ``fissa.Experiment``, but in
   a more streamlined interface.
   (`#169 <https://github.com/rochefort-lab/fissa/pull/169>`__,
   `#237 <https://github.com/rochefort-lab/fissa/pull/237>`__)
-  Added a ``verbosity`` argument to control how much feedback is given
   by FISSA when it is running.
   (`#200 <https://github.com/rochefort-lab/fissa/pull/200>`__,
   `#225 <https://github.com/rochefort-lab/fissa/pull/225>`__,
   `#238 <https://github.com/rochefort-lab/fissa/pull/238>`__,
   `#240 <https://github.com/rochefort-lab/fissa/pull/240>`__)
-  A new ``fissa.Experiment.to_matfile`` method was added. The arrays
   saved in this matfile have a different format to the previous
   ``fissa.Experiment.save_to_matlab`` method, which is now deprecated.
   (`#249 <https://github.com/rochefort-lab/fissa/pull/249>`__)
-  A new data handler ``extract.DataHandlerTifffileLazy`` was added,
   which is able to handle TIFF files of all data types while in
   low-memory mode.
   (`#156 <https://github.com/rochefort-lab/fissa/pull/156>`__,
   `#179 <https://github.com/rochefort-lab/fissa/pull/179>`__,
   `#187 <https://github.com/rochefort-lab/fissa/pull/187>`__).
-  In ``fissa.Experiment``, arguments ``max_iter``, ``tol``, and
   ``max_tries`` were added which pass through to ``neuropil.separate``
   to control the stopping conditions for the signal separation routine.
   (`#224 <https://github.com/rochefort-lab/fissa/pull/224>`__,
   `#230 <https://github.com/rochefort-lab/fissa/pull/230>`__)
-  In ``fissa.Experiment``, add ``__repr__`` and ``__str__`` methods.
   These changes mean that ``str(experiment)`` describes the content of
   a ``fissa.Experiment instance`` in a human readable way, and
   ``repr(experiment)`` in a machine-readable way.
   (`#209 <https://github.com/rochefort-lab/fissa/pull/209>`__,
   `#231 <https://github.com/rochefort-lab/fissa/pull/231>`__)
-  Support for arbitrary ``sklearn.decomposition`` classes
   (e.g. FactorAnalysis), not just ICA and NMF.
   (`#188 <https://github.com/rochefort-lab/fissa/pull/188>`__)

.. _v1.0.0 Deprecated:

Deprecated
~~~~~~~~~~

-  The ``fissa.Experiment.save_to_matlab`` method was deprecated. Please
   use the new ``fissa.Experiment.to_matfile`` method instead. The new
   method has a different output structure by default (which better
   matches the structure in Python). If you need to continue using the
   old structure, you can use
   ``fissa.Experiment.to_matfile(legacy=True)``.
   (`#249 <https://github.com/rochefort-lab/fissa/pull/249>`__)

.. _v1.0.0 Documentation:

Documentation
~~~~~~~~~~~~~

-  Reworked all the tutorial notebooks to have better flow, and use
   matplotlib instead of holoviews which is more approachable for new
   users.
   (`#205 <https://github.com/rochefort-lab/fissa/pull/205>`__,
   `#228 <https://github.com/rochefort-lab/fissa/pull/228>`__,
   `#239 <https://github.com/rochefort-lab/fissa/pull/239>`__,
   `#279 <https://github.com/rochefort-lab/fissa/pull/279>`__)
-  The Suite2p example notebook was moved to a `separate
   repository <https://github.com/rochefort-lab/fissa-suite2p-example>`__.
   This change was made because we want to test our other notebooks with
   the latest versions of their dependencies, but this did not fit well
   with running Suite2p, which needs a precise combination of
   dependencies to run.
-  Integrated the example notebooks into the documentation generated by
   Sphinx and shown on readthedocs.
   (`#273 <https://github.com/rochefort-lab/fissa/pull/273>`__)
-  Other various notebook improvements.
   (`#248 <https://github.com/rochefort-lab/fissa/pull/248>`__)
-  Various documentation improvements.
   (`#153 <https://github.com/rochefort-lab/fissa/pull/153>`__,
   `#162 <https://github.com/rochefort-lab/fissa/pull/162>`__,
   `#166 <https://github.com/rochefort-lab/fissa/pull/166>`__,
   `#167 <https://github.com/rochefort-lab/fissa/pull/167>`__,
   `#175 <https://github.com/rochefort-lab/fissa/pull/175>`__,
   `#182 <https://github.com/rochefort-lab/fissa/pull/182>`__,
   `#183 <https://github.com/rochefort-lab/fissa/pull/183>`__,
   `#184 <https://github.com/rochefort-lab/fissa/pull/184>`__,
   `#193 <https://github.com/rochefort-lab/fissa/pull/193>`__,
   `#194 <https://github.com/rochefort-lab/fissa/pull/194>`__,
   `#204 <https://github.com/rochefort-lab/fissa/pull/204>`__,
   `#207 <https://github.com/rochefort-lab/fissa/pull/207>`__,
   `#210 <https://github.com/rochefort-lab/fissa/pull/210>`__,
   `#214 <https://github.com/rochefort-lab/fissa/pull/214>`__,
   `#218 <https://github.com/rochefort-lab/fissa/pull/218>`__,
   `#232 <https://github.com/rochefort-lab/fissa/pull/232>`__,
   `#233 <https://github.com/rochefort-lab/fissa/pull/233>`__,
   `#236 <https://github.com/rochefort-lab/fissa/pull/236>`__,
   `#253 <https://github.com/rochefort-lab/fissa/pull/253>`__)

.. _v1.0.0 Dev changes:

Dev changes
~~~~~~~~~~~

-  Changed the code style to black.
   (`#215 <https://github.com/rochefort-lab/fissa/pull/215>`__,
   `#258 <https://github.com/rochefort-lab/fissa/pull/258>`__)
-  Add pre-commit hooks to enforce code style and catch pyflake errors.
   (`#161 <https://github.com/rochefort-lab/fissa/pull/161>`__,
   `#180 <https://github.com/rochefort-lab/fissa/pull/180>`__,
   `#217 <https://github.com/rochefort-lab/fissa/pull/217>`__,
   `#234 <https://github.com/rochefort-lab/fissa/pull/234>`__,
   `#261 <https://github.com/rochefort-lab/fissa/pull/261>`__)
-  Migrate CI test suite to GitHub Actions.
   (`#154 <https://github.com/rochefort-lab/fissa/pull/154>`__,
   `#195 <https://github.com/rochefort-lab/fissa/pull/195>`__)
-  Various changes and updates to the test suite.
   (`#170 <https://github.com/rochefort-lab/fissa/pull/170>`__,
   `#191 <https://github.com/rochefort-lab/fissa/pull/191>`__,
   `#197 <https://github.com/rochefort-lab/fissa/pull/197>`__,
   `#201 <https://github.com/rochefort-lab/fissa/pull/201>`__
   `#202 <https://github.com/rochefort-lab/fissa/pull/202>`__,
   `#211 <https://github.com/rochefort-lab/fissa/pull/211>`__,
   `#221 <https://github.com/rochefort-lab/fissa/pull/221>`__,
   `#222 <https://github.com/rochefort-lab/fissa/pull/222>`__,
   `#226 <https://github.com/rochefort-lab/fissa/pull/226>`__,
   `#235 <https://github.com/rochefort-lab/fissa/pull/235>`__,
   `#255 <https://github.com/rochefort-lab/fissa/pull/255>`__)
-  Notebooks are now automatically deployed on github pages.
   (`#178 <https://github.com/rochefort-lab/fissa/pull/178>`__)


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
    Add ``python_requires`` to package metadata, specifying Python 2.7 or >=3.5 is required.
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
