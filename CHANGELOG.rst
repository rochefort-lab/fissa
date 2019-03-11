Changelog
=========

All notable changes to this project will be documented here.

The format is based on `Keep a
Changelog <https://keepachangelog.com/en/1.0.0/>`__, and this project
adheres to `Semantic
Versioning <https://semver.org/spec/v2.0.0.html>`__.

Categories for changes are: Added, Changed, Deprecated, Removed, Fixed,
Security.


Version `0.6.1 <https://github.com/rochefort-lab/fissa/tree/0.6.1>`__
---------------------------------------------------------------------

Release date: 2019-03-11.
Full commit changelog
`on github <https://github.com/rochefort-lab/fissa/compare/0.6.0...0.6.1>`__.

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
Full commit changelog
`on github <https://github.com/rochefort-lab/fissa/compare/0.5.3...0.6.0>`__.

Added
~~~~~

-  Python 3 compatibility.
   (`#33 <https://github.com/rochefort-lab/fissa/pull/33>`__)
-  Documentation generation, with Sphinx, Sphinx-autodoc, and Napoleon.
   (`#38 <https://github.com/rochefort-lab/fissa/pull/38>`__)


Version `0.5.3 <https://github.com/rochefort-lab/fissa/tree/0.5.3>`__
---------------------------------------------------------------------

Release date: 2019-02-18.
Full commit changelog
`on github <https://github.com/rochefort-lab/fissa/compare/0.5.2...0.5.3>`__.

Fixed
~~~~~

-  Fix f0 detection with low sampling rates.
   (`#27 <https://github.com/rochefort-lab/fissa/pull/27>`__)


Version `0.5.2 <https://github.com/rochefort-lab/fissa/tree/0.5.2>`__
---------------------------------------------------------------------

Release date: 2018-03-07.
Full commit changelog
`on github <https://github.com/rochefort-lab/fissa/compare/0.5.1...0.5.2>`__.

Changed
~~~~~~~

-  The default alpha value was changed from 0.2 to 0.1.
   (`#20 <https://github.com/rochefort-lab/fissa/pull/20>`__)


Version `0.5.1 <https://github.com/rochefort-lab/fissa/tree/0.5.1>`__
---------------------------------------------------------------------

Release date: 2018-01-10.
Full commit changelog
`on github <https://github.com/rochefort-lab/fissa/compare/0.5.0...0.5.1>`__.

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
