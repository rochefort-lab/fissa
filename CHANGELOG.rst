Changelog
=========

All notable changes to this project will be documented here.

The format is based on `Keep a
Changelog <https://keepachangelog.com/en/1.0.0/>`__, and this project
adheres to `Semantic
Versioning <https://semver.org/spec/v2.0.0.html>`__.

Categories for changes are: Added, Changed, Deprecated, Removed, Fixed,
Security.


`Unreleased <https://github.com/rochefort-lab/fissa/tree/HEAD>`__
-----------------------------------------------------------------

`Full Changelog <https://github.com/rochefort-lab/fissa/compare/0.5.3...HEAD>`__

Added
~~~~~

-  Python 3 compatibility.
   `#33 <https://github.com/rochefort-lab/fissa/pull/33>`__
-  Documentation generation, with Sphinx, Sphinx-autodoc, and Napoleon.
   `#38 <https://github.com/rochefort-lab/fissa/pull/38>`__
   (`scottclowe <https://github.com/scottclowe>`__)


`0.5.3 <https://github.com/rochefort-lab/fissa/tree/0.5.3>`__ (2019-02-18)
--------------------------------------------------------------------------

`Full Changelog <https://github.com/rochefort-lab/fissa/compare/0.5.2...0.5.3>`__

Fixed
~~~~~

-  Fix f0 detection with low sampling rates
   `#27 <https://github.com/rochefort-lab/fissa/pull/27>`__
   (`scottclowe <https://github.com/scottclowe>`__)


`0.5.2 <https://github.com/rochefort-lab/fissa/tree/0.5.2>`__ (2018-03-07)
--------------------------------------------------------------------------

`Full Changelog <https://github.com/rochefort-lab/fissa/compare/0.5.1...0.5.2>`__

Changed
~~~~~~~

-  The default alpha value was changed from 0.2 to 0.1
   `#20 <https://github.com/rochefort-lab/fissa/pull/20>`__
   (`swkeemink <https://github.com/swkeemink>`__)


`0.5.1 <https://github.com/rochefort-lab/fissa/tree/0.5.1>`__ (2018-01-10)
--------------------------------------------------------------------------

`Full Changelog <https://github.com/rochefort-lab/fissa/compare/0.5.0...0.5.1>`__

Added
~~~~~

-  Possibility to define custom datahandler script for other formats
-  Added low memory mode option to load larger tiffs frame-by-frame
   `#14 <https://github.com/rochefort-lab/fissa/pull/14>`__
   (`swkeemink <https://github.com/swkeemink>`__)
-  Added option to use ICA instead of NMF (not recommended, but is a lot
   faster)
-  Added the option for users to define a custom data and ROI loading
   script `#13 <https://github.com/rochefort-lab/fissa/pull/13>`__
   (`swkeemink <https://github.com/swkeemink>`__)

Fixed
~~~~~

-  Fixed custom datahandler usage
   `#14 <https://github.com/rochefort-lab/fissa/pull/14>`__
   (`swkeemink <https://github.com/swkeemink>`__)
-  Documentation fixes
   `#12 <https://github.com/rochefort-lab/fissa/pull/12>`__
   (`swkeemink <https://github.com/swkeemink>`__)


`0.5.0 <https://github.com/rochefort-lab/fissa/tree/0.5.0>`__ (2017-10-05)
--------------------------------------------------------------------------

Initial release
