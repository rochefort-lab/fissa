Releasing the code
==================
Version info
------------
For version numbering, we use the [Semantic Versioning 2.0.0 standard][semver].

Briefly, the version number is always MAJOR.MINOR.PATCH (e.g. 0.3.4).

With changes, increment the:

    MAJOR version when you make incompatible API changes,
    MINOR version when you add functionality in a backwards-compatible manner, and
    PATCH version when you make backwards-compatible bug fixes.

The bleeding-edge development version is on the [master branch][our repo] of
the repository. Its version number is always of the form x.x.x.dev.

Releasing a new version
-----------------------
### Fully update the master branch
Make sure the master branch passes all tests, and is fully updated as required
for the update.

### Update the version number
In [setup.py][setup py] update the version number as required (see above and
  the [Semantic Versioning standard][semver] for details).

### Make a new release-tag
[TODO: explain how to make a new release through the Github interface]

### Update PyPI
[TODO: explain how to update PyPI]

Notes
-----

This document was based on the release guidelines for
[numpy][numpy release] and
[Semantic Versioning 2.0.0 standard][semver].

  [our repo]: http://github.com/rochefort-tools/fissa/
  [numpy release]: https://github.com/numpy/numpy/blob/master/doc/HOWTO_RELEASE.rst.txt
  [setup py]: https://github.com/rochefort-lab/fissa/blob/master/setup.py
  [semver]: https://semver.org/spec/v2.0.0.html
