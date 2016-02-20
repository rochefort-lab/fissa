Contributing code
=================

Reporting Issues
----------------

If you encounter a problem when implementing or using FISSA, we want
to hear about it!

### Gitter

To get help resolving implementation difficulties, or similar one-off
problems, please ask for help on our [gitter channel].

### Reporting Bugs and Issues

If you experience a bug, please report it by opening a [new issue].
When reporting issues, please include code that reproduces the issue
and whenever possible, an image that demonstrates the issue. The best
reproductions are self-contained scripts with minimal dependencies.

Make sure you mention the following things:
- What did you do?
- What did you expect to happen?
- What actually happened?
- What versions of FISSA and Python are you using, and on which
  operating system?

### Feature requests

If you have a new feature or enhancement to an existing feature you
would like to see implemented, please check the list of
[existing issues][issues] and if you can't find it make a [new issue]
to request it. If you do find it in the list, you can post a comment
saying `+1` (or `:+1:` if you are a fan of emoticons) to indicate your
support for this feature.


Documentation
-------------

We are glad to accept any sort of documentation: function docstrings,
tutorials, Jupyter notebooks demonstrating implementation details, etc.

reStructuredText documents and notebooks live in the source code
repository under the `doc/` directory.

### Docstrings

Documentation for classes and functions should follow the
[format prescribed for numpy][numpy documenting].

A complete example of this is available [here][numpy documenting example].


How to contribute
-----------------

The preferred way to contribute to FISSA is to fork the
[main repository][our repo] on GitHub.

1. Fork the [project repository][our repo].
   Click on the 'Fork' button near the top of the page. This creates
   a copy of the code under your account on the GitHub server.

2. Clone this copy to your local disk:

        $ git clone git@github.com:YourUserName/fissa.git
        $ cd fissa

3. Create a branch to hold and track your changes

        $ git checkout develop
        $ git checkout -b my-feature

   and start making changes. Never work in the `master` or `develop`
   branches!

4. Work on this copy on your computer using Git to do the version
   control. When you're done editing, do:

        $ git add modified_files
        $ git commit

   to record your changes in Git, writing a commit message following
   the [specifications below](#commit-messages), then push them to
   GitHub with:

        $ git push -u origin my-feature

5. Finally, go to the web page of your fork of the FISSA repo, and
   click 'Pull request' to send your changes to the maintainers for
   review. This will send an email to the committers.

If any of the above seems like magic to you, then look up the 
[Git documentation] on the web.

It is recommended to check that your contribution complies with the
following rules before submitting a pull request.

-  All public functions and methods should have informative docstrings,
   with sample usage included in the doctest format when appropriate.

-  All unit tests pass. Check with (from the top level source folder):

        $ pip install pytest
        $ py.test

-  Code with good unit test coverage (at least 90%, ideally 100%).
   Check with

        $ pip install pytest pytest-cov
        $ py.test --cov=fissa --cov-config .coveragerc

   and look at the value in the 'Cover' column for any files you have
   added or amended.

   If the coverage value is too low, you can inspect which lines are or
   are not being tested by generating a html report.
   After opening this, you can navigate to the appropriate module and
   see lines which were not covered, or were only partially covered.
   If necessary, you can do this as follows:

        $ py.test --cov=fissa --cov-config .coveragerc --cov-report html --cov-report term-missing
        $ sensible-browser ./htmlcov/index.html

-  No [pyflakes] warnings. Check with:

        $ pip install pyflakes
        $ pyflakes path/to/module.py

-  No [PEP8] warnings. Check with:

        $ pip install pep8
        $ pep8 path/to/module.py

   AutoPEP8 can help you fix some of the easier PEP8 errors.

        $ pip install autopep8
        $ autopep8 -i -a -a path/to/module.py

   Note that using the `-i` flag will modify your existing file in-place,
   so be sure to save any changes made in your editor beforehand.

These tests can be collectively performed in one line with:

    $ pip install -r requirements-dev.txt
    $ py.test --flake8 --cov=fissa --cov-config .coveragerc --cov-report html --cov-report term

### Commit messages

Commit messages should be clear, precise and stand-alone.
Lines should not exceed 72 characters.

It is useful to indicate the nature of your commits with a commit flag, as
described in the [numpy development guide][numpy workflow commits].

You can use these flags at the start of your commit messages:

    API: an (incompatible) API change
    BLD: change related to building the package
    BUG: bug fix
    CI: change continuous integration build
    DEP: deprecate something, or remove a deprecated object
    DEV: development tool or utility
    DOC: documentation
    ENH: enhancement
    MAINT: maintenance commit (refactoring, typos, etc.)
    REV: revert an earlier commit
    RF: refactoring
    STY: style fix (whitespace, PEP8)
    TST: addition or modification of tests
    REL: related to releases


Notes
-----

This document was based on the contribution guidelines for
[sklearn][sklearn contributing],
[numpy][numpy workflow] and
[Pillow][Pillow contributing].


  [our repo]: http://github.com/rochefort-tools/fissa/
  [issues]: https://github.com/rochefort-lab/fissa/issues
  [new issue]:https://github.com/rochefort-lab/fissa/issues/new
  [gitter channel]: https://gitter.im/rochefort-lab/fissa
  [PEP8]: https://www.python.org/dev/peps/pep-0008/
  [pyflakes]: https://pypi.python.org/pypi/pyflakes
  [Git documentation]: http://git-scm.com/documentation
  [numpy workflow]: https://docs.scipy.org/doc/numpy-1.10.1/dev/gitwash/development_workflow.html
  [numpy workflow commits]: https://docs.scipy.org/doc/numpy-1.10.1/dev/gitwash/development_workflow.html#writing-the-commit-message
  [numpy documenting]: https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt#sections
  [numpy documenting example]: https://sphinxcontrib-napoleon.readthedocs.org/en/latest/example_numpy.html
  [sklearn contributing]: https://github.com/scikit-learn/scikit-learn/blob/master/CONTRIBUTING.md
  [sklearn guide]: http://scikit-learn.org/stable/developers/contributing.html#contributing-code
  [Pillow contributing]: https://github.com/python-pillow/Pillow/blob/master/CONTRIBUTING.md
