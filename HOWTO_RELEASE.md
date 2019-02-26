Releasing a new version
=======================

In this document, we describe how to release a new version of the codebase.


1. Version numbering
--------------------

First, you must determine the version number for your new release.

For version numbering, we use the [Semantic Versioning 2.0.0 standard][semver].
Briefly, the version number is always MAJOR.MINOR.PATCH (e.g. 0.3.4).

With changes, increment the:

    MAJOR version when you make incompatible API changes,
    MINOR version when you add functionality in a backwards-compatible manner, and
    PATCH version when you make backwards-compatible bug fixes.

The bleeding-edge development version is on the [master branch][our repo] of the repository.
Its version number is always of the form `x.y.dev0`.
This version numbering, including the developmental version, is compliant with [PEP440].

Let us suppose you are making a new release `M.N.P`, where `M` is your major version, `N` is your minor version, and `P` is your patch version.

This release may be any of the following:
- A new major release
- A new minor release
- A patch for the latest minor release
- A patch for a pre-existing minor release


2. Collect changes for release
------------------------------

Next, we must create a stable copy of the code base as it will be released.

This should be put on a branch corresponding to the minor version of the release.

### 2.1 Determination of branch name

The branch should be named `vM.N.x`, where `M` is your major version, `N` is your minor version, `v` is the literal character `v`, and `x` is the literal character `x`.
In this document, we will refer to this as your release candidate branch.

For instance, if you are releasing version `0.5.0`, the branch should be named `v0.5.x`.

As a sidenote, the `vM.N.x` branch should always contain the latest patch to minor version `M.N`.
For instance, branch `v0.5.x` contains the latest patched version of the `0.5` series of releases, which may be `0.5.2`, say.

### 2.2 Create release candidate branch

If it does not already exist, create your release candidate branch whose name bears the format `vM.N.x`.

### 2.3 Merge your changes

Merge all changes which should be included in your new release into the release candidate branch, `vM.N.x`.
If you are releasing a new minor or major version, you may wish to merge the master branch into your release-candidate branch.

However, if you are releasing a patch for a previous version (the current release is `M.(N+1).0`, or later), you should not merge master into `vM.N.x` as it will contain features or API changes not suitable for your patch release.
Instead, you should merge the relevant bug-fix branches directly into `vM.N.x`.
If the bug-fix branches you need have been deleted, you may have to [cherry-pick](https://git-scm.com/docs/git-cherry-pick) commits from master.


3. Update metadata for your release
-----------------------------------

On your release-candidate branch, `vM.N.x`, update the metadata to reflect your new release.

### 3.1 Update the version number

In `__meta__.py`, update the version number as required (see above and the [Semantic Versioning standard][semver] for details).

### 3.2 Update the CHANGELOG

On your release-candidate branch, update the CHANGELOG.

Follow the style of the existing CHANGELOG entries to add a new entry for your release.
Use as many of the categories (Added, Changed, Deprecated, Removed, Fixed, Security) as necessary.

List all non-trivial changes, following the guidance of [Keep a Changelog].
If you're not sure what has changed, consult the [list of closed pull requests](https://github.com/rochefort-lab/fissa/pulls?q=is%3Apr+is%3Aclosed+sort%3Aupdated-desc), check which have closed recently and whether they are worth including in the CHANGELOG.
If you have nothing to list in the CHANGELOG, you should seriously question whether it is a good idea to create a new release!

Usually, previous pull requests which added noteworthy changes have also made changes to the CHANGELOG under the `Unreleased` section at the top, in which case you can move them down into your new release.

Your CHANGELOG entry should begin like so:
```
`M.N.P <https://github.com/rochefort-lab/fissa/tree/A.B.C>`__ (YYYY-MM-DD)
-------------------------------------------------------------------------

`Full Changelog <https://github.com/rochefort-lab/fissa/compare/A.B.C...M.N.P>`__
```
Where `M.N.P` is the new release, `A.B.C < M.N.P` is the previous ancestor to the new release, and `YYYY-MM-DD` is today's date.
The Full Changelog link will use the release tags on github to automatically generate a comparison.

The changes you list should be grouped into categories: Added, Changed, Deprecated, Removed, Fixed, Security.
See the description at the top of the CHANGELOG for more details.

For each change, the pull request which implemented that change in the master branch should also be linked to.

### 3.3 Update the Unreleased entry in CHANGELOG

There should always be a section at the top of the CHANGELOG which compares the last stable release with the current unstable version.

This entry should read as follows:
```
`Unreleased <https://github.com/rochefort-lab/fissa>`__
-----------------------------------------------------------------

`Full Changelog <https://github.com/rochefort-lab/fissa/compare/A.B.C...master>`__
```
where `A.B.C >= M.N.P` is the latest release, including your forthcoming `M.N.P` release.

If your new release will become the highest version released (it is for a new major or minor version, or a patch for the latest minor version), you will need to update the URL for the Unreleased Full Changelog comparison.

### 3.4 Push your changes to the vM.N.x branch on GitHub


### 3.5 Ensure CHANGELOG is formatted correctly


4. Confirm tests pass
---------------------

The test suite should also have been run on the continuous integration server during development.
This step is included to double-check what you are about to submit is a viable copy of the code.

### 4.1 Checkout release-candidate branch

Checkout the release-candidate branch, `vM.N.x`.

Then use `git status` to make sure you don't have any unstaged changes.
We need to run the tests on a clean copy of the branch.

### 4.2 Run test suite

Run the unit test suite, with either
```
py.test
```
or
```
python setup.py test
```
and make sure all the unit tests pass locally.


5. Build distribution
---------------------

Follow the instructions in the [PyPI tutorial](https://packaging.python.org/tutorials/packaging-projects/) to build your distribution.
```
rm -rf dist
python -m pip install --upgrade setuptools wheel
python -m pip install --upgrade twine
python setup.py sdist bdist_wheel
```

6. Test the submission
----------------------

### 6.1 Upload to the PyPI test server

Use twine to upload your new distribution to the PyPI test server
```
python -m twine upload --repository-url https://test.pypi.org/legacy/ dist/*
```
You will be prompted for your test.pypi username and password.

### 6.2 Create a venv, and install from the PyPI test server

`cd` away from your local git repository, and create a new virtual environment

```
REPODIR="$(pwd)"
cd ~
rm -rf pypi_test_env
virtualenv -p /usr/bin/python3 pypi_test_env
source pypi_test_env/bin/activate
python -m pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple fissa
```

### 6.3 Test the installation

Check whether you can import the new package.
```
python -c "import fissa; print(fissa.__version__)"
```

### 6.4 Remove the test venv and return to repository

Return to your repository
```
deactivate
rm -rf pypi_test_env
cd "$REPODIR"
```


7. Make a new release-tag
-------------------------

On GitHub, [make a new release-tag](https://github.com/rochefort-lab/fissa/releases/new).

### 7.1 Release tag

Your release-tag should be named `M.N.P`.
Note that, unlike the release candidate branch, there is no `v` character at the beginning of the tag name.

**Note:** Be sure to set the target of the tag to your release candidate branch `vM.N.x`.

### 7.2 Release title

The title of your release should be `Version M.N.P`.

### 7.3 Release body

For the body of the release, we will use your entry in the CHANGELOG, including subheadings such as `Added`, `Fixed`, etc.

However, you will need to convert the formatting from RST to Markdown.
The full changelog can be converted using this command:
```
pandoc --from rst --to markdown CHANGELOG.rst >> .CHANGELOG.md
```
From the converted document, you can select the relevant section to include in the github release body.

If you make a formatting mistake and realise after publishing the release on GitHub, don't worry as the release metadata can be editted.


8. Push to PyPI
---------------

You will need to be a maintainer of this project in order to push a new version.

**Note that this step is irreversible.**
If you push the wrong thing here, you can't overwrite it.
```
python -m twine upload dist/*
```
You will be prompted for your username and password, which must be associated with a maintainer for the project.

### Try out the PyPI release

Again, we create a fresh virtual environment and install the new release into it.
```
REPODIR="$(pwd)"
cd ~
rm -rf pypi_test_env
virtualenv -p /usr/bin/python3 pypi_test_env
source pypi_test_env/bin/activate
python -m pip install fissa
python -c "import fissa; print(fissa.__version__)"
```

You should confirm that the version number outputted matches the version of your new release.

We now remove the temporary environment and return to the repository.
```
deactivate
rm -rf pypi_test_env
cd "$REPODIR"
```

9. Update metadata on the master branch
---------------------------------------

### 9.1 Create a branch for your pull request

Create a new branch, titled `rel_M.N.P`, based on origin's current master branch.
```
git checkout master
git pull
git checkout -b rel_M.N.P
```

(If you've followed the steps in this guide in the order they are given, when you perform the `git pull` step you should receive a new tag called `M.N.P` from the origin remote.)

### 9.2 Merge your metadata updates

You will need to merge your CHANGELOG update, and possibly your new version number into the master branch.

If the only commits on `vM.N.x` which are not on the master branch are your metadata updates, you can directly merge the branch.
```
git merge vM.N.x
```

If not, cherry-pick the last commit from your `vM.N.x` branch
```
git cherry-pick vM.N.x
```
or the last two commits from your `vM.N.x` branch
```
git cherry-pick vM.N.x^^..vM.N.x
```
as appropriate.

Afterwards, you can double-check which commits were included with `git log`, or an interactive rebase `git rebase -i master`.

### 9.3 Update version number

Check the [current version number on the master branch](https://github.com/rochefort-lab/fissa/blob/master/fissa/__meta__.py).

If you have released a new major/minor version with a version number higher than this, you need to update the version on the master branch to be `M.N.dev0`.

If you know development on the bleeding-edge master branch is progressing to a new major/minor version, you should update the version number in `__meta__.py` on the master branch to `M.(N+1).dev0` or `(M+1).0.dev0` as appropriate.

If you have released a patch, or the master branch otherwise already has the correct version number, just make sure your version number in `__meta__.py` matches that of master (check with `less fissa/__meta__.py`).

Once you've edited the `__meta__.py` file, commit this change.

### 9.4 Push branch and create pull request

Push the branch with a command formatted like
```
git push -u origin rel_M.N.P
```
and go to the [repository page][our repo] to create a pull request.

Once you've created the PR, double-check that the only changed files are `CHANGELOG` and `__meta__.py`.
If they are, once the continuous integration tests have passed, you can merge the PR into master.
If there are more changes beyond this, your PR may need further review.


10. Enable branch protection on your release-candidate branch
-------------------------------------------------------------

You can add branch protection so your (now released) release-candidate branch cannot receive force pushes or be deleted.
This is important to ensure the branch is preserved for posterity.

This can be enabled in the [settings for the repository](https://github.com/rochefort-lab/fissa/settings/branches), under the Branches subsection.
Select the Add rule option.

The pattern for your rule should be `*M.N.x`.
If the `vM.N.x` branch already existed before you made this release, there should already be a rule for it listed.

When creating the rule, you may additionally select the checkbox for "Require status checks to pass before merging", along with the unit test CI (Travis) as the requirement.
When this is enabled, the `vM.N.x` branch can no longer be updated directly - only through pull requests.


11. Activate the release on ReadTheDocs
---------------------------------------

In the [settings on ReadTheDocs](https://readthedocs.org/dashboard/fissa/versions/), locate the release tag (`M.N.P`) in the versions list and activate it.


  [our repo]: http://github.com/rochefort-lab/fissa/
  [numpy release]: https://github.com/numpy/numpy/blob/master/doc/HOWTO_RELEASE.rst.txt
  [semver]: https://semver.org/spec/v2.0.0.html
  [PEP440]: https://www.python.org/dev/peps/pep-0440
  [Keep a Changelog]: https://keepachangelog.com
