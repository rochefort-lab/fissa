Releasing a new version
=======================

In this document, we describe how to release a new version of the codebase.


## 1. Version numbering

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

The rest of this guide includes code blocks which you can use to automate some of the steps in the release.
To help this process, you can adapt the following code block to declare the version number of your release with the variable `vMNP`.

First declare the variable `vMNP`, such as
```sh
vMNP="v0.1.0"
```

Then run the following code block to assign some other variables based on `vMNP`.
```sh
MNP="${vMNP#v}"
vMNP="v$MNP"
MNx="${MNP%.*}.x"
vMNx="v$MNx"

echo "\
vMNP = '$vMNP'
MNP  = '$MNP'
vMNx = '$vMNx'
MNx  = '$MNx'"
```


## 2. Collect changes for release

Next, we must create a release candidate branch.

### 2.1 Determination of branch name

The release candidate branch should be named `rel_M.N.P`, where `M` is your major version, `N` is your minor version, `P` is the patch version, and `v` is the literal character `v`.
In this document, we will refer to this as your release candidate branch.

For instance, if you are releasing version `0.5.1`, the branch should be named `rel_0.5.1`.

Meanwhile, the stable `vM.N.x` branch (where `x` is literal character `x`) should always contain the latest patch to minor version `M.N`.
For instance, branch `v0.5.x` contains the latest patched version of the `0.5` series of releases, which may be `0.5.2`, say.

### 2.2 Create release candidate branch

Create your release candidate branch whose name bears the format `rel_M.N.P`.

Merge all changes which should be included in your new release into the release candidate branch, `rel_M.N.P`.

#### 2.2a Create release candidate branch from bleeding-edge

Typically, your new release will contain everything currently in the bleeding-edge codebase from the master branch of the repository.
```sh
git checkout master
git pull
git checkout -b "rel_$MNP"
```

#### 2.2b Create release candidate branch to patch previous minor release

However, if you are releasing a patch for a previous version (the current release is `M.(N+1).0`, or later), you should not merge master into `rel_M.N.P` as it will contain features or API changes not suitable for your patch release.
Instead, you should merge the relevant bug-fix branches directly into `rel_M.N.P`.
If the bug-fix branches you need have been deleted, you may have to [cherry-pick](https://git-scm.com/docs/git-cherry-pick) commits from master.
```sh
git checkout "$vMNx"
git pull
git checkout -b "rel_$MNP"
```


## 3. Update metadata for your release

On your release-candidate branch, `rel_M.N.P`, update the metadata to reflect your new release.

### 3.1 Update the version number

In `__meta__.py`, update the version number as required (see above and the [Semantic Versioning standard][semver] for details).

```sh
gedit fissa/__meta__.py
# edit file and save changes
git add fissa/__meta__.py
git commit -m "REL: Version $MNP"
```

### 3.2 Update the CHANGELOG

On your release-candidate branch, update the CHANGELOG.
```sh
gedit CHANGELOG.rst
```

Follow the style of the existing CHANGELOG entries to add a new entry for your release.
Use as many of the categories (Added, Changed, Deprecated, Removed, Fixed, Security) as necessary.

List all non-trivial changes, following the guidance of [Keep a Changelog].
If you're not sure what has changed, consult the [list of closed pull requests](https://github.com/rochefort-lab/fissa/pulls?q=is%3Apr+is%3Aclosed+sort%3Aupdated-desc), check which have closed recently and whether they are worth including in the CHANGELOG.
If you have nothing to list in the CHANGELOG, you should seriously question whether it is a good idea to create a new release!

Usually, previous pull requests which added noteworthy changes have also made changes to the CHANGELOG under the `Unreleased` section at the top, in which case you can move them down into your new release.

Your CHANGELOG entry should begin like so:
```
Version `M.N.P <https://github.com/rochefort-lab/fissa/tree/M.N.P>`__
---------------------------------------------------------------------

Release date: YYYY-MM-DD.
`Full commit changelog <https://github.com/rochefort-lab/fissa/compare/A.B.C...M.N.P>`__.
```
Where `M.N.P` is the new release, `A.B.C < M.N.P` is the previous ancestor to the new release, and `YYYY-MM-DD` is today's date.
The Full Changelog link will use the release tags on github to automatically generate a comparison.

The changes you list should be grouped into categories: Added, Changed, Deprecated, Removed, Fixed, Security.
See the description at the top of the CHANGELOG for more details.

For each category which appears in the Changelog of the new release, make sure to include an rST anchor declaration, such as the example below.
This tells sphinx what anchor to use for the subheading (in this case Fixed), which appears many times within the changelog document.
```
.. _vM.N.P Fixed:

Fixed
~~~~~
-   Details of bug which was fixed.
    (`#1 <https://github.com/rochefort-lab/fissa/pull/1>`__)
```

For each change, the pull request which implemented that change in the master branch should also be linked to.

Note that the CHANGELOG should not contain an "Unreleased" section on the release-candidate branch.

Once you are done, add and commit your addition to the CHANGELOG.
```sh
git add CHANGELOG.rst
git commit -m "DOC: Add $vMNP to CHANGELOG"
```

### 3.3 Push your changes to the rel_M.N.P branch on GitHub

Push your changes with
```sh
git push -u origin "rel_$MNP"
```
(where `rel_M.N.P` is replaced with your release candidate branch) to establish tracking with the remote branch.

### 3.4 Ensure CHANGELOG is formatted correctly

Check the rendering of the CHANGELOG on GitHub at
<https://github.com/rochefort-lab/fissa/blob/rel_M.N.P/CHANGELOG.rst>,
where `rel_M.N.P` is replaced with your release candidate branch.

```sh
sensible-browser https://github.com/rochefort-lab/fissa/blob/rel_$MNP/CHANGELOG.rst
```

## 4. Make a Pull Request

You'll need to make a PR to merge the new release into a target stable branch.

## 4.1 Ensure target stable branch exists

If you are releasing a new major or minor version, you may first have to instantiate a new stable branch, which will be the target of the PR.

Browse the the list of [branches](https://github.com/rochefort-lab/fissa/branches) for the repository, and ensure that a branch bearing the name corresponding to `vM.N.x` exists.
If not, create a new branch for it based on the last stable release and push it to github.

## 4.2 Make the Pull Request

If you are creating a new patch release, initiate a pull request to merge `rel_M.N.P` into the stable branch for this minor release, named `vM.N.x`,
<https://github.com/rochefort-lab/fissa/compare/vM.N.x...rel_M.N.P?expand=1>.
```sh
sensible-browser "https://github.com/rochefort-lab/fissa/compare/v${MNx}...rel_${MNP}?expand=1&title=REL:%20Release%20version%20${MNP}"
```
You can use the contents of the CHANGELOG update as the basis of the body of your PR, but you will need to convert it from RST to markdown format first.
```sh
pandoc --from rst --to markdown+hard_line_breaks CHANGELOG.rst | sed '/^:::/d' > .CHANGELOG.md
```

Unless the release was pre-approved, you'll need to wait for another maintainer to review the release candidate before you can merge it into the stable release branch.
Don't delete the release-candidate branch when the PR is closed, as we'll make use of it again in a later step.


## 5. Confirm tests pass

After the release-candidate branch has been merged into the release branch, you must release the new branch.
First, double-check the test suite runs successfully.
The test suite should also have been run on the continuous integration server during development.
This step is included to double-check what you are about to submit is a viable copy of the code.

### 5.1 Checkout the release branch

Checkout and update your local copy of the release branch, `vM.N.x`.
```sh
git fetch
git checkout "v$MNx"
git pull
```

Then use `git status` to make sure you don't have any unstaged changes.
We need to run the tests on a clean copy of the branch.

### 5.2 Run test suite

Run the unit test suite, with either
```sh
py.test
```
or
```sh
python setup.py test
```
and make sure all the unit tests pass locally.


## 6. Build distribution

Follow the instructions as per the [PyPI tutorial](https://packaging.python.org/tutorials/packaging-projects/) to build your distribution.
```sh
rm -f .CHANGELOG.md
rm -rf dist
python -m pip install --upgrade setuptools wheel twine
python setup.py sdist bdist_wheel --universal
```

## 7. Test the submission

### 7.1 Upload to the PyPI test server

Use twine to upload your new distribution to the PyPI test server
```
python -m twine upload --repository-url https://test.pypi.org/legacy/ dist/*
```
You will be prompted for your test.pypi username and password.

### 7.2 Create a venv, and install from the PyPI test server

`cd` away from your local git repository, and create a new virtual environment

```sh
REPODIR="$(pwd)"
cd ~
rm -rf pypi_test_env
virtualenv -p python3 pypi_test_env
source pypi_test_env/bin/activate
python -m pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple fissa
```

### 7.3 Test the installation

Check whether you can import the new package, and that the version installed is `M.N.P`.
```sh
python -c "import fissa; print(fissa.__version__)"
```

### 7.4 Remove the test venv and return to repository

Return to your repository
```sh
deactivate
rm -rf pypi_test_env
cd "$REPODIR"
```


## 8. Make a new release-tag

On GitHub, [make a new release-tag](https://github.com/rochefort-lab/fissa/releases/new).

### 8.1 Release tag

Your release-tag should be named `M.N.P`.
Note that, unlike the release candidate branch, there is no `rel_` or `v` at the beginning of the tag name.
```sh
sensible-browser "https://github.com/rochefort-lab/fissa/releases/new?target=${vMNx}&tag=${MNP}&title=Version%20${MNP}"
```

**Note:** Be sure to set the target of the tag to your stable release branch `M.N.x`.

### 8.2 Release title

The title of your release should be `Version M.N.P`.

### 8.3 Release body

For the body of the release, we will use your entry in the CHANGELOG, including subheadings such as `Added`, `Fixed`, etc.

However, you will need to convert the formatting from RST to Markdown.
The full changelog can be converted using this command:
```sh
pandoc --from rst --to markdown+hard_line_breaks CHANGELOG.rst | sed '/^:::/d' > .CHANGELOG.md
```
From the converted document, you can select the relevant section to include in the github release body.

If you make a formatting mistake and realise after publishing the release on GitHub, don't worry as the release metadata can be editted.

When you are done, delete the temporary file generated by pandoc.
```sh
rm .CHANGELOG.md
```

## 9. Push to PyPI

You will need to be a maintainer of this project in order to push a new version.

**Note that this step is irreversible.**
If you push the wrong thing here, you can't overwrite it.
```sh
python -m pip install --upgrade setuptools wheel twine
python -m twine upload dist/*
```
You will be prompted for your username and password, which must be associated with a maintainer for the project.

### Try out the PyPI release

You'll first need to wait a couple of minutes for the new PyPI version to become available.

Again, we create a fresh virtual environment and install the new release into it.
```sh
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
```sh
deactivate
rm -rf pypi_test_env
cd "$REPODIR"
```

## 10. Update metadata on the master branch

You need to merge the updated CHANGELOG into the master branch.
There are two procedures for doing this, depending on whether your new release has the highest version number, or if it was a patch release for an older version.

### 10.1a If your release has the highest version number

If your new release has the highest version number, you should do this on your `rel_M.N.P` branch.

First, update the `rel_M.N.P` to contain the release itself, contained in branch `vM.N.x`.
```sh
git fetch
git checkout "rel_$MNP"
git merge "origin/v$MNx"
```

If you've followed the steps in this guide in the order they are given, when you perform the `git fetch` step you should receive a new tag called `M.N.P` from the origin remote.
The merge into `rel_M.N.P` will be done with the fast forward method, adding one merge commit from the PR to its history.

Now proceed to step 10.2.

### 10.1b If your release is a patch for a stale version

If your release does not have the highest version number (because it is a patch for an older release), you should do the following:
```sh
git fetch
git checkout master
git checkout -b "doc_changelog-$MNP"
git cherry-pick "origin/rel_$MNP"
```

This will take the last commit from your release-candidate branch, and add it to the commit history.
If you have followed this guide, the last commit will be the CHANGELOG update.
You will have to handle a merge conflict resulting from the cherry-pick.

Afterwards, you can double-check which commits were included with `git log`, or an interactive rebase `git rebase -i master`.

### 10.2 Add Unreleased section back to CHANGELOG

Our unstable version on the master branch should always contain a CHANGELOG entry which compares the last stable release with the current unstable version.

Edit the CHANGELOG
```sh
gedit CHANGELOG.rst
```
to include an entry at the top of the list of entries, which reads as follows:
```
Unreleased
----------

`Full commit changelog <https://github.com/rochefort-lab/fissa/compare/A.B.C...master>`__.

```
where `A.B.C >= M.N.P` is the latest release, including your forthcoming `M.N.P` release.

Commit this change.
```sh
git add CHANGELOG.rst
git commit -m "DOC: Add Unreleased section to CHANGELOG"
```

### 10.4 Update version number

Check the [current version number on the master branch](https://github.com/rochefort-lab/fissa/blob/master/fissa/__meta__.py).

If you have released a new major/minor version with a version number higher than this, you need to update the version on the master branch to be `M.N.dev0`.

If you have released a patch, or the master branch otherwise already has the correct version number, just make sure your version number in `__meta__.py` matches that of master (check with `less fissa/__meta__.py`).

Edit the `__meta__.py` file
```sh
gedit fissa/__meta__.py
```
to be `A.B.dev0`, where `A.B` is the latest minor version.

Once you've edited the `__meta__.py` file, commit this change (replacing the placeholder text `A.B.dev0` with the actual current minor version in this example code).
```sh
git add fissa/__meta__.py
git commit -m "REL: Change bleeding-edge version number to A.B.dev0"
```

### 10.5 Push branch and create pull request

Push the branch to the repository and go to the [repository page][our repo] to create a pull request into the master branch.
```sh
git push -u origin $(git symbolic-ref --short HEAD)
sensible-browser "https://github.com/rochefort-lab/fissa/compare/$(git symbolic-ref --short HEAD)?expand=1&labels=doc&title=DOC:%20Add%20${vMNP}%20to%20CHANGELOG&body=Adds%20new%20release%20${vMNP}%20to%20the%20CHANGELOG."
```

The title of the PR should be `DOC: Add vM.N.P to CHANGELOG`.

Once you've created the PR, double-check that the only changed files are `CHANGELOG` and (possibly) `__meta__.py`.
If they are, once the continuous integration tests have passed, you can merge the PR into master.
If there are more changes beyond this, your PR needs further review.


  [our repo]: http://github.com/rochefort-lab/fissa/
  [numpy release]: https://github.com/numpy/numpy/blob/master/doc/HOWTO_RELEASE.rst.txt
  [semver]: https://semver.org/spec/v2.0.0.html
  [PEP440]: https://www.python.org/dev/peps/pep-0440
  [Keep a Changelog]: https://keepachangelog.com
