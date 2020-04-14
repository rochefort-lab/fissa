#!/usr/bin/env python


import os

from distutils.core import setup
from setuptools.command.test import test as TestCommand


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

# Can't import __meta__.py if the requirements aren't installed
# due to imports in __init__.py. This is a workaround.
meta = {}
exec(read('fissa/__meta__.py'), meta)

install_requires = read('requirements.txt').splitlines()

extras_require = {}

# Notebook dependencies for plotting
extras_require['plotting'] = read('requirements_plots.txt').splitlines()

# Dependencies for generating documentation
extras_require['docs'] = read('requirements_docs.txt').splitlines()

# Dev dependencies
extras_require['dev'] = read('requirements-dev.txt').splitlines()

# Everything as a list. Replicated items are removed by use of set with {}.
extras_require['all'] = sorted(list(
    {x for v in extras_require.values() for x in v}
))


class PyTest(TestCommand):

    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = []
        self.test_suite = True

    def run_tests(self):
        import pytest
        pytest.main(self.test_args)


setup(
    name = meta['name'],
    python_requires='>=2.7, !=3.0.*, !=3.1.*, !=3.2.*, !=3.3.*, !=3.4.*',
    install_requires = install_requires,
    extras_require = extras_require,
    version = meta['version'],
    author = meta['author'],
    author_email = meta['author_email'],
    description = meta['description'],
    url = meta['url'],
    package_dir = {meta['name']: os.path.join(".", meta['path'])},
    packages = [meta['name']],
    license = "GNU",
    long_description = read('README.rst'),
    # https://pypi.org/pypi?%3Aaction=list_classifiers
    classifiers = [
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    project_urls={
        "Documentation": "https://fissa.readthedocs.io",
        "Source Code": "https://github.com/rochefort-lab/fissa",
        "Bug Tracker": "https://github.com/rochefort-lab/fissa/issues",
        "Citation": "https://www.doi.org/10.1038/s41598-018-21640-2",
    },
    cmdclass = {'test': PyTest},
)
