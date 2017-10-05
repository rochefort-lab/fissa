#! /usr/bin/env python

import os

from distutils.core import setup
from setuptools.command.test import test as TestCommand

NAME = 'fissa'


install_requires = [	'numpy>=1.12.1',
                     'scipy>=0.19.0',
                     'future>=0.16.0',
                     'scikit-learn>=0.17.0',
                     'scikit-image>=0.12.3',
                     'shapely>=1.5.17.post1',
                     'tifffile>=0.10.0',
                     'multiprocessing>=2.6.2.1']
extras_require = {}

# Notebook dependencies for plotting
extras_require['plotting'] = (['holoviews>=1.8.2'])

# Dev dependencies
extras_require['dev'] = (['pytest>=2.8.1',
                          'pytest-cov',
                          'pytest-flake8'])

# Everything including cyordereddict (optimization) and nosetests
extras_require['all'] = (extras_require['plotting']
                         + extras_require['dev'])


class PyTest(TestCommand):

    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = []
        self.test_suite = True

    def run_tests(self):
        import pytest
        pytest.main(self.test_args)


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name=NAME,
    install_requires=install_requires,
    extras_require=extras_require,
    version="0.5.0",
    author="Sander Keemink & Scott Lowe & Nathalie Rochefort",
    author_email="swkeemink@scimail.eu",
    description="A Python Library estimating somatic signals in 2-photon data",
    url="https://github.com/rochefort-lab/fissa",
    download_url="NA",
    package_dir={NAME: "./fissa"},
    packages=[NAME],
    license="GNU",
    long_description=read('README.md'),
    classifiers=[
        "Natural Language :: English",
        "Programming Language :: Python",
        "Topic :: Scientific/Engineering"
    ],
    cmdclass={'test': PyTest},
)
