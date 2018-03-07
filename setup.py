#! /usr/bin/env python

import os

from distutils.core import setup
from setuptools.command.test import test as TestCommand

NAME = 'fissa'


install_requires = [	'numpy>=1.13.1',
                     'scipy>=0.19.1',
                     'future>=0.16.0',
                     'scikit-learn>=0.18.2',
                     'scikit-image>=0.13.0',
                     'shapely>=1.5.17.post1',
                     'tifffile>=0.12.1',
                     'multiprocessing>=2.6.2.1',
                     'pillow>=5.0.0']
extras_require = {}

# Notebook dependencies for plotting
extras_require['plotting'] = (['holoviews>=1.8.2',
                               'jupyter>=1.00'])

# Dev dependencies
extras_require['dev'] = (['pytest>=3.2.2',
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
    version="0.5.2",
    author="Sander Keemink & Scott Lowe & Nathalie Rochefort",
    author_email="swkeemink@scimail.eu",
    description="A Python Library estimating somatic signals in 2-photon data",
    url="https://github.com/rochefort-lab/fissa",
    download_url="https://github.com/rochefort-lab/fissa/archive/0.5.2.tar.gz",
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
