#!/usr/bin/env python


import os

from distutils.core import setup
from setuptools.command.test import test as TestCommand

NAME = 'fissa'


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


install_requires = read('requirements.txt')

extras_require = {}

# Notebook dependencies for plotting
extras_require['plotting'] = read('requirements_plots.txt')

# Dev dependencies
extras_require['dev'] = read('requirements-dev.txt')

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


setup(
    name=NAME,
    install_requires=install_requires,
    extras_require=extras_require,
    version="0.5.2.dev",
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
