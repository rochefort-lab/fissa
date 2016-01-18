#! /usr/bin/env python

import os

from distutils.core import setup
from setuptools.command.test import test as TestCommand

NAME = 'fissa'


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
    version="0.2.1",
    author="Sander Keemink & Scott Lowe & Nathalie Rochefort",
    author_email="swkeemink@scimail.eu",
    description="A Python Library estimating somatic signals in 2-photon data",
    url="https://github.com/rochefort-lab/fissa",
    download_url="NA",
    package_dir={NAME: "./fissa"},
    packages=[NAME],
    license="Closed source",
    long_description=read('README.md'),
    classifiers=[
        "License :: Closed source",
        "Natural Language :: English",
        "Programming Language :: Python",
        "Topic :: Scientific/Engineering"
    ],
    cmdclass={'test': PyTest},
)
