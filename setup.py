#! /usr/bin/env python

from distutils.core import setup
import os

NAME = 'fissa'

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = NAME,
    version = "0.2",
    author = "Sander Keemink & Scott Lowe & Nathalie Rochefort",
    author_email = "swkeemink@scimail.eu",
    description = "A Python Library estimating somatic signals in 2-photon data",
    url = "NA",
    download_url = "NA",
    package_dir = {NAME: "./fissa"},
    packages=[NAME],
    license = "Closed source",
    long_description=read('README.md'),
    classifiers = [
        "License :: Closed source",
        "Natural Language :: English",
        "Programming Language :: Python",
        "Topic :: Scientific/Engineering"
        ]
    )
