#!/usr/bin/env python


import os

from distutils.core import setup
from setuptools.command.test import test as TestCommand

from fissa import __meta__ as meta


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


install_requires = read('requirements.txt')

extras_require = {}

# Notebook dependencies for plotting
extras_require['plotting'] = read('requirements_plots.txt')

# Dependencies for generating documentation
extras_require['docs'] = read('requirements_docs.txt')

# Dev dependencies
extras_require['dev'] = read('requirements-dev.txt')

# Everything
extras_require['all'] = (
    extras_require['plotting']
    + extras_require['docs']
    + extras_require['dev']
)


class PyTest(TestCommand):

    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = []
        self.test_suite = True

    def run_tests(self):
        import pytest
        pytest.main(self.test_args)


setup(
    name = meta.__name__,
    install_requires = install_requires,
    extras_require = extras_require,
    version = meta.__version__,
    author = meta.__author__,
    author_email = meta.__author_email__,
    description = meta.__description__,
    url = meta.__url__,
    package_dir = {meta.__name__: os.path.join(".", meta.__path__)},
    packages = [meta.__name__],
    license = "GNU",
    long_description = read('README.rst'),
    classifiers = [
        "Natural Language :: English",
        "Programming Language :: Python",
        "Topic :: Scientific/Engineering"
    ],
    cmdclass = {'test': PyTest},
)
