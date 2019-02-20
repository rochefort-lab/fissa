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
    name = meta['name'],
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
    classifiers = [
        "Natural Language :: English",
        "Programming Language :: Python",
        "Topic :: Scientific/Engineering"
    ],
    cmdclass = {'test': PyTest},
)
