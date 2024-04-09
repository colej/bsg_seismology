from setuptools import setup, find_packages, Extension
import numpy as np


# Automatically include all Python packages
packages = find_packages()

setup(
    name='bsg',
    version='0.1.0',
    description='Analysis tools for Blue Super Giants observed with TESS.',
    author='Cole Johnston',
    author_email='colej@mpa-garching.mpg.de',
    url='https://github.com/colej/bsg_seismology',
    packages=packages,
    install_requires=[
        # List your project's dependencies here
    ],
    classifiers=[
        # Choose appropriate classifiers from:
        # https://pypi.org/classifiers/
    ],
)