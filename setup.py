#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os, codecs
from setuptools import setup, find_packages
from codecs import open


if os.path.exists('requirements.txt'):
    with open('requirements.txt', 'r') as fb:
        requirements = fb.readlines()
else:
    requirements = []


def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), "r", encoding="utf-8") as fp:
        return fp.read()


def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith("__version__"):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    raise RuntimeError("Unable to find version string.")

    
setup(
  name='wormholes',
  version=get_version("wormholes/__init__.py"),
  description="Code for 'Robustified ANNs Reveal Wormholes Between Human Category Percepts' (NeurIPS 2023)",
  long_description=open('README.md', encoding='utf-8').read(),
  long_description_content_type="text/markdown",
  author='Guy Gaziv',
  author_email='guyga@mit.edu',
  license='MIT',
  keywords='Vision, Object Recognition, Human, Primate, Ventral Stream, Adversarial Examples, Behavior Modulation, Behavioral Alignment',
  url='https://github.com/ggaziv/Wormholes',
  packages=['wormholes', 'scripts'],
  # install_requires=requirements,
  # tests_require=['pytest'],
  include_package_data=True,
  classifiers=[
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.9',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Natural Language :: English',
  ],
  test_suite='tests',
)
