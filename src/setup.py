#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 14:34:35 2020

@author: yinjiang
"""
import numpy
from setuptools import setup
from Cython.Build import cythonize

setup(
      ext_modules = cythonize("myFunction.pyx"),include_path = [numpy.get_include()])