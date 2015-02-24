# -*- coding: utf-8 -*-
"""
Created on Mon Jan 12 14:18:20 2015

@author: ash
"""

from distutils.core import setup
from distutils.extension import Extension

setup(
    name='evoMPS_c_alt',
    ext_modules = [Extension("evoMPS.eps_maps_c", ["evoMPS/eps_maps_c.c"], 
                             libraries=["cblas"])]
)