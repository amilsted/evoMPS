# -*- coding: utf-8 -*-
"""
Created on Sun Jun  3 16:02:54 2012

@author: ash
"""
cimport numpy as np

ctypedef np.complex128_t DTYPE_t

ctypedef DTYPE_t (*h_nn_func)(int s, int t, int u, int v) nogil

cpdef calc_C(np.ndarray[DTYPE_t, ndim=4, mode="c"] AA, h_nn_cptr, 
             np.ndarray[DTYPE_t, ndim=4, mode="c"] out)