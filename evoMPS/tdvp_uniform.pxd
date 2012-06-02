# -*- coding: utf-8 -*-
"""
Created on Sat Jun  2 18:29:31 2012

@author: ash
"""
import cython
cimport numpy as np

#np.ndarray[np.complex128_t, ndim=3]

cdef class evoMPS_TDVP_Uniform:
    cdef cython.int q
    cdef cython.int D
    
    @cython.locals(s = cython.int, t = cython.int)
    cpdef Calc_AA(self)
    
    @cython.locals(i = cython.int)
    cpdef _Calc_lr(self, x, e, tmp, int max_itr=*, rtol=*, atol=*)