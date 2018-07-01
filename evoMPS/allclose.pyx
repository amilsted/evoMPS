# -*- coding: utf-8 -*-
"""
Created on Sat Jan 21 13:21:20 2012

@author: ash
"""
from __future__ import absolute_import, division, print_function

import cython as cy
import numpy as np
cimport numpy as np

ctypedef np.complex128_t DTYPE_t

from libc.math cimport sqrt

cdef double abs_cmp(complex X):
    return sqrt(X.real**2 + X.imag**2)

@cy.boundscheck(False)
@cy.wraparound(False)
cpdef allclose_mat(np.ndarray[DTYPE_t, ndim=2, mode="c"] A,
                   np.ndarray[DTYPE_t, ndim=2, mode="c"] B, 
                   double rtol, double atol):
    
    assert A.shape[0] == B.shape[0] and A.shape[1] == B.shape[1]
    
    cdef int D1 = A.shape[0]
    cdef int D2 = A.shape[1]    
    
    for i in range(D1):
        for j in range(D2):
            if abs_cmp(A[i, j] - B[i, j]) > atol + rtol * abs_cmp(B[i, j]):
                return False
    
    return True
    