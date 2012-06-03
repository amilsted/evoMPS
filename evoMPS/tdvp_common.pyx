# -*- coding: utf-8 -*-
"""
Created on Sat Jan 21 13:21:20 2012

@author: ash
"""

import cython as cy
import numpy as np
cimport numpy as np
cimport cpython.pycapsule as pc
from cython.parallel import prange

ctypedef np.complex128_t DTYPE_t

ctypedef DTYPE_t (*h_nn_func)(int s, int t, int u, int v) nogil

@cy.boundscheck(False)
@cy.wraparound(False)
cpdef calc_C(np.ndarray[DTYPE_t, ndim=4] AA,
             h_nn_cptr, np.ndarray[DTYPE_t, ndim=4] out):
    
    assert pc.PyCapsule_CheckExact(h_nn_cptr)
    
    cdef h_nn_func h_nn = <h_nn_func>pc.PyCapsule_GetPointer(h_nn_cptr, 'h_nn')
    
    cdef int q1 = AA.shape[0]
    cdef int q2 = AA.shape[1]
    
    cdef int D1 = AA.shape[2]
    cdef int D2 = AA.shape[3]
    
    if out is None:
        out = np.empty([q1, q2, D1, D2], dtype=AA.dtype)
    else:
        assert out.shape[0] == q1 and out.shape[1] == q2
        assert out.shape[2] == D1 and out.shape[3] == D2
        
    out.fill(0)
    
    cdef int s, t, u, v
    
    cdef DTYPE_t h
    
    for s in prange(q1, nogil=True):
        for t in range(q2):
            for u in range(q1):
                for v in range(q2):
                    h = h_nn(s, t, u, v)
                    if h != 0:
                        for i in range(D1): #avoid slicing!
                            for j in range(D2):
                                out[s, t, i, j] = out[s, t, i, j] + h * AA[u, v, i, j]
    
    return out