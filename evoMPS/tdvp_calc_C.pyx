# -*- coding: utf-8 -*-
"""
Created on Sat Jan 21 13:21:20 2012

@author: ash
"""
from __future__ import absolute_import, division, print_function

import cython as cy
import numpy as np
cimport numpy as np
cimport cpython.pycapsule as pc

@cy.boundscheck(False)
@cy.wraparound(False)
cpdef calc_C(np.ndarray[DTYPE_t, ndim=4, mode="c"] AA,
             h_nn_cptr, np.ndarray[DTYPE_t, ndim=4, mode="c"] out):
    
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
    
    cdef int i, j, s, t, u, v
    
    cdef DTYPE_t h
    
    cdef DTYPE_t [:,:,:,:] AA_view = AA
    cdef DTYPE_t [:,:,:,:] out_view = out
    
    with nogil:
        for s in range(q1):
            for t in range(q2):
                for u in range(q1):
                    for v in range(q2):
                        h = h_nn(s, t, u, v)
                        if h != 0:
                            for i in range(D1): 
                                for j in range(D2):
                                    out_view[s, t, i, j] = out_view[s, t, i, j] + h * AA_view[u, v, i, j]
        
    return out