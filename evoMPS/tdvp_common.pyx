# -*- coding: utf-8 -*-
"""
Created on Sat Jan 21 13:21:20 2012

@author: ash
"""

import scipy as sp
import numpy as np
cimport numpy as np

ctypedef np.complex128_t DTYPE_t

ctypedef DTYPE_t (*h_nn_func)(int s, int t, int u, int v)

def calc_C(np.ndarray[DTYPE_t, ndim=3] A1 not None,
             np.ndarray[DTYPE_t, ndim=3] A2 not None,
             long int h_nn_ptr, np.ndarray[DTYPE_t, ndim=4] out):
    
    cdef h_nn_func h_nn = <h_nn_func>h_nn_ptr
    
    cdef int q1 = A1.shape[0]
    cdef int q2 = A2.shape[0]
    
    if out is None:
        out = np.empty([q1, q2, A1.shape[1], A2.shape[2]], dtype=A1.dtype)
    else:
        assert out.shape[0] == q1 and out.shape[1] == q2
        assert out.shape[2] == A1.shape[1] and out.shape[3] == A2.shape[2]
        
    out.fill(0)
    
    cdef int s, t, u, v
    
    cdef np.ndarray[DTYPE_t, ndim=2] AA = np.empty_like(out[0, 0])
    
    cdef DTYPE_t h
    
    for u in range(q1):
        for v in range(q2):
            np.dot(A1[u], A2[v], out=AA)
            for s in range(q1):
                for t in range(q2):
                    h = h_nn(s, t, u, v)
                    if h != 0:
                        out[s, t] += h * AA
    
    return out