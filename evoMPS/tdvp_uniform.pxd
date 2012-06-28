# -*- coding: utf-8 -*-
# cython: profile=False
"""
Created on Sat Jun  2 18:29:31 2012

@author: ash
"""
import cython
cimport numpy as np
cimport tdvp_common as tc

#np.ndarray[np.complex128_t, ndim=3]

cdef class EvoMPS_TDVP_Uniform:
    cdef public int q
    cdef public int D

    #odr = 'C'
    #typ = sp.complex128
    cdef public float eps
    
    cdef public float itr_rtol
    cdef public float itr_atol
    
    cdef public object h_nn
    cdef public object h_nn_cptr
    
    cdef public bint symm_gauge
    
    cdef public bint sanity_checks
    cdef public int check_fac
    
    cdef public bint conv_l, conv_r
    cdef public int itr_l, itr_r
    
    cdef public object userdata
    
    cdef public np.complex128_t h
    cdef public np.complex128_t eta
    cdef public float S_hc
    
    cdef public object A, l, r, K, K_left
    cdef object AA, C, tmp
    cdef object l_sqrt, l_sqrt_i, r_sqrt, r_sqrt_i
    cdef object Vsh, x
    
    @cython.locals(s = cython.int, t = cython.int)
    cpdef calc_AA(self)
    
    @cython.locals(i = cython.int)
    cpdef _calc_lr(self, x, e, tmp, int max_itr=*, rtol=*, atol=*)