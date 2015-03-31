# -*- coding: utf-8 -*-
"""
Created on Mon Jan 12 11:15:35 2015

@author: ash
"""

import matmul as mm
import cython as cy
import numpy as np
cimport numpy as np

from libc.stdlib cimport malloc, free

cdef extern from "cblas.h":
    cdef enum CBLAS_ORDER:
        CblasRowMajor=101
        CblasColMajor=102

    cdef enum CBLAS_TRANSPOSE:
        CblasNoTrans=111
        CblasTrans=112
        CblasConjTrans=113
        CblasConjNoTrans=114

    void cblas_zgemm(CBLAS_ORDER Order, CBLAS_TRANSPOSE TransA,
                     CBLAS_TRANSPOSE TransB, int m, int n, int k,
    double *alpha, double *A, int lda, double *B, int ldb,
    double *beta, double *C, int ldc) nogil

    void cblas_zgemv(CBLAS_ORDER Order, CBLAS_TRANSPOSE TransA,
                     int m, int n, double *alpha, double *A,
                     int lda, double *X, int incx, double *beta,
                     double *Y, int incY) nogil

    void cblas_zscal(int N, double *alpha, double *X, int incX) nogil
    
#cdef extern from "capsule.h":
#    void* SMCapsule_AsVoidPtr(object ptr)
#
#from blas_lapack cimport dgemm_t, zgemm_t, ddot_t, dgemv_t, zgemv_t, zdotu_t
#
#cdef dgemm_t *dgemm = <dgemm_t*>SMCapsule_AsVoidPtr(scipy.linalg.blas.get_blas_funcs('gemm', dtype=float64)._cpointer)
#cdef zgemm_t *zgemm = <zgemm_t*>SMCapsule_AsVoidPtr(scipy.linalg.blas.get_blas_funcs('gemm', dtype=complex128)._cpointer)
#cdef ddot_t *ddot = <ddot_t*>SMCapsule_AsVoidPtr(scipy.linalg.blas.get_blas_funcs('dot', dtype=float64)._cpointer)
#cdef dgemv_t *dgemv = <dgemv_t*>SMCapsule_AsVoidPtr(scipy.linalg.blas.get_blas_funcs('gemv', dtype=float64)._cpointer)
#cdef zdotu_t *zdotu = <zdotu_t*>SMCapsule_AsVoidPtr(scipy.linalg.blas.get_blas_funcs('dotu', dtype=complex128)._cpointer)
#cdef zgemv_t *zgemv = <zgemv_t*>SMCapsule_AsVoidPtr(scipy.linalg.blas.get_blas_funcs('gemv', dtype=complex128)._cpointer)

ctypedef np.complex128_t cplx
ctypedef np.complex128_t[::1] cmp1d
ctypedef np.complex128_t[:, ::1] cmp2d
ctypedef np.complex128_t[:, :, ::1] cmp3d
ctypedef np.complex128_t[:, :, :, ::1] cmp4d

cpdef np.ndarray eps_l_noop(x, ndcmp3d A1, ndcmp3d A2):
    cdef np.ndarray ndout = np.empty((A1.shape[2], A2.shape[2]), dtype=np.complex128)
    return eps_l_noop_inplace(x, A1, A2, ndout)

cpdef np.ndarray eps_l_noop_inplace(x, ndcmp3d A1, ndcmp3d A2, ndcmp2d ndout):
    ndout.fill(0)
    cdef cmp2d out = ndout
    cdef cmp2d A1Hx
    
    if not A1.flags['C_CONTIGUOUS']:
        A1 = np.ascontiguousarray(A1)
    if not A2.flags['C_CONTIGUOUS']:
        A2 = np.ascontiguousarray(A2)
    
    assert A1.shape[0] == A2.shape[0], "dimension mismatch"
    assert A1.shape[1] == x.shape[0], "dimension mismatch"
    assert x.shape[1] == A2.shape[1], "dimension mismatch"
    
    if isinstance(x, mm.eyemat):
        eps_l_noop_id(A1, A2, out)
    elif isinstance(x, mm.simple_diag_matrix):
        d = x.diag
        if not d.flags['C_CONTIGUOUS']:
            d = np.ascontiguousarray(d)
        A1Hx = np.zeros((A1.shape[2], x.shape[1]), dtype=np.complex128)
        eps_l_noop_diag(d, A1, A2, A1Hx, out)
    else:
        if not x.flags['C_CONTIGUOUS']:
            x = np.ascontiguousarray(x)
        A1Hx = np.zeros((A1.shape[2], x.shape[1]), dtype=np.complex128)
        eps_l_noop_dense(x, A1, A2, A1Hx, out)
        
    return ndout

cpdef np.ndarray eps_r_noop(x, ndcmp3d A1, ndcmp3d A2):
    cdef np.ndarray ndout = np.empty((A1.shape[1], A2.shape[1]), dtype=np.complex128)
    return eps_r_noop_inplace(x, A1, A2, ndout)
        
cpdef np.ndarray eps_r_noop_inplace(x, ndcmp3d A1, ndcmp3d A2, ndcmp2d ndout):
    ndout.fill(0)
    cdef cmp2d out = ndout
    cdef cmp2d A1x
    
    if not A1.flags['C_CONTIGUOUS']:
        A1 = np.ascontiguousarray(A1)
    if not A2.flags['C_CONTIGUOUS']:
        A2 = np.ascontiguousarray(A2)
    
    assert A1.shape[0] == A2.shape[0], "dimension mismatch"
    assert A1.shape[2] == x.shape[0], "dimension mismatch"
    assert x.shape[1] == A2.shape[2], "dimension mismatch"
    
    if isinstance(x, mm.eyemat):
        eps_r_noop_id(A1, A2, out)
    elif isinstance(x, mm.simple_diag_matrix):
        d = x.diag
        if not d.flags['C_CONTIGUOUS']:
            d = np.ascontiguousarray(d)
        A1x = np.zeros((A1.shape[1], x.shape[1]), dtype=np.complex128)
        eps_r_noop_diag(d, A1, A2, A1x, out)
    else:
        if not x.flags['C_CONTIGUOUS']:
            x = np.ascontiguousarray(x)
        A1x = np.zeros((A1.shape[1], x.shape[1]), dtype=np.complex128)
        eps_r_noop_dense(x, A1, A2, A1x, out)

    return ndout
    
@cy.boundscheck(False)
@cy.wraparound(False)
cdef void gdmm(bint HX, cmp2d X, cmp1d Y, cmp2d out) nogil:
    cdef int i, j
    if HX:
        if (not out.shape[0] == X.shape[1] or not out.shape[1] == X.shape[0]
            or not Y.shape[0] == out.shape[1]):
                with gil:
                    raise ValueError("gdmm: Invalid array dimensions!")
        for i in xrange(X.shape[0]):
            for j in xrange(X.shape[1]):
                out[j, i].real = X[i, j].real
                out[j, i].imag = -X[i, j].imag
                out[j, i] = out[j, i] * Y[i]
    else:
        if (not out.shape[0] == X.shape[0] or not out.shape[1] == X.shape[1]
            or not Y.shape[0] == out.shape[1]):
                with gil:
                    raise ValueError("gdmm: Invalid array dimensions!")
        for i in xrange(X.shape[0]):
            for j in xrange(X.shape[1]):
                out[i, j] = X[i, j] * Y[j]

@cy.boundscheck(False)
@cy.wraparound(False)
cdef void eps_l_noop_diag(cmp1d x, cmp3d A1, cmp3d A2, cmp2d A1Hx, cmp2d out) nogil:
    cdef int s, M, N, K
    cdef complex alpha = 1
    cdef complex beta = 0
    
    M = A1Hx.shape[0]
    N = A2.shape[2]
    K = A1Hx.shape[1]
    for s in xrange(A1.shape[0]):
        gdmm(True, A1[s, :, :], x, A1Hx)

        cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K,
                    <double *>&alpha, <double *>&A1Hx[0, 0], K,
                    <double *>&A2[s, 0, 0], N, <double *>&alpha, #Use gemm to sum up result! 
                    <double *>&out[0, 0], N)
                        
@cy.boundscheck(False)
@cy.wraparound(False)
cdef void eps_r_noop_diag(cmp1d x, cmp3d A1, cmp3d A2, cmp2d A1x, cmp2d out) nogil:
    cdef int s, M, N, K, L
    cdef complex alpha = 1
    cdef complex beta = 0
    M = A1x.shape[0]
    K = A1x.shape[1]
    N = A2.shape[1]
    L = A2.shape[2]

    for s in xrange(A1.shape[0]):
        gdmm(False, A1[s, :, :], x, A1x)

        cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasConjTrans, M, N, K,
                    <double *>&alpha, <double *>&A1x[0, 0], K,
                    <double *>&A2[s, 0, 0], L, <double *>&alpha, #Use gemm to sum up result! 
                    <double *>&out[0, 0], N)

@cy.boundscheck(False)
@cy.wraparound(False)
cdef void eps_l_noop_dense(cmp2d x, cmp3d A1, cmp3d A2, cmp2d A1Hx, cmp2d out) nogil:
    cdef int s
    cdef complex alpha = 1
    cdef complex beta = 0
    
    for s in xrange(A1.shape[0]):
        cblas_zgemm(CblasRowMajor, CblasConjTrans, CblasNoTrans, 
                    A1.shape[2], x.shape[1], A1.shape[1],
                    <double *>&alpha, <double *>&A1[s, 0, 0], A1.shape[2],
                    <double *>&x[0, 0], x.shape[1], <double *>&beta,
                    <double *>&A1Hx[0, 0], A1Hx.shape[1])

        #M = A1Hx.shape[0]
        #N = A2.shape[2]
        #K = A1Hx.shape[1]
        cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
                    A1Hx.shape[0], A2.shape[2], A1Hx.shape[1],
                    <double *>&alpha, <double *>&A1Hx[0, 0], A1Hx.shape[1],
                    <double *>&A2[s, 0, 0], A2.shape[2], <double *>&alpha, #Use gemm to sum up result! 
                    <double *>&out[0, 0], out.shape[1])

@cy.boundscheck(False)
@cy.wraparound(False)
cdef void eps_r_noop_dense(cmp2d x, cmp3d A1, cmp3d A2, cmp2d A1x, cmp2d out) nogil:
    cdef int s
    cdef complex alpha = 1
    cdef complex beta = 0

    for s in xrange(A1.shape[0]):
        cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
                    A1.shape[1], x.shape[1], A1.shape[2],
                    <double *>&alpha, <double *>&A1[s, 0, 0], A1.shape[2],
                    <double *>&x[0, 0], x.shape[1], <double *>&beta,
                    <double *>&A1x[0, 0], A1x.shape[1])

        cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasConjTrans, 
                    A1x.shape[0], A2.shape[1], A1x.shape[1],
                    <double *>&alpha, <double *>&A1x[0, 0], A1x.shape[1],
                    <double *>&A2[s, 0, 0], A2.shape[2], <double *>&alpha, #Use gemm to sum up result! 
                    <double *>&out[0, 0], out.shape[1])

@cy.boundscheck(False)
@cy.wraparound(False)
cdef void eps_r_noop_id(cmp3d A1, cmp3d A2, cmp2d out) nogil:
    cdef int s
    cdef complex alpha = 1
    cdef complex beta = 1

    for s in xrange(A1.shape[0]):
        cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasConjTrans, 
                    A1.shape[1], A2.shape[1], A1.shape[2],
                    <double *>&alpha, <double *>&A1[s, 0, 0], A1.shape[2],
                    <double *>&A2[s, 0, 0], A2.shape[2], <double *>&beta, #Use gemm to sum up result! 
                    <double *>&out[0, 0], out.shape[1])

@cy.boundscheck(False)
@cy.wraparound(False)
cdef void eps_l_noop_id(cmp3d A1, cmp3d A2, cmp2d out) nogil:
    cdef int s
    cdef complex alpha = 1
    cdef complex beta = 1
    
    for s in xrange(A1.shape[0]):
        cblas_zgemm(CblasRowMajor, CblasConjTrans, CblasNoTrans, 
                    A1.shape[2], A2.shape[2], A1.shape[1],
                    <double *>&alpha, <double *>&A1[s, 0, 0], A1.shape[2],
                    <double *>&A2[s, 0, 0], A2.shape[2], <double *>&beta, #Use gemm to sum up result! 
                    <double *>&out[0, 0], out.shape[1])