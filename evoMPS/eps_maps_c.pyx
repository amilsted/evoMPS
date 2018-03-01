# -*- coding: utf-8 -*-
"""
Created on Mon Jan 12 11:15:35 2015

@author: ash
"""

import matmul as mm
import cython as cy
import numpy as np
cimport numpy as np
from cython.parallel cimport parallel, prange, threadid
cimport openmp

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
    
    void cblas_zaxpy(int n, double *a, double *x, int incx, double *y, int incy) nogil

cdef inline int int_min(int a, int b) nogil: return a if a <= b else b
cdef inline int int_max(int a, int b) nogil: return a if a > b else b

ctypedef np.complex128_t cplx
ctypedef np.complex128_t[::1] cmp1d
ctypedef np.complex128_t[:, ::1] cmp2d
ctypedef np.complex128_t[:, :, ::1] cmp3d
ctypedef np.complex128_t[:, :, :, ::1] cmp4d

cpdef np.ndarray eps_l_noop(x, ndcmp3d A1, ndcmp3d A2):
    cdef np.ndarray ndout = np.empty((A1.shape[2], A2.shape[2]), dtype=np.complex128)
    return eps_l_noop_inplace(x, A1, A2, ndout)

@cy.boundscheck(False)
@cy.wraparound(False)
cpdef np.ndarray eps_l_noop_inplace(x, ndcmp3d A1, ndcmp3d A2, ndcmp2d ndout):
    ndout.fill(0)
    cdef cmp2d out = ndout
    cdef cmp3d outs
    cdef int Nout = out.shape[0] * out.shape[1]
    cdef complex alpha = 1.0
    
    cdef int d = A1.shape[0]
    cdef int i, si, sf
    cdef int num_chunks = int_min(openmp.omp_get_max_threads(), d)
    cdef int lpc = int_max(1, d // num_chunks) #loops per chunk

    if not A1.flags['C_CONTIGUOUS']:
        A1 = np.ascontiguousarray(A1)
    if not A2.flags['C_CONTIGUOUS']:
        A2 = np.ascontiguousarray(A2)

    cdef cmp3d A1_ = A1
    cdef cmp3d A2_ = A2
    cdef cmp2d x_
    cdef cmp1d xd_
    cdef cmp3d A1Hx
    
    assert A1_.shape[0] == A2_.shape[0], "dimension mismatch"
    assert A1_.shape[1] == x.shape[0], "dimension mismatch"
    assert x.shape[1] == A2_.shape[1], "dimension mismatch"
    
    if num_chunks > 1: #only allocate extra memory if we spawn extra threads
        outs = np.zeros((num_chunks, out.shape[0], out.shape[1]), dtype=np.complex128) 
    
    if isinstance(x, mm.eyemat):
        if num_chunks == 1:
            eps_l_noop_id(A1_, A2_, out)
        else:
            for i in prange(num_chunks, nogil=True):
                si = i * lpc
                sf = d if i == num_chunks - 1 else (i + 1) * lpc
                eps_l_noop_id(A1_[si:sf,:,:], A2_[si:sf,:,:], outs[i,:,:])
    elif isinstance(x, mm.simple_diag_matrix):
        if x.diag.flags['C_CONTIGUOUS']:
            xd_ = x.diag
        else:
            xd_ = np.ascontiguousarray(x.diag)
        A1Hx = np.zeros((num_chunks, A1_.shape[2], xd_.shape[0]), dtype=np.complex128)
        if num_chunks == 1:
            eps_l_noop_diag(xd_, A1_, A2_, A1Hx[0,:,:], out)
        else:
            for i in prange(num_chunks, nogil=True):
                si = i * lpc
                sf = d if i == num_chunks - 1 else (i + 1) * lpc
                eps_l_noop_diag(xd_, A1_[si:sf,:,:], A2_[si:sf,:,:], A1Hx[i,:,:], outs[i,:,:])
    else:
        if x.flags['C_CONTIGUOUS']:
            x_ = x
        else:
            x_ = np.ascontiguousarray(x)
        A1Hx = np.zeros((num_chunks, A1_.shape[2], x_.shape[1]), dtype=np.complex128)
        
        if num_chunks == 1:
            eps_l_noop_dense(x_, A1_, A2_, A1Hx[0,:,:], out)
        else:
            outs = np.zeros((num_chunks, out.shape[0], out.shape[1]), dtype=np.complex128) 
            for i in prange(num_chunks, nogil=True):
                si = i * lpc
                sf = d if i == num_chunks - 1 else (i + 1) * lpc
                eps_l_noop_dense(x_, A1_[si:sf,:,:], A2_[si:sf,:,:], A1Hx[i,:,:], outs[i,:,:])
                    
    if num_chunks > 1: #add thread results up in case we used threads
        with nogil:
            for i in xrange(0, num_chunks):
                cblas_zaxpy(Nout, <double *>&alpha, <double *>&outs[i, 0, 0], 1, <double *>&out[0, 0], 1)

    return ndout

cpdef np.ndarray eps_r_noop(x, ndcmp3d A1, ndcmp3d A2):
    cdef np.ndarray ndout = np.empty((A1.shape[1], A2.shape[1]), dtype=np.complex128)
    return eps_r_noop_inplace(x, A1, A2, ndout)

@cy.boundscheck(False)
@cy.wraparound(False)
cpdef np.ndarray eps_r_noop_inplace(x, ndcmp3d A1, ndcmp3d A2, ndcmp2d ndout):
    ndout.fill(0)
    cdef cmp2d out = ndout
    cdef cmp3d outs
    cdef int Nout = out.shape[0] * out.shape[1]
    cdef complex alpha = 1.0
    
    cdef int d = A1.shape[0]
    cdef int i, si, sf
    cdef int num_chunks = int_min(openmp.omp_get_max_threads(), d)
    cdef int lpc = int_max(1, d // num_chunks) #loops per chunk
    
    if not A1.flags['C_CONTIGUOUS']:
        A1 = np.ascontiguousarray(A1)
    if not A2.flags['C_CONTIGUOUS']:
        A2 = np.ascontiguousarray(A2)
        
    cdef cmp3d A1_ = A1
    cdef cmp3d A2_ = A2
    cdef cmp2d x_
    cdef cmp1d xd_
    cdef cmp3d A1x
    
    assert A1.shape[0] == A2.shape[0], "dimension mismatch"
    assert A1.shape[2] == x.shape[0], "dimension mismatch"
    assert x.shape[1] == A2.shape[2], "dimension mismatch"
    
    if num_chunks > 1: #only allocate extra memory if we spawn extra threads
        outs = np.zeros((num_chunks, out.shape[0], out.shape[1]), dtype=np.complex128) 
    
    if isinstance(x, mm.eyemat):
        if num_chunks == 1:
            eps_r_noop_id(A1_, A2_, out)
        else:
            for i in prange(num_chunks, nogil=True):
                si = i * lpc
                sf = d if i == num_chunks - 1 else (i + 1) * lpc
                eps_r_noop_id(A1_[si:sf,:,:], A2_[si:sf,:,:], outs[i,:,:])
    elif isinstance(x, mm.simple_diag_matrix):
        if x.diag.flags['C_CONTIGUOUS']:
            xd_ = x.diag
        else:
            xd_ = np.ascontiguousarray(x.diag)
        A1x = np.zeros((num_chunks, A1_.shape[1], xd_.shape[0]), dtype=np.complex128)
        if num_chunks == 1:
            eps_r_noop_diag(xd_, A1_, A2_, A1x[0,:,:], out)
        else:
            for i in prange(num_chunks, nogil=True):
                si = i * lpc
                sf = d if i == num_chunks - 1 else (i + 1) * lpc
                eps_r_noop_diag(xd_, A1_[si:sf,:,:], A2_[si:sf,:,:], A1x[i,:,:], outs[i,:,:])
    else:
        if x.flags['C_CONTIGUOUS']:
            x_ = x
        else:
            x_ = np.ascontiguousarray(x)
        A1x = np.zeros((num_chunks, A1_.shape[1], x_.shape[1]), dtype=np.complex128)
        if num_chunks == 1:
            eps_r_noop_dense(x_, A1_, A2_, A1x[0,:,:], out)
        else:
            for i in prange(num_chunks, nogil=True):
                si = i * lpc
                sf = d if i == num_chunks - 1 else (i + 1) * lpc
                eps_r_noop_dense(x_, A1_[si:sf,:,:], A2_[si:sf,:,:], A1x[i,:,:], outs[i,:,:])
        
    if num_chunks > 1: #add thread results up in case we used threads
        with nogil:
            for i in xrange(0, num_chunks):
                cblas_zaxpy(Nout, <double *>&alpha, <double *>&outs[i, 0, 0], 1, <double *>&out[0, 0], 1)

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
