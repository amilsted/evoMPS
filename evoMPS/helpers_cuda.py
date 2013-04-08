# -*- coding: utf-8 -*-
"""
Created on Sat Jul  7 15:26:57 2012

@author: ash
"""

import scipy as sp
import scipy.linalg as la
import pycuda.driver as cuda
import pycuda.autoinit
import pycuda.gpuarray as garr
import pycuda.cumath as gma

from time import time as now #use real time, since cuda doesn't happen on the CPU!

from ctypes import *

class c_float2(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float)]

class c_double2(Structure):
    _fields_ = [("x", c_double),
                ("y", c_double)]

libcublas = cdll.LoadLibrary('libcublas.so')

# defines

CUBLAS_STATUS_SUCCESS           = 0x00000000
CUBLAS_STATUS_NOT_INITIALIZED   = 0x00000001
CUBLAS_STATUS_ALLOC_FAILED      = 0x00000003
CUBLAS_STATUS_INVALID_VALUE     = 0x00000007
CUBLAS_STATUS_MAPPING_ERROR     = 0x0000000B
CUBLAS_STATUS_EXECUTION_FAILED  = 0x0000000D
CUBLAS_STATUS_INTERNAL_ERROR    = 0x0000000E

CUBLAS_OP_N = 0x00000000
CUBLAS_OP_T = 0x00000001
CUBLAS_OP_C = 0x00000002

cublasStatus = c_uint
cublasOperation = c_uint

# Exceptions

class CublasError(Exception):
    pass

def checkCublasStatus(status):
    if status != CUBLAS_STATUS_SUCCESS:
        raise CublasError("Internal cuda error: %i" % status)

# Helper functions

# cublasInit
_cublasCreate = libcublas.cublasCreate_v2
_cublasCreate.restype = cublasStatus
_cublasCreate.argtypes = [POINTER(c_void_p)]

def cublasCreate():
    handle = c_void_p()
    status = _cublasCreate(byref(handle))
    checkCublasStatus(status)
    return handle

# cublasShutdown
_cublasDestroy = libcublas.cublasDestroy_v2
_cublasDestroy.restype = cublasStatus
_cublasDestroy.argtypes = [c_void_p]

def cublasDestroy(handle):
    status = _cublasDestroy(handle)
    checkCublasStatus(status)    

_sgemm = libcublas.cublasSgemm_v2
_sgemm.restype = cublasStatus
_sgemm.argtypes = [c_void_p, cublasOperation, cublasOperation, c_int, c_int, c_int,
                  POINTER(c_float), POINTER(c_float), c_int,
                  POINTER(c_float), c_int, POINTER(c_float), 
                  POINTER(c_float), c_int]                            
def sgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc):
    status = _sgemm(handle, transa, transb, m, n, k, byref(c_float(alpha)), 
                    A, lda, B, ldb, byref(c_float(beta)), C, ldc)
    checkCublasStatus(status)
    return C

_dgemm = libcublas.cublasDgemm_v2
_dgemm.restype = cublasStatus
_dgemm.argtypes = [c_void_p, cublasOperation, cublasOperation, c_int, c_int, c_int,
                  POINTER(c_double), POINTER(c_double), c_int,
                  POINTER(c_double), c_int, POINTER(c_double), 
                  POINTER(c_double), c_int]                            
def dgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc):
    status = _dgemm(handle, transa, transb, m, n, k, byref(c_double(alpha)), 
                    A, lda, B, ldb, byref(c_double(beta)), C, ldc)
    checkCublasStatus(status)
    return C

_cgemm = libcublas.cublasCgemm_v2
_cgemm.restype = cublasStatus
_cgemm.argtypes = [c_void_p, cublasOperation, cublasOperation, c_int, c_int, c_int,
                  POINTER(c_float2), POINTER(c_float2), c_int,
                  POINTER(c_float2), c_int, POINTER(c_float2), 
                  POINTER(c_float2), c_int]                            
def cgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc):
    status = _cgemm(handle, transa, transb, m, n, k, byref(c_float2(alpha)), 
                    A, lda, B, ldb, byref(c_float2(beta)), C, ldc)
    checkCublasStatus(status)
    return C
    
_zgemm = libcublas.cublasZgemm_v2
_zgemm.restype = cublasStatus
_zgemm.argtypes = [c_void_p, cublasOperation, cublasOperation, c_int, c_int, c_int,
                  POINTER(c_double2), POINTER(c_double2), c_int,
                  POINTER(c_double2), c_int, POINTER(c_double2), 
                  POINTER(c_double2), c_int]                            
def zgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc):
    status = _zgemm(handle, transa, transb, m, n, k, byref(c_double2(alpha)), 
                    A, lda, B, ldb, byref(c_double2(beta)), C, ldc)
    checkCublasStatus(status)
    return C
    
_znrm2 = libcublas.cublasDznrm2
_znrm2.restype = cublasStatus
_znrm2.argtypes = [c_void_p, c_int, POINTER(c_double2), c_int, POINTER(c_double)]
def znrm2(handle, n, x, incx, res):
    #res = sp.empty((1), dtype=sp.float64)
    #resg = garr.to_gpu(res)
    #resp = cast(int(resg.gpudata), POINTER(c_double))
    
    #res = c_double()
    status = _znrm2(handle, n, x, incx, res)
    checkCublasStatus(status)

    #resg.get(res)
    
    return res
    
def nrm2_z(x, res, cbHandle):
    x = x.ravel()
    xp = cast(int(x.gpudata), POINTER(c_double2))
    #resp = cast(int(res.gpudata), POINTER(c_double))
    res = cuda.pagelocked_zeros((1), dtype=sp.float64)
    print res.data
    resp = cast(int(res.data[0]), POINTER(c_double))
    #print x.strides
    znrm2(cbHandle, x.size, xp, x.strides[0], resp)
    return res
    
def dot_z(a, b, out, cbHandle, Hb=False):
    ap = cast(int(a.gpudata), POINTER(c_double2))
    bp = cast(int(b.gpudata), POINTER(c_double2))
    cp = cast(int(out.gpudata), POINTER(c_double2))
    
    if Hb:
        M = b.shape[0]
        K = b.shape[1]
    else:
        M = b.shape[1]
        K = b.shape[0]
        
    N = a.shape[0]
    
    if Hb:
        opB = CUBLAS_OP_C
    else:
        opB = CUBLAS_OP_N
    
    cp = zgemm(cbHandle, opB, CUBLAS_OP_N, M, N, K, 1., 
               bp, M, ap, K, 0., cp, M)
    
    return out
    
def dotnrm(x):
    x = x.ravel()
    return gma.sqrt(garr.dot(x.conj(), x).real)

def test_zgemm():
    K = 512    
    N = 756
    M = 1024    
    
    a = sp.asarray(sp.rand(M, K) + 1.j * sp.rand(M, K), dtype=sp.complex128)
    b = sp.asarray(sp.rand(K, N) + 1.j * sp.rand(K, N), dtype=sp.complex128)
    a_g = garr.to_gpu(a) #to_gpu() creates a C-ordered array by default, but cuBLAS expects Fortran-ordering
    b_g = garr.to_gpu(b)
    c_g = garr.empty((M, N), dtype=a.dtype)
    #cnrm_g = garr.empty((1), dtype=sp.float64)
    
    c = sp.zeros((M, N), dtype=a.dtype)
    
    handle = cublasCreate()
    
    dot_z(a_g, b_g, c_g, handle)
    
    #print nrm2_z(a_g, cnrm_g, handle)
    
    c_g.get(c)
    print dotnrm(c_g)
    print la.norm(c.ravel())
    
    #cnrm = sp.array([0.0])
    #cnrm_g.get(cnrm) 
    #print cnrm
    
    cublasDestroy(handle)
    
    print la.norm(c - sp.dot(a, b))
    print sp.allclose(c, sp.dot(a, b))

def eps_r(A, x, out, tmp, tmp2, handle):
    out.fill(0)
    for s in xrange(len(A)):
        dot_z(A[s], x, out=tmp, cbHandle=handle)
        dot_z(tmp, A[s], out=tmp2, Hb=True, cbHandle=handle)
        out += tmp2
        
    return out
    
def eps_r_cpu(A, x):
    out = sp.zeros_like(x)
    for s in xrange(A.shape[0]):
        tmp = sp.dot(A[s], x)
        tmp2 = sp.dot(tmp, A[s].conj().T)
        out += tmp2
        
    return out
    
def calc_r_cpu(A, x, rescale=True, max_itr=1000, atol=1E-14, rtol=1E-14):
    n = x.size #we will scale x so that stuff doesn't get too small
    x *= n / la.norm(x.ravel())
       
    x_ = x.copy()
    for i in xrange(max_itr):                
        x[:] = x_        
        x_ = eps_r_cpu(A, x) 
        #print "After"
        
        ev = la.norm(x_.ravel()) / la.norm(x.ravel())
        #print "ev: " + str(ev)
        ev_mag_inv = n / la.norm(x_.ravel())

        x_ *= ev_mag_inv
        
        diff = la.norm((x_ - x).ravel())
        #print "diff: " + str(diff)
        #print la.norm(x_)
        if diff < atol + rtol * n:
            #print (i, ev, ev_mag, norm((tmp - x).ravel())/n, atol, rtol)            
            x[:] = x_
            break
        #print i
        
    ev = abs(ev)
    
    if rescale and not abs(ev - 1) < atol:
        A *= 1 / sp.sqrt(ev)
    
    return x, i < max_itr - 1, i, ev
    
def calc_r(A, x, rescale=True, max_itr=1000, atol=1E-14, rtol=1E-14):
    n = x.size #we will scale x so that stuff doesn't get too small
    x *= n / la.norm(x.ravel())
    
    sval = sp.zeros((1), dtype=sp.float64)

    norm = dotnrm
    handle = cublasCreate()    
    
    GA = []
    for s in xrange(A.shape[0]):
        GA.append(garr.to_gpu(A[s]))
        
    Gx = garr.to_gpu(x)
    
    Gx_ = garr.to_gpu(x)
    
    Gtmp = garr.zeros_like(GA[0])
    Gtmp2 = garr.zeros_like(GA[0])
    
    #print "STARTING"
    for i in xrange(max_itr):                
        tmp = Gx
        Gx = Gx_
        Gx_ = tmp
        
        Gx_ = eps_r(GA, Gx, Gx_, Gtmp, Gtmp2, handle) 
                        
        norm(Gx).get(sval)
        nrm_x = sp.asscalar(sval)
        norm(Gx_).get(sval)
        nrm_x_ = sp.asscalar(sval)
        ev = nrm_x_ / nrm_x

        Gx_ *= n / nrm_x_
        
        norm(Gx_ - Gx).get(sval)
        #print "diff: " + str(sval)
        if sp.asscalar(sval) < atol + rtol * n:
            #print (i, ev, ev_mag, norm((tmp - x).ravel())/n, atol, rtol)
            Gx_.get(x)
            break
        #print i
        
    ev = abs(ev)
    
    if rescale and not abs(ev - 1) < atol:
        A *= 1 / sp.sqrt(ev)
    
    cublasDestroy(handle)
    
    return x, i < max_itr - 1, i, ev

def test_calc_r(D, d):
    A = sp.rand(d, D, D) + 1.j*sp.rand(d, D, D)
    x = sp.rand(D, D) + 1.j*sp.rand(D, D)
    
    A2 = A.copy()
    x2 = x.copy()
    
    print "running on GPU"
    res, conv, i, ev = calc_r(A, x)
    
    print "running on CPU"
    res2, conv, i2, ev2 = calc_r_cpu(A2, x2)
    
    print (la.norm(res.ravel() - res2.ravel()), la.norm(A.ravel() - A2.ravel()), abs(ev - ev2), i, i2)
    
    x_ = eps_r_cpu(A, res)
    print la.norm(x_ - res)
    
if __name__ == '__main__':
    test_calc_r(512, 16)