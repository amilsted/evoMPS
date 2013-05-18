# -*- coding: utf-8 -*-
"""
Created on Sat Jul  7 15:26:57 2012

@author: ash
"""

import scipy as sp
import scipy.linalg as la
import pycuda.autoinit
import pycuda.gpuarray as garr
import pycuda.cumath as gma
import pycuda.driver as cd
import scikits.cuda.cublas as cb

from time import time as now #use real time, since cuda doesn't happen on the CPU!

from ctypes import *
    
def dot_z(a, b, out, cbHandle, Hb=False, Ha=False):

    if Hb:
        M = b.shape[0]
        K = b.shape[1]
    else:
        M = b.shape[1]
        K = b.shape[0]
        
    if Ha:
        N = a.shape[1]
    else:
        N = a.shape[0]
    
    if Hb:
        opB = cb._CUBLAS_OP['C']
    else:
        opB = cb._CUBLAS_OP['N']
        
    if Ha:
        opA = cb._CUBLAS_OP['C']
    else:
        opA = cb._CUBLAS_OP['N']
        
    cb.cublasZgemm(cbHandle, opB, opA, M, N, K, 1., 
                   b.gpudata, M, a.gpudata, K, 0., out.gpudata, M)
    
    return out
    
def dotnrm(x):
    x = x.ravel()
    return gma.sqrt(garr.dot(x.conj(), x).real)
    
def znrm(x, handle):
    return cb.cublasDznrm2(handle, x.size, x.gpudata, 1).real

def test_zgemm():
    K = 512    
    N = 756
    M = 1024    
    
    a = sp.asarray(sp.rand(M, K) + 1.j * sp.rand(M, K), dtype=sp.complex128)
    b = sp.asarray(sp.rand(K, N) + 1.j * sp.rand(K, N), dtype=sp.complex128)
    a_g = garr.to_gpu(a) #to_gpu() creates a C-ordered array by default, but cuBLAS expects Fortran-ordering
    b_g = garr.to_gpu(b)
    c_g = garr.empty((M, N), dtype=a.dtype)
    
    c = sp.zeros((M, N), dtype=a.dtype)
    
    handle = cb.cublasCreate()
    
    dot_z(a_g, b_g, c_g, handle)
    
    #print nrm2_z(a_g, cnrm_g, handle)
    
    c_g.get(c)
    print cb.cublasDznrm2(handle, c_g.size, c_g.gpudata, 1).real
    print la.norm(c.ravel())
    
    #cnrm = sp.array([0.0])
    #cnrm_g.get(cnrm) 
    #print cnrm
    
    cb.cublasDestroy(handle)
    
    print la.norm(c - sp.dot(a, b))
    print sp.allclose(c, sp.dot(a, b))
    
def test_nrm():
    M = 512    
    
    a = sp.rand(M) + 1.j * sp.rand(M)
    a_g = garr.to_gpu(a)
    
    handle = cb.cublasCreate()
    
    nrm = sp.zeros((1))
    nrm_g = garr.to_gpu(nrm)
    x1 = nrm2_z(a_g, nrm_g, handle)
    
    x1.get(nrm)
    print sp.asscalar(nrm)
    
    x2 = dotnrm(c_g)
    
    x2.get(nrm)
    print sp.asscalar(nrm)
    
    x3 = la.norm(c.ravel())
    
    print x3
    
    cublasDestroy(handle)
    
    print la.norm(c - sp.dot(a, b))
    print sp.allclose(c, sp.dot(a, b))
    
def test_dot():
    M = 512    
    
    a = sp.rand(M) + 1.j * sp.rand(M)
    b = sp.rand(M) + 1.j * sp.rand(M)
    res = sp.zeros((2, 2), dtype=sp.complex128)
    
    a_g = garr.to_gpu(a)
    b_g = garr.to_gpu(b)
    res_g = garr.to_gpu(res)
    
    handle = cublasCreate()
    cublasSetPointerMode(handle, False)
    
    zdotu(a_g, b_g, res_g, handle)
    res_g.get(res)
    
    print res[0, 0]
    print (res[0, 0] - sp.inner(a, b))
    
    cublasDestroy(handle)

def eps_r(A, x, out, tmp, tmp2, handle):
    out.fill(0)    
    for s in xrange(len(A)):
        dot_z(A[s], x, out=tmp, cbHandle=handle)
        dot_z(tmp, A[s], out=tmp2, Hb=True, cbHandle=handle)
        out += tmp2
        
    return out

def get_streams(A):
    streams = []
    for s in xrange(len(A)):
        streams.append(cd.Stream())
    return streams
    
def sync_streams(streams):
    for s in xrange(len(streams)):
        streams.pop().synchronize()

def eps_r_parallel(A, x, out, handle, streams):
    out.fill(0)
    prev_stream = cb.cublasGetStream(handle)
    tmps = []
    tmp2s = []
    
    for s in xrange(len(A)):
        cb.cublasSetStream(handle, streams[s].handle)
        tmps.append(garr.empty_like(A[s]))
        dot_z(A[s], x, out=tmps[s], cbHandle=handle)
        tmp2s.append(garr.empty_like(A[s]))
        dot_z(tmps[s], A[s], out=tmp2s[s], Hb=True, cbHandle=handle)
        
        #out += tmp2
        out._axpbyz(1, tmp2s[s], 1, out, stream=streams[s])
    
    cb.cublasSetStream(handle, prev_stream)
    return out
    
def eps_l(A, x, out, tmp, tmp2, handle):
    out.fill(0)
    for s in xrange(len(A)):
        dot_z(A[s], x, out=tmp, Ha=True, cbHandle=handle)
        dot_z(tmp, A[s], out=tmp2, cbHandle=handle)
        out += tmp2
        
    return out
    
def eps_l_parallel(A, x, out, handle, streams):
    out.fill(0)
    prev_stream = cb.cublasGetStream(handle)
    tmps = []
    tmp2s = []
    for s in xrange(len(A)):
        cb.cublasSetStream(handle, streams[s].handle)
        tmps.append(garr.empty_like(A[s]))
        dot_z(A[s], x, out=tmps[s], Ha=True, cbHandle=handle)
        tmp2s.append(garr.empty_like(A[s]))
        dot_z(tmps[s], A[s], out=tmp2s[s], cbHandle=handle)
        out._axpbyz(1, tmp2s[s], 1, out, stream=streams[s])
    
    cb.cublasSetStream(handle, prev_stream)
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
    
def calc_lr(A, x, calc_l=True, rescale=True, max_itr=1000, atol=1E-14, rtol=1E-14):
    n = x.size #we will scale x so that stuff doesn't get too small
    x *= n / la.norm(x.ravel())

    handle = cb.cublasCreate()
    
    GA = []
    for s in xrange(A.shape[0]):
        GA.append(garr.to_gpu(A[s]))
        
    Gx = garr.to_gpu(x)
    
    Gx_ = garr.to_gpu(x)
    
    #Gtmp = garr.zeros_like(GA[0])
    #Gtmp2 = garr.zeros_like(GA[0])
    
    streams = get_streams(A)
    
    #print "STARTING"
    for i in xrange(max_itr):                
        tmp = Gx
        Gx = Gx_
        Gx_ = tmp
        
        if calc_l:
            Gx_ = eps_l_parallel(GA, Gx, Gx_, handle)
        else:
            #Gx_ = eps_r(GA, Gx, Gx_, Gtmp, Gtmp2, handle)
            Gx_ = eps_r_parallel(GA, Gx, Gx_, handle, streams)
                        
        nrm_x = znrm(Gx, handle) #this should cause a sync
        nrm_x_ = znrm(Gx_, handle)
        ev = nrm_x_ / nrm_x

        Gx_ *= n / nrm_x_
        
        #print "diff: " + str(sval)
        if znrm(Gx_ - Gx, handle) < atol + rtol * n:
            #print (i, ev, ev_mag, norm((tmp - x).ravel())/n, atol, rtol)
            Gx_.get(x)
            break
        #print i
        
    ev = abs(ev)
    
    if rescale and not abs(ev - 1) < atol:
        A *= 1 / sp.sqrt(ev)
    
    #sync_streams(streams)
    cb.cublasDestroy(handle)
    
    return x, i < max_itr - 1, i, ev

def test_calc_lr(D, d):
    A = sp.rand(d, D, D) + 1.j*sp.rand(d, D, D)
    x = sp.rand(D, D) + 1.j*sp.rand(D, D)
    
    A2 = A.copy()
    x2 = x.copy()
    
    print "running on GPU"
    res, conv, i, ev = calc_lr(A, x, calc_l=False)
    
    print "running on CPU"
    res2, conv, i2, ev2 = calc_r_cpu(A2, x2)
    
    print (la.norm(res.ravel() - res2.ravel()), la.norm(A.ravel() - A2.ravel()), abs(ev - ev2), i, i2)
    
    x_ = eps_r_cpu(A, res)
    print la.norm(x_ - res)
    
if __name__ == '__main__':
    test_calc_lr(512, 16)
    #test_nrm()
    #test_dot()
    #test_zgemm()