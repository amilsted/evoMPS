# -*- coding: utf-8 -*-
"""
Created on Sat Jul  7 15:26:57 2012

@author: ash
"""
import matmul as m
import scipy as sp
import scipy.linalg as la
import pycuda.autoinit
import pycuda.gpuarray as garr
import pycuda.cumath as gma
import pycuda.driver as cd
import scikits.cuda.cublas as cb
import scikits.cuda.linalg as cla

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
    
def eps_r_2(x, A1, A2, out, handle):
    out.fill(0)    
    #tmp = garr.empty((A1[0].shape[0], x.shape[1]), dtype=A1[0].dtype)
    #tmp2 = garr.empty((tmp.shape[0], A2[0].shape[0]), dtype=A1[0].dtype)
    for s in xrange(len(A1)):
        tmp = cla.dot(A1[s], x, handle=handle)
        tmp2 = cla.dot(tmp, A2[s], transb='C', handle=handle)
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
    
def eps_l_2(x, A1, A2, out, handle):
    out.fill(0)    
    #tmp = garr.empty((A1[0].shape[1], x.shape[1]), dtype=x.dtype)
    #tmp2 = garr.empty((tmp.shape[0], A2[0].shape[1]), dtype=x.dtype)
    for s in xrange(len(A1)):
        tmp = cla.dot(A1[s], x, transa='C', handle=handle)
        tmp2 = cla.dot(tmp, A2[s], handle=handle)
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
    
def calc_x(Kp1, C, Cm1, rp1, lm2, Am1, A, Ap1, lm1_s, lm1_si, r_s, r_si, Vsh):
    D = A.shape[2]
    Dm1 = A.shape[1]
    q = A.shape[0]
    
    handle = cb.cublasCreate()
    
    if not rp1 is None:
        Grp1 = garr.to_gpu(sp.asarray(rp1))
    if not lm2 is None:
        Glm2 = garr.to_gpu(sp.asarray(lm2))
    
    Glm1_s = garr.to_gpu(sp.asarray(lm1_s))
    Glm1_si = garr.to_gpu(sp.asarray(lm1_si))
    
    Gr_s = garr.to_gpu(sp.asarray(r_s))
    Gr_si = garr.to_gpu(sp.asarray(r_si))
    
    GA = []
    GAm1 = []
    GAp1 = []
    for s in xrange(A.shape[0]):
        GA.append(garr.to_gpu(A[s]))
    if not Am1 is None:
        for s in xrange(Am1.shape[0]):
            GAm1.append(garr.to_gpu(Am1[s]))
    if not Ap1 is None:
        for s in xrange(Ap1.shape[0]):
            GAp1.append(garr.to_gpu(Ap1[s]))
    
    GVsh = []
    for s in xrange(Vsh.shape[0]):
        GVsh.append(garr.to_gpu(Vsh[s]))
    
    if not Cm1 is None:
        GCm1 = []
        for s in xrange(Cm1.shape[0]):
            GCm1s = []
            for t in xrange(Cm1.shape[1]):
                GCm1s.append(garr.to_gpu(Cm1[t, s]))
            GCm1.append(GCm1s)
    
    x = garr.zeros((Dm1, q * D - Dm1), dtype=A.dtype)
    x_part = garr.empty_like(x)
    x_subpart = garr.empty_like(GA[0])
    
    if not (C is None and Kp1 is None):
        GC = []
        for s in xrange(C.shape[0]):
            GCs = []
            for t in xrange(C.shape[1]):
                GCs.append(garr.to_gpu(C[s, t]))
            GC.append(GCs)
            
        GKp1 = garr.to_gpu(Kp1)
            
        assert (not C is None) and (not Kp1 is None)
        x_part.fill(0)
        for s in xrange(q):
            x_subpart = eps_r_2(Grp1, GC[s], GAp1, x_subpart, handle) #~1st line
            
            x_subpart += cla.dot(GA[s], GKp1, handle=handle) #~3rd line
    
            x_part += cla.dot(cla.dot(x_subpart, Gr_si, handle=handle), GVsh[s], handle=handle)

        x += cla.dot(Glm1_s, x_part, handle=handle)

    if not lm2 is None:
        x_part.fill(0)
        for s in xrange(q):     #~2nd line
            x_subpart = eps_l_2(Glm2, GAm1, GCm1[s], x_subpart, handle)
            x_part += cla.dot(x_subpart, cla.dot(Gr_s, GVsh[s], handle=handle), handle=handle)
        x += cla.dot(Glm1_si, x_part, handle=handle)
        
    cb.cublasDestroy(handle)
    
    return x.get()
    
def calc_x_G(Kp1, C, Cm1, rp1, lm2, Am1, A, Ap1, lm1_s, lm1_si, r_s, r_si, Vsh, handle=None):
    D = A[0].shape[1]
    Dm1 = A[0].shape[0]
    q = len(A)
    
    x = garr.zeros((Dm1, q * D - Dm1), dtype=A[0].dtype)
    x_part = garr.empty_like(x)
    x_subpart = garr.empty_like(A[0])
    
    if not (C is None and Kp1 is None):
        assert (not C is None) and (not Kp1 is None)
        x_part.fill(0)
        for s in xrange(q):
            x_subpart = eps_r_2(rp1, C[s], Ap1, x_subpart, handle) #~1st line
            
            x_subpart += cla.dot(A[s], Kp1, handle=handle) #~3rd line
    
            x_part += cla.dot(cla.dot(x_subpart, r_si, handle=handle), Vsh[s], handle=handle)

        x += cla.dot(lm1_s, x_part, handle=handle)

    if not lm2 is None:
        x_part.fill(0)
        for s in xrange(q):     #~2nd line
            x_subpart = eps_l_2(lm2, Am1, Cm1[s], x_subpart, handle)
            x_part += cla.dot(x_subpart, cla.dot(r_s, Vsh[s], handle=handle), handle=handle)
        x += cla.dot(lm1_si, x_part, handle=handle)
        
    return x
    
def calc_Bs(N, A, l, l_s, l_si, r, r_s, r_si, C, K, Vsh):
    GA = []
    for An in A:
        if An is None:
            GA.append(None)
        else:
            GAn = []
            for Ans in An:
                GAn.append(garr.to_gpu(Ans))
            GA.append(GAn)
    GA.append(None)
    
    Gl = []
    Gl_s = []
    Gl_si = []
    for n in xrange(len(l)):
        if l[n] is None:
            Gl.append(None)
            Gl_s.append(None)
            Gl_si.append(None)
        else:
            Gl.append(garr.to_gpu(sp.asarray(l[n]))) #TODO: Support special types...
            Gl_s.append(garr.to_gpu(sp.asarray(l_s[n])))
            Gl_si.append(garr.to_gpu(sp.asarray(l_si[n])))
    Gl.append(None)
    Gl_s.append(None)
    Gl_si.append(None)
        
    Gr = []
    Gr_s = []
    Gr_si = []
    for n in xrange(len(r)):
        if r[n] is None:
            Gr.append(None)
            Gr_s.append(None)
            Gr_si.append(None)
        else:
            Gr.append(garr.to_gpu(sp.asarray(r[n]))) #TODO: Support special types...
            Gr_s.append(garr.to_gpu(sp.asarray(r_s[n])))
            Gr_si.append(garr.to_gpu(sp.asarray(r_si[n])))
    Gr.append(None)
    Gr_s.append(None)
    Gr_si.append(None)

    GK = []
    for n in xrange(len(K)):
        if K[n] is None:
            GK.append(None)
        else:
            GK.append(garr.to_gpu(sp.asarray(K[n])))
    GK.append(None)
            
    GVsh = []
    for n in xrange(len(Vsh)):
        if Vsh[n] is None:
            GVsh.append(None)
        else:
            GVshn = []
            for s in xrange(Vsh[n].shape[0]):
                GVshn.append(garr.to_gpu(Vsh[n][s]))
            GVsh.append(GVshn)
    
    GC = []
    for n in xrange(len(C)):
        if C[n] is None:
            GC.append(None)
        else:
            GCn = []
            for s in xrange(C[n].shape[0]):
                GCns = []
                for t in xrange(C[n].shape[1]):
                    GCns.append(garr.to_gpu(C[n][s, t]))
                GCn.append(GCns)
            GC.append(GCn)
    GC.append(None)
    
    GCts = []
    for n in xrange(len(GC)):
        if GC[n] is None:
            GCts.append(None)
        else:
            GCtsn = []
            for t in xrange(len(GC[n])):
                GCtsns = []
                for s in xrange(len(GC[n][0])):
                    GCtsns.append(GC[n][s][t])
                GCtsn.append(GCtsns)
            GCts.append(GCtsn)
            
    hdl = cb.cublasCreate()
    
    num_strms = 10
    
    curr_stream = cb.cublasGetStream(hdl)
    
    sites_per_strm = max((N) / num_strms, 1)
    #print "sites_per_stream = ", sites_per_strm
    
    strms = []
    for i in xrange(N / sites_per_strm):
        strms.append(cd.Stream())
    
    GB = [None]
    for n in xrange(1, N + 1):
        if (n - 1) % sites_per_strm == 0:
            #print n
            #print "strm = ", (n - 1) / sites_per_strm
            cb.cublasSetStream(hdl, strms[(n - 1) / sites_per_strm].handle)
        if not Vsh[n] is None:
            if n > 1:
                Glm2 = Gl[n - 2]
            else:
                Glm2 = None
                
            Gx = calc_x_G(GK[n + 1], GC[n], GCts[n - 1], Gr[n + 1], Glm2, GA[n - 1], GA[n],
                          GA[n + 1], Gl_s[n - 1], Gl_si[n - 1], Gr_s[n], Gr_si[n], GVsh[n], handle=hdl)
            GBn = []
            for s in xrange(A[n].shape[0]):
                GBns = cla.dot(Gl_si[n - 1], Gx, handle=hdl) 
                GBns = cla.dot(GBns, GVsh[n][s], transb='C', handle=hdl)
                GBns = cla.dot(GBns, Gr_si[n], handle=hdl)
                GBn.append(GBns)
            GB.append(GBn)
        else:
            GB.append(None)
            
    cb.cublasSetStream(hdl, curr_stream)    
    cb.cublasDestroy(hdl)
    
    B = [None]
    for n in xrange(1, N + 1):
        if GB[n] is None:
            B.append(None)
        else:
            Bn = sp.empty_like(A[n])
            for s in xrange(A[n].shape[0]):
                Bn[s] = GB[n][s].get()
            B.append(Bn)
        
    return B

def eps_r_noop_strm(x, A1, A2, out, tmp, tmp2, streams, handle):
    D = A1[0].shape[0]
    Dm1 = D
    
    out.fill(0)
    
    for s in xrange(len(A1)):
        cb.cublasSetStream(handle, streams[s].handle)
        cb.cublasZgemm(handle, 'N', 'N', D, Dm1, D, 1., x.gpudata, D, 
                       A1[s].gpudata, D, 0., tmp[s].gpudata, D)
        cb.cublasZgemm(handle, 'C', 'N', Dm1, Dm1, D, 1., A2[s].gpudata, D, 
                       tmp[s].gpudata, D, 0., tmp2[s].gpudata, Dm1)
        
    for s in streams:
        s.synchronize()
        
    cb.cublasSetStream(handle, 0)
    for s in xrange(len(A1)):
        cb.cublasZaxpy(handle, Dm1 * Dm1, 1., tmp2[s].gpudata, 1, out.gpudata, 1)
        
    return out
    
def eps_l_noop_strm(x, A1, A2, out, tmp, tmp2, streams, handle):
    D = A1[0].shape[0]
    
    out.fill(0)
    
    for s in xrange(len(A1)):
        cb.cublasSetStream(handle, streams[s].handle)
        cb.cublasZgemm(handle, 'N', 'C', D, D, D, 1., x.gpudata, D, 
                       A1[s].gpudata, D, 0., tmp[s].gpudata, D)
        cb.cublasZgemm(handle, 'N', 'N', D, D, D, 1., A2[s].gpudata, D, 
                       tmp[s].gpudata, D, 0., tmp2[s].gpudata, D)
        
    for s in streams:
        s.synchronize()
        
    cb.cublasSetStream(handle, 0)
    for s in xrange(len(A1)):
        cb.cublasZaxpy(handle, D * D, 1., tmp2[s].gpudata, 1, out.gpudata, 1)
        
    return out
    
class EOp_CUDA:
    def __init__(self, A1, A2, left):
        """Creates a new LinearOperator interface to the superoperator E.
        
        This is a wrapper to be used with SciPy's sparse linear algebra routines.
        
        Parameters
        ----------
        A1 : ndarray
            Ket parameter tensor. 
        A2 : ndarray
            Bra parameter tensor.
        left : bool
            Whether to multiply with a vector to the left (or to the right).
        """
        self.A1G = map(garr.to_gpu, A1)
        self.A2G = map(garr.to_gpu, A2)
        self.tmp = map(garr.empty_like, self.A1G)
        self.tmp2 = map(garr.empty_like, self.A1G)
        
        self.D = A1.shape[1]
        
        self.shape = (self.D**2, self.D**2)
        
        self.dtype = sp.dtype(A1[0].dtype)
        
        self.out = garr.empty((self.D, self.D), dtype=self.dtype)        
        self.xG = garr.empty((self.D, self.D), dtype=self.dtype)
        
        self.calls = 0
        
        self.left = left
        
        self.hdl = cb.cublasCreate()
        
        self.streams = []
        for s in xrange(A1.shape[0]):
            self.streams.append(cd.Stream())
    
    def matvec(self, v):
        """Matrix-vector multiplication. 
        Result = Ev or vE (if self.left == True).
        """
        x = v.reshape((self.D, self.D))
        
        self.xG.set(x)
        
        if self.left:
            Ex = eps_l_noop_strm(self.xG, self.A1G, self.A2G, self.out, 
                                 self.tmp, self.tmp2, self.streams, self.hdl)
        else:
            Ex = eps_r_noop_strm(self.xG, self.A1G, self.A2G, self.out, 
                                 self.tmp, self.tmp2, self.streams, self.hdl)
        
        self.calls += 1
        
        return Ex.get().ravel()
        
    def close_cuda(self):
        if not self.hdl is None:
            cb.cublasDestroy(self.hdl)
            self.hdl = None
        
    def __del__(self):
        self.close_cuda()
        
class PinvOp_CUDA:
    def __init__(self, p, A1, A2, l=None, r=None, left=False, pseudo=True):
        assert not (pseudo and (l is None or r is None)), 'For pseudo-inverse l and r must be set!'
        
        self.A1G = map(garr.to_gpu, A1)
        self.A2G = map(garr.to_gpu, A2)
        self.tmp = map(garr.empty_like, self.A1G)
        self.tmp2 = map(garr.empty_like, self.A1G)
        
        self.l = l
        self.r = r
        
        self.lG = garr.to_gpu(sp.asarray(l))
        self.rG = garr.to_gpu(sp.asarray(r))
        self.p = p
        self.left = left
        self.pseudo = pseudo
        
        self.D = A1.shape[1]
        
        self.shape = (self.D**2, self.D**2)
        
        self.dtype = A1.dtype
        
        self.out = garr.empty((self.D, self.D), dtype=self.dtype)
        self.xG = garr.empty((self.D, self.D), dtype=self.dtype)
        
        self.hdl = cb.cublasCreate()
        
        self.streams = []
        for s in xrange(A1.shape[0]):
            self.streams.append(cd.Stream())
    
    def matvec(self, v):
        x = v.reshape((self.D, self.D))
        
        self.xG.set(x)
        
        
        if self.left: #Multiplying from the left, but x is a col. vector, so use mat_dagger
            Ehx = eps_l_noop_strm(self.xG, self.A1G, self.A2G, self.out, 
                                 self.tmp, self.tmp2, self.streams, self.hdl)
            if self.pseudo:
                QEQhx = Ehx - self.lG * m.adot(self.r, x)
                #res = QEQhx.mul_add(-sp.exp(-1.j * self.p), self.xG, 1)
                cb.cublasZaxpy(self.hdl, self.D**2, -sp.exp(-1.j * self.p), 
                               QEQhx.gpudata, 1, self.xG.gpudata, 1)
                res = self.xG
            else:
                #res = Ehx.mul_add(-sp.exp(-1.j * self.p), self.xG, 1)
                cb.cublasZaxpy(self.hdl, self.D**2, -sp.exp(-1.j * self.p), 
                               Ehx.gpudata, 1, self.xG.gpudata, 1)
                res = self.xG
        else:
            Ex = eps_r_noop_strm(self.xG, self.A1G, self.A2G, self.out, 
                                 self.tmp, self.tmp2, self.streams, self.hdl)
            if self.pseudo:
                QEQx = Ex - self.rG * m.adot(self.l, x)
                #res = QEQx.mul_add(-sp.exp(1.j * self.p), self.xG, 1)
                cb.cublasZaxpy(self.hdl, self.D**2, -sp.exp(1.j * self.p), 
                               QEQx.gpudata, 1, self.xG.gpudata, 1)
                res = self.xG
            else:
                #res = Ex.mul_add(-sp.exp(1.j * self.p), self.xG, 1)
                cb.cublasZaxpy(self.hdl, self.D**2, -sp.exp(1.j * self.p), 
                               Ex.gpudata, 1, self.xG.gpudata, 1)
                res = self.xG
        
        return res.get().ravel()
        
    def close_cuda(self):
        if not self.hdl is None:
            cb.cublasDestroy(self.hdl)
            self.hdl = None
        
    def __del__(self):
        self.close_cuda()

if __name__ == '__main__':
    test_calc_lr(512, 16)
    #test_nrm()
    #test_dot()
    #test_zgemm()