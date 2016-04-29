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

def eps_l(x, A1, A2, out, handle):
    out.fill(0)    
    #tmp = garr.empty((A1[0].shape[1], x.shape[1]), dtype=x.dtype)
    #tmp2 = garr.empty((tmp.shape[0], A2[0].shape[1]), dtype=x.dtype)
    for s in xrange(len(A1)):
        tmp = cla.dot(A1[s], x, transa='C', handle=handle)
        tmp2 = cla.dot(tmp, A2[s], handle=handle)
        out += tmp2
        
    return out    

def eps_r(x, A1, A2, out, handle):
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

def cublasZgemm_dev(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc):
    """
    Matrix-matrix product for complex general matrix.

    """

    status = cb._libcublas.cublasZgemm_v2(handle,
                                       cb._CUBLAS_OP[transa],
                                       cb._CUBLAS_OP[transb], m, n, k,
                                       int(alpha),
                                       int(A), lda, int(B), ldb,
                                       int(beta),
                                       int(C), ldc)
    cb.cublasCheckStatus(status)

cb._libcublas.cublasSetPointerMode_v2.restype = int
cb._libcublas.cublasSetPointerMode_v2.argtypes = [cb._types.handle,
                                                  c_int]

def eps_r_noop_strm_dev(x, A1, A2, out, tmp, tmp2, ones, zeros, streams, handle):
    D = A1[0].shape[0]
    Dm1 = D

    cb._libcublas.cublasSetPointerMode_v2(handle, 1)
 
    for s in xrange(len(A1)):
        cb.cublasSetStream(handle, streams[s].handle)
        cublasZgemm_dev(handle, 'N', 'N', D, Dm1, D, ones[s].gpudata, x.gpudata, D, 
                       A1[s].gpudata, D, zeros[s].gpudata, tmp[s].gpudata, D)
        cublasZgemm_dev(handle, 'C', 'N', Dm1, Dm1, D, ones[s].gpudata, A2[s].gpudata, D, 
                       tmp[s].gpudata, D, zeros[s].gpudata, tmp2[s].gpudata, Dm1)
        
    for s in streams:
        s.synchronize()

    cb._libcublas.cublasSetPointerMode_v2(handle, 0)
        
    cb.cublasSetStream(handle, 0)
    out.fill(0)
    for s in xrange(len(A1)):
        #cb.cublasZgeam(handle, 'N', 'N', Dm1, Dm1, 0. if s == 0 else 1., out.gpudata, Dm1, 1., tmp2[s].gpudata, Dm1, out.gpudata, Dm1)
        cb.cublasZaxpy(handle, Dm1 * Dm1, 1., tmp2[s].gpudata, 1, out.gpudata, 1)
    	
    return out

def eps_r_noop_strm(x, A1, A2, out, tmp, tmp2, streams, handle):
    D = A1[0].shape[0]
    Dm1 = D

    for s in xrange(len(A1)):
        cb.cublasSetStream(handle, streams[s].handle)
        cb.cublasZgemm(handle, 'N', 'N', D, Dm1, D, 1., x.gpudata, D, 
                       A1[s].gpudata, D, 0., tmp[s].gpudata, D)
        cb.cublasZgemm(handle, 'C', 'N', Dm1, Dm1, D, 1., A2[s].gpudata, D, 
                       tmp[s].gpudata, D, 0., tmp2[s].gpudata, Dm1)
        
    for s in streams:
        s.synchronize()

    cb.cublasSetStream(handle, 0)
    out.fill(0)
    for s in xrange(len(A1)):
        #cb.cublasZgeam(handle, 'N', 'N', Dm1, Dm1, 0. if s == 0 else 1., out.gpudata, Dm1, 1., tmp2[s].gpudata, Dm1, out.gpudata, Dm1)
        cb.cublasZaxpy(handle, Dm1 * Dm1, 1., tmp2[s].gpudata, 1, out.gpudata, 1)
    	
    return out
    
def get_batch_ptrs(arrs):
    return garr.to_gpu(sp.array(map(lambda x: int(x.gpudata), arrs)))
    
def eps_r_noop_batch(x_ptrs, A1_ptrs, A2_ptrs, out, tmp_ptrs, tmp2_ptrs, tmp2, handle):
    D = out.shape[0]
    Dm1 = D
    d = len(tmp2)
    
    cb.cublasZgemmBatched(handle, 'N', 'N', D, Dm1, D, 1., x_ptrs.gpudata, D, 
                          A1_ptrs.gpudata, D, 0., tmp_ptrs.gpudata, D, d)
    cb.cublasZgemmBatched(handle, 'C', 'N', Dm1, Dm1, D, 1., A2_ptrs.gpudata, D, 
                          tmp_ptrs.gpudata, D, 0., tmp2_ptrs.gpudata, Dm1, d)
                          
    out.fill(0)
    for s in xrange(d):
        cb.cublasZaxpy(handle, Dm1 * Dm1, 1., tmp2[s].gpudata, 1, out.gpudata, 1)
    	
    return out

def eps_l_noop_strm(x, A1, A2, out, tmp, tmp2, streams, handle):
    D = A1[0].shape[0]
    
    for s in xrange(len(A1)):
        cb.cublasSetStream(handle, streams[s].handle)
        cb.cublasZgemm(handle, 'N', 'C', D, D, D, 1., x.gpudata, D, 
                       A1[s].gpudata, D, 0., tmp[s].gpudata, D)
        cb.cublasZgemm(handle, 'N', 'N', D, D, D, 1., A2[s].gpudata, D, 
                       tmp[s].gpudata, D, 0., tmp2[s].gpudata, D)
        
    for s in streams:
        s.synchronize()
        
    cb.cublasSetStream(handle, 0)
    out.fill(0)
    for s in xrange(len(A1)):
        cb.cublasZaxpy(handle, D * D, 1., tmp2[s].gpudata, 1, out.gpudata, 1)
        
    return out
    
def eps_l_noop_strm_dev(x, A1, A2, out, tmp, tmp2, ones, zeros, streams, handle):
    D = A1[0].shape[0]

    cb._libcublas.cublasSetPointerMode_v2(handle, 1)
 
    for s in xrange(len(A1)):
        cb.cublasSetStream(handle, streams[s].handle)
        cublasZgemm_dev(handle, 'N', 'C', D, D, D, ones[s].gpudata, x.gpudata, D, 
                       A1[s].gpudata, D, zeros[s].gpudata, tmp[s].gpudata, D)
        cublasZgemm_dev(handle, 'N', 'N', D, D, D, ones[s].gpudata, A2[s].gpudata, D, 
                       tmp[s].gpudata, D, zeros[s].gpudata, tmp2[s].gpudata, D)
        
    for s in streams:
        s.synchronize()

    cb._libcublas.cublasSetPointerMode_v2(handle, 0)
        
    cb.cublasSetStream(handle, 0)
    out.fill(0)
    for s in xrange(len(A1)):
        cb.cublasZaxpy(handle, D * D, 1., tmp2[s].gpudata, 1, out.gpudata, 1)
        
    return out
 
def eps_l_noop_batch(x_ptrs, A1_ptrs, A2_ptrs, out, tmp_ptrs, tmp2_ptrs, tmp2, handle):
    D = out.shape[0]
    d = len(tmp2)
    
    cb.cublasZgemmBatched(handle, 'N', 'C', D, D, D, 1., x_ptrs.gpudata, D, 
                          A1_ptrs.gpudata, D, 0., tmp_ptrs.gpudata, D, d)
    cb.cublasZgemmBatched(handle, 'N', 'N', D, D, D, 1., A2_ptrs.gpudata, D, 
                          tmp_ptrs.gpudata, D, 0., tmp2_ptrs.gpudata, D, d)
                          
    out.fill(0)
    for s in xrange(d):
        cb.cublasZaxpy(handle, D * D, 1., tmp2[s].gpudata, 1, out.gpudata, 1)
    	
    return out
    
def calc_x(Kp1, C, Cm1, rp1, lm2, Am1, A, Ap1, lm1_s, lm1_si, r_s, r_si, Vsh):
    handle = cb.cublasCreate()
    
    if not rp1 is None:
        rp1 = garr.to_gpu(sp.asarray(rp1))
    if not lm2 is None:
        lm2 = garr.to_gpu(sp.asarray(lm2))
    
    lm1_s = garr.to_gpu(sp.asarray(lm1_s))
    lm1_si = garr.to_gpu(sp.asarray(lm1_si))
    
    r_s = garr.to_gpu(sp.asarray(r_s))
    r_si = garr.to_gpu(sp.asarray(r_si))
    
    A = map(garr.to_gpu, A)
    if not Am1 is None:
        Am1 = map(garr.to_gpu, Am1)
    if not Ap1 is None:
        Ap1 = map(garr.to_gpu, Ap1)
    
    Vsh = map(garr.to_gpu, Vsh)
    
    if not Cm1 is None:
        Cm1 = [[garr.to_gpu(Cm1[t, s]) for t in xrange(Cm1.shape[1])] for s in xrange(Cm1.shape[0])]
        
    if not (C is None and Kp1 is None):
        C = [[garr.to_gpu(C[s, t]) for t in xrange(C.shape[1])] for s in xrange(C.shape[0])]
        Kp1 = garr.to_gpu(Kp1)
    
    x = calc_x_G(Kp1, C, Cm1, rp1, lm2, Am1, A, Ap1, lm1_s, lm1_si, r_s, r_si, Vsh, handle=handle)
        
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
            x_subpart = eps_r(rp1, C[s], Ap1, x_subpart, handle) #~1st line
            
            x_subpart += cla.dot(A[s], Kp1, handle=handle) #~3rd line
    
            x_part += cla.dot(cla.dot(x_subpart, r_si, handle=handle), Vsh[s], handle=handle)

        x += cla.dot(lm1_s, x_part, handle=handle)

    if not lm2 is None:
        x_part.fill(0)
        for s in xrange(q):     #~2nd line
            x_subpart = eps_l(lm2, Am1, Cm1[s], x_subpart, handle)
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

class EOp_CUDA:
    def __init__(self, A1, A2, left, use_batch=False):
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
        self.A1G = [map(garr.to_gpu, A1k) for A1k in A1]
        self.A2G = [map(garr.to_gpu, A2k) for A2k in A2]
        self.tmp = map(garr.empty_like, self.A1G[0])
        self.tmp2 = map(garr.empty_like, self.A1G[0])
        
        self.use_batch = use_batch
        self.left = left
        
        self.D = A1[0].shape[1]        
        self.shape = (self.D**2, self.D**2)        
        self.dtype = sp.dtype(A1[0][0].dtype)
        
        self.calls = 0        
        
        self.out = garr.empty((self.D, self.D), dtype=self.dtype)        
        self.xG = garr.empty((self.D, self.D), dtype=self.dtype)

        if use_batch:
            self.A1G_p = map(get_batch_ptrs, self.A1G)
            self.A2G_p = map(get_batch_ptrs, self.A2G)
            self.tmp_p = get_batch_ptrs(self.tmp)
            self.tmp2_p = get_batch_ptrs(self.tmp2)
            self.xG_p = get_batch_ptrs([self.xG] * len(A1[0]))
            self.out_p = get_batch_ptrs([self.out] * len(A1[0]))
        else:
            self.A1G_p = None
            self.A2G_p = None
            self.tmp_p = None
            self.tmp2_p = None
            self.xG_p = None
            self.out_p = None

            self.ones = [garr.zeros((1), dtype=sp.complex128) for s in xrange(len(A1[0]))]
            self.ones = [one.fill(1) for one in self.ones]
            self.zeros = [garr.zeros((1), dtype=sp.complex128) for s in xrange(len(A1[0]))]
            
            self.streams = []
            for s in xrange(A1[0].shape[0]):
                self.streams.append(cd.Stream())
        
        self.hdl = cb.cublasCreate()

    def matvec(self, v):
        """Matrix-vector multiplication. 
        Result = Ev or vE (if self.left == True).
        """
        self.xG.set(v.reshape((self.D, self.D)))
        
        xnext = [self.xG, self.xG_p]
        x = [self.out, self.out_p]
        
        if self.left:
            for k in xrange(len(self.A1G)):
                x, xnext = xnext, x
                if self.use_batch:
                    eps_l_noop_batch(x[1], self.A1G_p[k], self.A2G_p[k], xnext[0], 
                                     self.tmp_p, self.tmp2_p, self.tmp2, self.hdl)
                else:
                    eps_l_noop_strm_dev(x[0], self.A1G[k], self.A2G[k], xnext[0], 
                                        self.tmp, self.tmp2, self.ones, self.zeros, 
                                        self.streams, self.hdl)
        else:
            for k in xrange(len(self.A2G) - 1, -1, -1):
                x, xnext = xnext, x
                if self.use_batch:
                    eps_r_noop_batch(x[1], self.A1G_p[k], self.A2G_p[k], xnext[0], 
                                     self.tmp_p, self.tmp2_p, self.tmp2, self.hdl)
                else:
                    eps_r_noop_strm_dev(x[0], self.A1G[k], self.A2G[k], xnext[0], 
                                        self.tmp, self.tmp2, self.ones, self.zeros, 
                                        self.streams, self.hdl)
            
        Ex = xnext[0]
        
        self.calls += 1
        
        return Ex.get().ravel()
        
    def close_cuda(self):
        if not self.hdl is None:
            cb.cublasDestroy(self.hdl)
            self.hdl = None
        
    def __del__(self):
        self.close_cuda()
        
class PinvOp_CUDA:
    def __init__(self, p, A1, A2, l=None, r=None, left=False, pseudo=True, use_batch=False):
        assert not (pseudo and (l is None or r is None)), 'For pseudo-inverse l and r must be set!'
        
        self.use_batch = use_batch
        self.p = p
        self.left = left
        self.pseudo = pseudo
        self.D = A1[0].shape[1]
        self.shape = (self.D**2, self.D**2)
        self.dtype = A1[0].dtype
        
        self.A1G = [map(garr.to_gpu, A1k) for A1k in A1]
        self.A2G = [map(garr.to_gpu, A2k) for A2k in A2]
        self.tmp = map(garr.empty_like, self.A1G[0])
        self.tmp2 = map(garr.empty_like, self.A1G[0])

        self.l = l
        self.r = r
        self.lG = garr.to_gpu(sp.asarray(l))
        self.rG = garr.to_gpu(sp.asarray(r))
        
        self.out = garr.empty((self.D, self.D), dtype=self.dtype)
        self.out2 = garr.empty((self.D, self.D), dtype=self.dtype)
        self.xG = garr.empty((self.D, self.D), dtype=self.dtype)
        
        if use_batch:
            self.A1G_p = map(get_batch_ptrs, self.A1G)
            self.A2G_p = map(get_batch_ptrs, self.A2G)
            self.tmp_p = get_batch_ptrs(self.tmp)
            self.tmp2_p = get_batch_ptrs(self.tmp2)
            self.xG_p = get_batch_ptrs([self.xG] * len(A1[0]))
            self.out_p = get_batch_ptrs([self.out] * len(A1[0]))
            self.out2_p = get_batch_ptrs([self.out2] * len(A1[0]))
        else:
            self.A1G_p = None
            self.A2G_p = None
            self.tmp_p = None
            self.tmp2_p = None
            self.xG_p = None
            self.out_p = None
            self.out2_p = None

            self.ones = [garr.zeros((1), dtype=sp.complex128) for s in xrange(len(A1[0]))]
            self.ones = [one.fill(1) for one in self.ones]
            self.zeros = [garr.zeros((1), dtype=sp.complex128) for s in xrange(len(A1[0]))]
            
            self.streams = []
            for s in xrange(A1[0].shape[0]):
                self.streams.append(cd.Stream())
        
        self.hdl = cb.cublasCreate()
    
    def matvec(self, v):
        x = v.reshape((self.D, self.D))
        
        self.xG.set(x)
        #self.out2.set(self.xG)
        #self.out2[:] = self.xG
        cd.memcpy_dtod(self.out2.gpudata, self.xG.gpudata, self.xG.nbytes)
        
        out = [self.out, self.out_p]
        out2 = [self.out2, self.out2_p]
        
        if self.left: #Multiplying from the left, but x is a col. vector, so use mat_dagger
            for k in xrange(len(self.A1G)):
                if self.use_batch:
                    eps_l_noop_batch(out2[1], self.A1G_p[k], self.A2G_p[k], out[0], 
                                     self.tmp_p, self.tmp2_p, self.tmp2, self.hdl)
                else:
                    eps_l_noop_strm_dev(out2[0], self.A1G[k], self.A2G[k], out[0],
                                        self.tmp, self.tmp2, self.ones, self.zeros,
                                        self.streams, self.hdl)
                out, out2 = out2, out
            Ehx = out2[0]
            
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
            for k in xrange(len(self.A2G) - 1, -1, -1):
                if self.use_batch:
                    eps_r_noop_batch(out2[1], self.A1G_p[k], self.A2G_p[k], out[0], 
                                     self.tmp_p, self.tmp2_p, self.tmp2, self.hdl)
                else:
                    eps_r_noop_strm_dev(out2[0], self.A1G[k], self.A2G[k], out[0],
                                        self.tmp, self.tmp2, self.ones, self.zeros,
                                        self.streams, self.hdl)
                out, out2 = out2, out
            Ex = out2[0]
            
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
