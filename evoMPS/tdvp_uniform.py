# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 17:29:27 2011

@author: Ashley Milsted

"""
import scipy as sp
import scipy.linalg as la
from scipy import *
import nullspace as ns
from matmul import *


def myAop(x):
    opres = zeros_like(A[0])
    for i in xrange(A.shape[0]):
        opres += matmul(None, A[i], x, H(A[i]))
        
    return x - opres  + r * np.trace(dot(l, x))

        
class evoMPS_TDVP_Uniform:
    odr = 'C'
    typ = complex128
    
    h_nn = None
    
    def __init__(self, D, q):
        self.D = D
        self.q = q
        
        self.A = zeros((q, D, D), dtype=self.typ, order=self.odr)
        
        self.C = empty((q, q, D, D), dtype=self.typ, order=self.odr)
        
        self.K = ones_like(A[0])
        
        self.l = empty_like(A[0])
        self.r = empty_like(A[0])
        
        for s in xrange(q):
            self.A[s] = eye(D)
    
    def Calc_C(self):
        self.C.fill(0)
        
        AA = empty_like(self.A[0])
        
        for (u, v) in ndindex(self.q, self.q):
            matmul(AA, self.A[u], self.A[v])
            for (s, t) in ndindex(self.q, self.q):
                C[s, t] += h_nn(n, s, t, u, v) * AA
    
    def Calc_K(self):
        Hr = empty_like(self.A[0])
        
        AAst = empty_like(self.A[0])
        
        for (s, t) in ndindex(self.q, self.q):
            matmul(AAst, self.A[s], self.A[t])
            for (u, v) in ndindex(self.q, self.q):
                Hr += h_nn(n, u, v, s, t) * matmul(None, AAst, self.r, H(self.A[v]), H(self.A[u]))
                
        QHr = Hr - self.r * sp.trace(matmul(None, self.l, Hr))
        
        EK = zeros_like(self.K)
        for (s, t) in ndindex(self.q, self.q):
            EK += matmul(None, self.A[s], self.K, H(self.A[t]))
            
        LHS = self.K - EK + self.r * np.trace(matmul(None, self.l, self.K))
        
        #Solve LHS = QHr