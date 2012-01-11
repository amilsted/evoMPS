# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 17:29:27 2011

@author: Ashley Milsted

WIP

"""
import scipy as sp
import scipy.linalg as la
from scipy import *
import nullspace as ns
from matmul import *
        
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
