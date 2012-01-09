# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 17:29:27 2011

@author: Ashley Milsted

TODO:
    - Implement evaluation of the error due to restriction to bond dim.
    - Investigate whether a different gauge choice would reduce numerical inaccuracies.
        - The current choice gives us r_n = eye() and l_n containing
          the Schmidt spectrum.
        - Maybe the l's could be better conditioned?
    - Build more into TakeStep or add a new method that does Restore_ON_R etc. itself.
    - Add an algorithm for expanding the bond dimension.
    - Adaptive step size.

"""
import scipy as sp
import scipy.linalg as la
from scipy import *
import nullspace as ns
from matmul import *
        
class evoMPS_TDVP_Uniform:
    odr = 'C'
    typ = complex128
    
    def __init__(self, D, q):
        self.D = D
        self.q = q
        
        self.A = zeros((q, D, D), dtype=self.typ, order=self.odr)
        
        for s in xrange(self.q):
            self.A[s] = eye(D)
