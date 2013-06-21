# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 15:18:25 2013

@author: ash
"""

import numpy as np
import scipy as sp
import scipy.linalg as la
import scipy.sparse.linalg as las
import tdvp_common as tm
import matmul as m
import logging

log = logging.getLogger(__name__)

class PinvOp:    
    def __init__(self, p, A1, A2, l=None, r=None, left=False, pseudo=True):
        assert not (pseudo and (l is None or r is None)), 'For pseudo-inverse l and r must be set!'
        
        self.A1 = A1
        self.A2 = A2
        self.l = l
        self.r = r
        self.p = p
        self.left = left
        self.pseudo = pseudo
        
        self.D = A1.shape[1]
        
        self.shape = (self.D**2, self.D**2)
        
        self.dtype = A1.dtype
        
        self.out = np.empty((self.D, self.D), dtype=self.dtype)
    
    def matvec(self, v):
        x = v.reshape((self.D, self.D))
        
        if self.left: #Multiplying from the left, but x is a col. vector, so use mat_dagger
            Ehx = tm.eps_l_noop_inplace(x, self.A1, self.A2, self.out)
            if self.pseudo:
                QEQhx = Ehx - self.l * m.adot(self.r, x)
                res = x - sp.exp(-1.j * self.p) * QEQhx
            else:
                res = x - sp.exp(-1.j * self.p) * Ehx
        else:
            Ex = tm.eps_r_noop_inplace(x, self.A1, self.A2, self.out)
            if self.pseudo:
                QEQx = Ex - self.r * m.adot(self.l, x)
                res = x - sp.exp(1.j * self.p) * QEQx
            else:
                res = x - sp.exp(1.j * self.p) * Ex
        
        return res.ravel()
        
def pinv_1mE_brute(A1, A2, l, r, p=0, pseudo=True):
    D = A1.shape[1]
    E = np.zeros((D**2, D**2), dtype=A1.dtype)

    for s in xrange(A1.shape[0]):
        E += np.kron(A1[s], A2[s].conj())
    
    l = np.asarray(l)
    r = np.asarray(r)
    
    if pseudo:
        QEQ = E - r.reshape((D**2, 1)).dot(l.reshape((1, D**2)).conj())
    else:
        QEQ = E
    
    EyemE = np.eye(D**2, dtype=A1.dtype) - sp.exp(1.j * p) * QEQ
    
    return la.inv(EyemE)
    
def pinv_1mE_brute_LOP(A1, A2, l, r, p=0, pseudo=True, left=False):
    op = PinvOp(p, A1, A2, l, r, left=left, pseudo=pseudo)
    
    bop = sp.zeros(op.shape, dtype=op.dtype)
    for i in xrange(op.shape[1]):
        x = sp.zeros((op.shape[1]), dtype=op.dtype)
        x[i] = 1
        bop[:, i] = op.matvec(x)
    
    if left:
        bop = bop.conj().T
    
    return la.inv(bop)
    
def pinv_1mE(x, A1, A2, l, r, p=0, left=False, pseudo=True, tol=1E-6, maxiter=4000,
             out=None, sanity_checks=False, sc_data='', brute_check=False, solver=None):
    """Iteratively calculates the result of an inverse or pseudo-inverse of an 
    operator (eye - exp(1.j*p) * E) multiplied by a vector.
    
    In left mode, x is still taken to be a column vector and OP.conj().T.dot(x) 
    is performed.
    """
    D = A1.shape[1]
    
    if out is None:
        out = np.ones_like(A1[0])
    
    op = PinvOp(p, A1, A2, l, r, left=left, pseudo=pseudo)
    
    res = out.ravel()
    x = x.ravel()
    
    if solver is None:
        solver = las.bicgstab
        #bicgstab fails sometimes, e.g. with nearly rank-deficient stuff.
    
    res, info = solver(op, x, x0=res, maxiter=maxiter, tol=tol) #tol: norm( b - A*x ) / norm( b )
    
    if info != 0:
        log.warning("Warning: Did not converge on solution for ppinv! %s", sc_data)
    
    #Test
    if sanity_checks and x.shape[0] > 1:
        RHS_test = op.matvec(res)
        norm = la.norm(x)
        if norm == 0:
            d = 0
        else:
            d = la.norm(RHS_test - x) / norm
        #d = abs(RHS_test - x).sum() / abs(x).sum()
        if not d < tol:
            log.warning("Sanity check failed: Bad ppinv solution! Off by: %s %s", d, sc_data)
    
    res = res.reshape((D, D))
        
    if brute_check:
        pinvE = pinv_1mE_brute(A1, A2, l, r, p=p, pseudo=pseudo)
        
        if left:
            #res_brute = (x.reshape((1, D**2)).conj().dot(pinvE)).ravel().conj().reshape((D, D))
            res_brute = (pinvE.conj().T.dot(x)).reshape((D, D))
        else:
            res_brute = pinvE.dot(x).reshape((D, D))
        
        if not la.norm(res - res_brute) < la.norm(res) * tol * 10:
            log.warning("Brute check fail in calc_PPinv (left: %s): Bad brute check! Off by: %g %s", left, la.norm(res - res_brute) / la.norm(res), sc_data)
        
        if sanity_checks and x.shape[0] > 1:
            RHS_test = op.matvec(res_brute.ravel())
            norm = la.norm(x)
            if norm == 0:
                d2 = 0
            else:
                d2 = la.norm(RHS_test - x) / norm
            #d = abs(RHS_test - x).sum() / abs(x).sum()
            if not d2 < tol:
                log.warning("Sanity check failed: Bad ppinv brute solution! Off by: %s %s", d2, sc_data)
    
    out[:] = res
    
    return out