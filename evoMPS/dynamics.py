# -*- coding: utf-8 -*-
"""
@author: Ashley Milsted
"""
import copy as cp
import scipy as sp
import tdvp_common as tm
import matmul as m
from mps_uniform import EvoMPS_MPS_Uniform
import logging

def evolve(sys, t, dt=0.01, integ="euler", dynexp=True, maxD=None):
    for i in xrange(max_itr):
        sys.update()
        B = sys.calc_B()

        if not cb_func is None:
            cb_func(i, sys)
        
        if sys.eta.sum() < tol:
            break

        if integ.lower() == "euler":
            sys.take_step(dt * 1.j, B=B)
        elif integ.lower() == "rk4":
            if dynexp and sys.D < maxD:
                dt_e = dt**(5. / 2.) #Do a small Euler step with an error of the same order as the RK4 step
                dt_r = dt - dt_e
                sys.take_step(dt_e * 1.j, B=B, dynexp=dynexp, maxD=maxD)
                sys.update(auto_truncate=True)
                sys.take_step_RK4(dt_r * 1.j)
            else:
                sys.take_step_RK4(dt * 1.j, B_i=B)
        
    return sys    

def find_ground(sys, tol=1E-6, dtau=0.04, use_CG=True, CG_gap=5, CG_max=10, max_itr=10000, cb_func=None):
    j = 0
    if not cb_func is None:
        def cb_wrap(i, sys):
            cb_func(j + i, sys)
    else:
        cb_wrap = None

    while j < max_itr:
        #sys, dj = opt_im_time(sys, tol=tol, dtau=dtau, max_itr=min(CG_gap, max_itr - j), cb_func=cb_wrap, auto_trunc=True)
        sys, dj = opt_grad_descent(sys, tol=tol, h0=dtau, max_itr=min(CG_gap, max_itr - j), cb_func=cb_wrap)
        j += dj
        
        if sys.eta.real.sum() < tol: #Check convergence after im_time steps
            break

        if use_CG:
            sys, dj = opt_conj_grad(sys, tol=tol, h0=dtau, max_itr=min(CG_max, max_itr - j), cb_func=cb_wrap)
            j += dj
            
    return sys, j

def opt_im_time(sys, tol=1E-6, dtau=0.04, max_itr=10000, cb_func=None, auto_trunc=True):
    for i in xrange(max_itr):
        sys.update(auto_truncate=auto_trunc)
        B = sys.calc_B()

        if not cb_func is None:
            cb_func(i, sys)
        
        if sys.eta.sum() < tol:
            break
        
        sys.take_step(dtau, B=B)
        
    return sys, i + 1

def opt_conj_grad(sys, tol=1E-6, h0=0.01, reset_every=10, max_itr=10000, cb_func=None):
    B = None
    h = h0
    eta = 0
    
    for i in xrange(max_itr):
        sys.update()
        B, B_grad, eta, h = sys.calc_B_CG(B, eta, h, reset=i % reset_every == 0)

        if not cb_func is None:
            cb_func(i, sys)

        if eta < tol:
            break

        sys.take_step(h, B=B)

    return sys, i + 1

def opt_grad_descent(sys, tol=1E-6, h0=0.01, max_itr=10000, cb_func=None):
    return opt_conj_grad(sys, tol=tol, h0=h0, max_itr=max_itr, reset_every=1, cb_func=cb_func)
