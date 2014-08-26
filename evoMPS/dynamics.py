# -*- coding: utf-8 -*-
"""
@author: Ashley Milsted
"""
import copy as cp
import scipy as sp
import scipy.optimize as opti
import scipy.linalg as la
import tdvp_common as tm
import matmul as m
from mps_uniform import EvoMPS_MPS_Uniform
import logging

def evolve(sys, t, dt=0.01, integ="euler", dynexp=True, maxD=None, cb_func=None):
    for i in xrange(int(t / dt)):
        sys.update()
        B = sys.calc_B()

        if not cb_func is None:
            cb_func(sys, i)

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

def find_ground(sys, tol=1E-6, h_init=0.04, use_CG=True, CG_gap=5, CG_max=5, 
                CG_tol=1000,
                max_itr=10000, gap_gd=False, cb_func=None):
    j = 0
    if not cb_func is None:
        def cb_wrap(sys, i, **kwargs):
            cb_func(sys, j + i, **kwargs)
    else:
        cb_wrap = None
        
    dtau = h_init
    h0 = None
    while j < max_itr:
        if gap_gd:
            sys, dj = opt_grad_descent(sys, tol=tol, h0=dtau, 
                                       max_itr=min(CG_gap, max_itr - j), 
                                       cb_func=cb_wrap)
        else:
            sys, dj, tau, dtau = opt_im_time(sys, tol=tol, dtau_base=h_init, 
                                             dtau0=dtau, 
                                             max_itr=min(CG_gap, max_itr - j), 
                                             cb_func=cb_wrap, auto_trunc=True)
        j += dj
        
        if use_CG:
#            if j / (CG_gap + CG_max) % 4 == 0:
#                h0 = None
            if sys.eta < CG_tol:
                CG_max_itr = min(CG_max, max_itr - j)
            else:
                CG_max_itr = min(1, max_itr - j)
            sys, dj, h0 = opt_conj_grad(sys, tol=tol, h_init=h_init, h0_prev=h0, 
                                        max_itr=CG_max_itr, 
                                        reset_every=CG_max, cb_func=cb_wrap)
            j += dj
            
        if sys.eta < tol: #Note: In case tol not reached, this value is out of date, but that's okay.
            break         
            
    return sys, j

def _im_time_autostep(dtau, dtau_base, eta, dh, dh_pred):
    #Adjust step size depending on eta vs. dh.
    #Normally, |dh| ~ dtau * eta**2. If we're in a tight spot in the effective
    #energy landscape, too large a step will raise the energy. |dh| >> dtau * eta**2.
    #Taking a step should change the energy as |dh| ~ dtau * eta**2, unless
    #the second derivative is large. 
    if abs(dh) < 5E-13:
        fac = 0 #If energy changes are approaching machine epsilon, stop adjusting.
    else:
        fac = (dh_pred / dh)

    dh_pred_next = eta**2 * dtau

    if fac == 0:
        if dh_pred != 0:
            if dh > 1E-13:
                dtau = max(dtau * 0.95, dtau_base * 0.1)
            else:
                dtau = min(dtau * 1.005, dtau_base * 1.0)
    elif fac < 0:
        dtau = dtau_base * 0.01
    #else:
    #    dtau = min(max(dtau * (1 - 0.01 * (fac - 2)), dtau0 * 0.01), dtau0 * 2)
    elif abs(fac) > 3:
        dtau = max(dtau * (1 - 0.01 * (fac - 3)), dtau_base * 0.01)
    elif 0.2 < fac < 2:
        dtau = min(dtau * (1 + 0.001 * sp.sqrt(2 - fac)), dtau_base * 2)

    print "fac = %g" % fac, "dtau =", dtau

    return dtau, dh_pred_next

    
def opt_im_time(sys, tol=1E-6, dtau_base=0.04, dtau0=0.04, max_itr=10000, cb_func=None, auto_trunc=True, auto_dtau=True):
    i = -1
    dtau = dtau0
    tau = 0
    h = 0
    dh_pred = 0
    for i in xrange(max_itr):
        sys.update(auto_truncate=auto_trunc)
        B = sys.calc_B()

        if not cb_func is None:
            cb_func(sys, i, tau=tau)

        eta = sys.eta
        dh = h - sys.h_expect.real
        h = sys.h_expect.real

        if auto_dtau:
            dtau, dh_pred = _im_time_autostep(dtau, dtau_base, eta, dh, dh_pred)
        
        if sys.eta < tol:
            break
        
        sys.take_step(dtau, B=B)
        tau += dtau
        
    return sys, i + 1, tau, dtau

def opt_conj_grad(sys, tol=1E-6, h_init=0.01, h0_prev=None, reset_every=20, 
                  max_itr=10000, cb_func=None):
    B = None
    B_grad = None
    if not h0_prev is None:
        h = min(max(h0_prev, h_init * 5), h_init * 15)
    else:
        h = h_init * 15
    h0 = h
    BgdotBg = 0
    g0 = 0
    #e = 0
    i = -1
    for i in xrange(max_itr):
        reset = i % reset_every == 0
        sys.update(restore_CF=reset) #This simple CG works much better without restore_CF within a run.
                                     #With restore_CF, one would have to transform B_prev to match.
        
        B, B_grad, BgdotBg, h, g0 = sys.calc_B_CG(B, BgdotBg, h_init, dtau_prev=h, 
                                                  g0_prev=g0, reset=reset, 
                                                  B_prev=B_grad, use_PR=True)

        if not cb_func is None:
            cb_func(sys, i, h=h)

        if sys.eta < tol:
            break

        sys.take_step(h, B=B)

        if i == 0:
            h0 = h #Store the first (steepest descent) step
            h = h_init #second step is usually far shorter than the first (GD) step!

    return sys, i + 1, h0

def opt_grad_descent(sys, tol=1E-6, h_init=0.01, max_itr=10000, im_gap=5, 
                     gap_every=1, cb_func=None):
    j = 0
    if not cb_func is None:
        def cb_wrap(sys, i, **kwargs):
            cb_func(sys, j + i, **kwargs)
    else:
        cb_wrap = None
        
    dtau = h_init
    h0 = None
    while j < max_itr:
        sys, dj, tau, dtau = opt_im_time(sys, tol=tol, dtau_base=h_init, 
                                         dtau0=dtau, 
                                         max_itr=min(im_gap, max_itr - j), 
                                         cb_func=cb_wrap, auto_trunc=True)
        j += dj
        
        sys, dj, h0 = opt_conj_grad(sys, tol=tol, h_init=h_init, h0_prev=h0, 
                                    max_itr=min(gap_every, max_itr - j), 
                                    reset_every=1, cb_func=cb_wrap)
        j += dj
            
        if sys.eta < tol: #Note: In case tol not reached, this value is out of date, but that's okay.
            break  
    return sys, j


def _opt_conj_grad2_f(x, sys, pars):
    sys.A = list((x[:pars] + 1.j * x[pars:]).reshape((sys.L, sys.q, sys.D, sys.D)))
    sys.update(restore_CF=False)
    print "f:", sys.h_expect.real
    return sys.h_expect.real

def _opt_conj_grad2_g(x, sys, pars):
    sys.A = list((x[:pars] + 1.j * x[pars:]).reshape((sys.L, sys.q, sys.D, sys.D)))
    sys.update(restore_CF=False)
    B = sys.calc_B()
    B = sp.array(B).ravel()
    B = sp.concatenate((B.real * 2, B.imag * 2))
    print "g:", sys.eta, la.norm(B)
    return B
    
def opt_conj_grad2(sys, tol=1E-6):
    pars = sys.L * sys.q * sys.D**2
    x0 = sp.array(sys.A).ravel() 
    x0 = sp.concatenate((x0.real, x0.imag)) #CG doesn't do nonanalytic stuff.
    #xo = opti.fmin_cg(_opt_conj_grad2_f, x0, 
    #                      fprime=_opt_conj_grad2_g,
    #                      args=(sys, pars), gtol=tol, norm=2)
    opts = {'gtol' : tol, 'disp': True, 'norm': 2, 'xtol': tol}
    res = opti.minimize(_opt_conj_grad2_f, x0, method='CG',
                        jac=_opt_conj_grad2_g,
                        args=(sys, pars), options=opts)
    xo = res.x
    sys.A = list((xo[:pars] + 1.j * xo[pars:]).reshape((sys.L, sys.q, sys.D, sys.D)))
    sys.update()
    sys.calc_B()
    print sys.h_expect.real, sys.eta
    