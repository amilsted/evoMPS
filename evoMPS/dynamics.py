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

def find_ground(sys, tol=1E-6, dtau=0.04, use_CG=True, CG_gap=5, CG_max=5, 
                CG_tol=1000,
                max_itr=10000, gap_gd=False, cb_func=None):
    j = 0
    if not cb_func is None:
        def cb_wrap(sys, i, **kwargs):
            cb_func(sys, j + i, **kwargs)
    else:
        cb_wrap = None

    while j < max_itr:
        if gap_gd:
            sys, dj = opt_grad_descent(sys, tol=tol, h0=dtau, max_itr=min(CG_gap, max_itr - j), cb_func=cb_wrap)
        else:
            sys, dj, tau = opt_im_time(sys, tol=tol, dtau0=dtau, max_itr=min(CG_gap, max_itr - j), cb_func=cb_wrap, auto_trunc=True)
        j += dj
        
        if sys.eta < tol: #Check convergence after im_time steps.
            break                    #FIXME: In case tol not reached, this value is out of date.

        if use_CG:
            if sys.eta < CG_tol:
                CG_max_itr = min(CG_max, max_itr - j)
            else:
                CG_max_itr = min(1, max_itr - j)
            sys, dj = opt_conj_grad(sys, tol=tol, h0=dtau, max_itr=CG_max_itr, reset_every=CG_max, cb_func=cb_wrap)
            j += dj
            
    return sys, j

def _im_time_autostep(dtau, dtau0, eta, dh, dh_pred):
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
        if dh_pred != 0 and dh > 0:
            dtau = max(dtau * 0.95, dtau0 * 0.01)
        else:
            dtau = min(dtau, dtau0)
    elif fac < 0:
        dtau = dtau0 * 0.01
    #else:
    #    dtau = min(max(dtau * (1 - 0.01 * (fac - 2)), dtau0 * 0.01), dtau0 * 2)
    elif abs(fac) > 3:
        dtau = max(dtau * (1 - 0.01 * (fac - 3)), dtau0 * 0.01)
    elif 0.2 < fac < 2:
        dtau = min(dtau * (1 + 0.001 * sp.sqrt(2 - fac)), dtau0 * 2)

    print "fac = %g" % fac, "dtau =", dtau

    return dtau, dh_pred_next

    
def opt_im_time(sys, tol=1E-6, dtau0=0.04, max_itr=10000, cb_func=None, auto_trunc=True, auto_dtau=True):
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
            dtau, dh_pred = _im_time_autostep(dtau, dtau0, eta, dh, dh_pred)
        
        if sys.eta < tol:
            break
        
        sys.take_step(dtau, B=B)
        tau += dtau
        
    return sys, i + 1, tau

def opt_conj_grad(sys, tol=1E-6, h0=0.01, reset_every=10, max_itr=10000, cb_func=None):
    B = None
    B_grad = None
    h = h0 * 15
    BgdotBg = 0
    g0 = 0
    #e = 0
    i = -1
    for i in xrange(max_itr):
        sys.update(restore_CF=i == 0)
        
        #if e != 0 and not sp.isnan(sys.eta) and sys.eta > 1E-5:
        #    h = 2 * (sys.h_expect.real - e) / g0
        #e = sys.h_expect.real
        
        B, B_grad, BgdotBg, h, g0 = sys.calc_B_CG(B, BgdotBg, h0, dtau_prev=h, g0_prev=g0,
                                          reset=i % reset_every == 0, 
                                          B_prev=B_grad, use_PR=True,
                                          switch_threshold_eta=1E-5)

        if not cb_func is None:
            cb_func(sys, i, h=h)

        if sys.eta < tol:
            break

        sys.take_step(h, B=B)

        if i == 0:
           h = h0 #second step is usually far shorter than the first!        

    return sys, i + 1

def opt_grad_descent(sys, tol=1E-6, h0=0.01, max_itr=10000, cb_func=None):
    return opt_conj_grad(sys, tol=tol, h0=h0, max_itr=max_itr, reset_every=1, cb_func=cb_func)


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
    