# -*- coding: utf-8 -*-
"""
@author: Ashley Milsted
"""
from __future__ import absolute_import, division, print_function

import scipy as sp
import scipy.optimize as opti
import scipy.linalg as la
import scipy.integrate as intg

def evolve(sys, t, dt=0.01, integ="euler",
    dynexp=True, D_max=128, dD_max=128, sv_tol=1e-14, cb_func=None,
    auto_truncate=True):
    num_steps = int(t / dt)
    for i in range(num_steps + 1):
        sys.update(auto_truncate=auto_truncate)
        B = sys.calc_B()

        if not cb_func is None:
            cb_func(sys, i)
            
        if i == num_steps:
            break

        if integ.lower() == "euler":
            sys.take_step(dt * 1.j, B=B, dynexp=dynexp, D_max=D_max, dD_max=dD_max, sv_tol=sv_tol)
        elif integ.lower() == "rk4":
            if dynexp and sys.maxD_is_less_than(D_max):
                dt_e = dt**(5. / 2.)  # Do a small Euler step with an error of the same order as the RK4 step
                dt_r = dt - dt_e
                sys.take_step(dt_e * 1.j, B=B, dynexp=dynexp, D_max=D_max, dD_max=dD_max, sv_tol=sv_tol)
                sys.update(restore_CF=auto_truncate, normalize=auto_truncate, auto_truncate=auto_truncate)
                sys.take_step_RK4(dt_r * 1.j)
            else:
                sys.take_step_RK4(dt * 1.j, B_i=B)
        
    return sys    

def find_ground(sys, tol=1E-6, h_init=0.04, max_itr=10000, 
                expand_to_D=None, expand_step=128, expand_tol=None, 
                cb_func=None, CG_start_tol=1E-2, **kwargs):
    j = 0
    if not cb_func is None:
        def cb_wrap(sys, i, **kwargs):
            cb_func(sys, j + i, **kwargs)
    else:
        cb_wrap = None

    if expand_to_D is None:
        expand_to_D = sys.D

    #Converging a low D state first means there is less noise after expansion
    if expand_tol is None:
        expand_tol = tol

    #Do a little imaginary time evolution to condition the state
    sys, dj, tau, dtau = opt_im_time(sys, tol=CG_start_tol, dtau_base=h_init, 
                                     max_itr=max_itr, cb_func=cb_wrap)
    j += dj

    while sys.D < expand_to_D:
        sys, dj, h0, B = opt_conj_grad(sys, tol=expand_tol, h_init=h_init, 
                                       max_itr=max_itr, return_B_grad=True, 
                                       cb_func=cb_wrap, **kwargs)
                                       
        j += dj
        
        sys, dj, tau, dtau = opt_im_time(sys, tol=100, dtau_base=h_init, 
                                         max_itr=max_itr, cb_func=cb_wrap,
                                         expand_to_D=min(sys.D + expand_step, expand_to_D), 
                                         expand_step=expand_step,
                                         expand_tol=100)

        j += dj        
        
    sys, dj, h0 = opt_conj_grad(sys, tol=tol, h_init=h_init, max_itr=max_itr, 
                                cb_func=cb_wrap, **kwargs)

    return sys

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

    #print "fac = %g" % fac, "dtau =", dtau

    return dtau, dh_pred_next

    
def opt_im_time(sys, tol=1E-6, dtau_base=0.04, dtau0=None, max_itr=10000, 
                cb_func=None, auto_trunc=True, auto_dtau=True,
                expand_to_D=None, expand_tol=1E-2, expand_step=2,
                expand_wait_steps=5):
    if expand_to_D is None:
        expand_to_D = sys.D
        
    if dtau0 is None:
        dtau0 = dtau_base
        
    i = -1
    dtau = dtau0
    tau = 0
    h = 0
    dh_pred = 0
    expand_wait = 0
    for i in range(max_itr + 1):
        sys.update(auto_truncate=auto_trunc)
        B = sys.calc_B()

        if not cb_func is None:
            cb_func(sys, i, tau=tau)

        eta = sys.eta
        dh = h - sys.h_expect.real
        h = sys.h_expect.real

        if i == max_itr or eta.real < tol and sys.D == expand_to_D and expand_wait == 0:
            break

        if auto_dtau:
            dtau, dh_pred = _im_time_autostep(dtau, dtau_base, eta, dh, dh_pred)
        
        #Note: expand_wait is important for stability. Expansion adds relatively
        #      small entries, and repeated expansion can lead to bad conditioning.
        dynexp = sys.D < expand_to_D and eta.real < expand_tol and expand_wait == 0
        
        sys.take_step(dtau, B=B, dynexp=dynexp, dD_max=expand_step, 
                      D_max=expand_to_D)
                      
        expand_wait = max(expand_wait - 1, 0)
        if dynexp:
            expand_wait = expand_wait_steps
        tau += dtau
        
    return sys, i + 1, tau, dtau

def opt_conj_grad(sys, tol=1E-6, h_init=0.01, h0_prev=None, reset_every=20, 
                  max_itr=10000, cb_func=None, return_B_grad=False):
    B_CG = None
    B_grad = None
    if not h0_prev is None:
        h = min(max(h0_prev, h_init * 5), h_init * 15)
    else:
        h = h_init * 15
    h0 = h
    BgdotBg = 0

    i = -1
    for i in range(max_itr + 1):
        reset = i % reset_every == 0 or i == max_itr
        sys.update(restore_CF=reset) #This simple CG works much better without restore_CF within a run.
                                     #With restore_CF, one would have to transform B_prev to match.
        
        if i == max_itr:
            B_grad = sys.calc_B()
        else:
            B_CG, B_grad, BgdotBg, h = sys.calc_B_CG(B_CG, B_grad, BgdotBg, h, 
                                                     tau_init=h_init, reset=reset)

        if not cb_func is None:
            cb_func(sys, i, h=h)

        if sys.eta < tol or i == max_itr:
            break

        sys.take_step(h, B=B_CG)

        if i == 0:
            h0 = h #Store the first (steepest descent) step
            h = h_init #second step is usually far shorter than the first (GD) step!
    
    if return_B_grad:
        return sys, i + 1, h0, B_grad
    else:
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
    print("f:", sys.h_expect.real)
    return sys.h_expect.real

def _opt_conj_grad2_g(x, sys, pars):
    sys.A = list((x[:pars] + 1.j * x[pars:]).reshape((sys.L, sys.q, sys.D, sys.D)))
    sys.update(restore_CF=False)
    B = sys.calc_B()
    B = sp.array(B).ravel()
    B = sp.concatenate((B.real * 2, B.imag * 2))
    print("g:", sys.eta, la.norm(B))
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
    print(sys.h_expect.real, sys.eta)

def evolve_scipy_ode(s, Tmax, max_steps, rtol=1e-6, atol=1e-8, callback=None, integrator=intg.RK45):
    def state_to_vec(s):
        return sp.hstack([s.A[n].ravel() for n in range(1, s.N+1)])

    def tensors_to_vec(A):
        return sp.hstack([A[n].ravel() for n in range(1, len(A))])

    def update_state_from_vec(s, y):
        ist = 0
        for n in range(1,s.N+1):
            iend = ist + s.A[n].size
            s.A[n][:] = sp.reshape(y[ist:iend], s.A[n].shape)
            ist = iend

    is_first_call_in_step = [True]
    def ifunc(t, y):
        #nonlocal is_first_call_in_step
        update_state_from_vec(s, y)
        if is_first_call_in_step[0]:
            s.update()
            B = s.calc_B(set_eta=True)
            is_first_call_in_step[0] = False
        else:
            s.update(restore_CF=False, normalize=False)
            B = s.calc_B(set_eta=False)

        for n in range(len(B)):
            if n > 0 and B[n] is None:
                B[n] = sp.zeros_like(s.A[n])

        Bvec = tensors_to_vec(B)
        Bvec = 1.j * Bvec #real time evolution!
        return Bvec

    print("Init:")
    igrtr = integrator(ifunc, 0.0, state_to_vec(s), Tmax, rtol=rtol, atol=atol)
    print("Init done: h_abs = %g" % igrtr.h_abs)
    
    for i in range(max_steps):
        s.update() #Note: This changes the variables (without changing the physical state).
                   #      Should be ok as along as the integrator does not share variable info between steps.
                   #      The integrator does store y_old, but only uses it for dense_output.

        if callback is not None:
            callback(s, igrtr, i)

        if igrtr.status != "running":
            print("Stopping with status:", igrtr.status)
            break

        is_first_call_in_step = True
        igrtr.step()

def evolve_scipy_ode_old(s, dt, steps, rtol=1e-6, atol=1e-8,
        callback=None, integrator="RK45", wrap_to_real=False, **kwargs):
    def state_to_vec(s):
        return sp.hstack([s.A[n].ravel() for n in range(1, s.N+1)])

    def tensors_to_vec(A):
        return sp.hstack([A[n].ravel() for n in range(1, len(A))])

    def update_state_from_vec(s, y):
        ist = 0
        for n in range(1,s.N+1):
            iend = ist + s.A[n].size
            s.A[n][:] = sp.reshape(y[ist:iend], s.A[n].shape)
            ist = iend

    is_first_call_in_step = [True]
    def ifunc(t, y):
        #nonlocal is_first_call_in_step
        update_state_from_vec(s, y)
        if is_first_call_in_step[0]:
            s.update()
            B = s.calc_B(set_eta=True)
            is_first_call_in_step[0] = False
        else:
            s.update(restore_CF=False, normalize=False)
            B = s.calc_B(set_eta=False)

        for n in range(len(B)):
            if n > 0 and B[n] is None:
                B[n] = sp.zeros_like(s.A[n])

        Bvec = tensors_to_vec(B)
        Bvec = 1.j * Bvec #real time evolution!
        return Bvec

    print("Init:")
    if wrap_to_real:
        igrtr = intg.complex_ode(ifunc)
    else:
        igrtr = intg.ode(ifunc)
    igrtr.set_integrator(integrator, **kwargs)
    igrtr.set_initial_value(state_to_vec(s))
    
    for i in range(steps):
        s.update() #Note: This changes the variables (without changing the physical state).
                   #      Should be ok as along as the integrator does not share variable info between steps.
                   #      The integrator does store y_old, but only uses it for dense_output.

        if callback is not None:
            callback(s, igrtr, i)

        if not igrtr.successful():
            print("Stopping with status:", igrtr.get_return_code())
            break

        is_first_call_in_step = True
        igrtr.integrate(igrtr.t + dt)