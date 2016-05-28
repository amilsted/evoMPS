#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
A demonstration of simulating open system dynamics by sampling
over randomized pure state trajectories.

This simulates the dynamics of an edge-driven XXZ chain,
which has an analytic solution for certain parameter combinations.
See: T. Prosen, Phys. Rev. Lett. 107, 137201 (2011)

@author: Ashley Milsted
"""
import math as ma
import scipy as sp
import evoMPS.tdvp_gen_diss as tdvp
import time
from multiprocessing import Pool, Queue, Manager
from multiprocessing.pool import ThreadPool
import copy

"""
First, we set up some global variables to be used as parameters.
"""
N = 10                       #The length of the finite spin chain.
bond_dim = 64                 #The maximum bond dimension

N_samp = 2                   #Number of samples

lam = 1.0
eps = 2.0

N_steps = 100000
res_every = 100
dt = 0.0001                     #Real-time step size

load_saved_ground = True      #Whether to load a saved ground state

auto_truncate = False         #Whether to reduce the bond-dimension if any Schmidt coefficients fall below a tolerance.
zero_tol = 0                  #Zero-tolerance for the Schmidt coefficients squared (right canonical form)

plot_results = True

sanity_checks = False         #Whether to perform additional (verbose) sanity checks

"""
Next, we define our Hamiltonian and some observables.
"""
Sx = sp.array([[0., 1.],
               [1., 0.]])
Sy = 1.j * sp.array([[0., -1.],
                     [1.,  0.]])
Sz = sp.array([[1.,  0.],
               [0., -1.]])

Sp = 0.5 * (Sx + 1.j * Sy)
Sm = 0.5 * (Sx - 1.j * Sy)
                     

def get_ham(N, lam):
    h = (2. * sp.kron(Sp, Sm) + 2. * sp.kron(Sm, Sp)
         + lam * sp.kron(Sz, Sz)).reshape(2, 2, 2, 2)
    return [None] + [h] * (N - 1)
    
def get_linds(N, eps):
    #Lindblad operators must have same range as Hamiltonian terms. In this case they are nearest-neighbour.
    Sp1 = (sp.kron(Sp, sp.eye(2))).reshape(2, 2, 2, 2)
    Sm2 = (sp.kron(sp.eye(2), Sm)).reshape(2, 2, 2, 2)
    
    L1 = (1, sp.sqrt(eps) * Sp1)
    L2 = (N-1, sp.sqrt(eps) * Sm2)
    
    return [L1, L2]

"""
The bond dimension for each site is given as a vector, length N + 1.
Here we set the bond dimension = bond_dim for all sites.
"""
D = [bond_dim] * (N + 1)

"""
The site Hilbert space dimension is also given as a vector, length N + 1.
Here, we set all sites to dimension = 2.
"""
q = [2] * (N + 1)

def get_full_state(s):
    psi = sp.zeros(tuple([2]*N), dtype=sp.complex128)
    
    for ind in sp.ndindex(psi.shape):
        A = 1.0
        for n in xrange(N, 0, -1):
            A = s.A[n][ind[n-1]].dot(A)
        #print A, psi[ind]
        psi[ind] = A[0,0]
    psi = psi.ravel()
    assert psi.shape[0] == 2**N
    print "norm", sp.vdot(psi, psi)
    return psi
    
def get_full_op(op):
    fop = sp.zeros((2**N, 2**N), dtype=sp.complex128)
    for n in xrange(1, min(len(op), N + 1)):
        if op[n] is None:
            continue
            
        n_sites = len(op[n].shape) / 2
        opn = op[n].reshape(2**n_sites, 2**n_sites)
        fop_n = sp.kron(sp.eye(2**(n - 1)), sp.kron(opn, sp.eye(2**(N - n - n_sites + 1))))
#        print n, n_sites, fop.shape, fop_n.shape, N - n - n_sites + 1
        assert fop.shape == fop_n.shape
        fop += fop_n
    return fop
    
def go_exact(load_from=None, resQ=None, pid=None, N_steps=100):
    sp.random.seed(pid)
    
    s_start = tdvp.EvoMPS_TDVP_Generic_Dissipative(N, D, q, get_ham(N, lam))
    #s_start = tdvp.EvoMPS_TDVP_Generic(N, D, q, get_ham(N, lam))

    if load_from is not None:
        s_start.load_state(load_from)
    else:
        s_start.randomize()
        
    #print "mps_sZ", sp.array([s_start.expect_1s(Sz, n).real for n in xrange(1, s_start.N + 1)])
        
    psi = get_full_state(s_start)
    s_start = None
    
    Hfull = get_full_op(get_ham(N, lam))

    linds = get_linds(N, eps)
    linds = [(n, L.reshape(tuple([sp.prod(L.shape[:sp.ndim(L)/2])]*2))) for (n, L) in linds]
    linds_full = [sp.kron(sp.eye(2**(n-1)), sp.kron(L, sp.eye(2**(N - n + 1) / L.shape[0]))) for (n, L) in linds]
    for L in linds_full:
        assert L.shape == Hfull.shape

    Qfull = -1.j * Hfull - 0.5 * sp.sum([L.conj().T.dot(L) for L in linds_full], axis=0)

    Szfull = []
    for n in xrange(1, N + 1):
        Szn = [None] * (N + 1)
        Szn[n] = Sz
        Szfull.append(get_full_op(Szn))
        
    #print Szfull
    Hexp = 0
    for i in xrange(N_steps):
        Szs = sp.array([sp.vdot(psi, Szfull[n].dot(psi)).real for n in xrange(N)])
        
        pHExp = Hexp
        Hexp = sp.vdot(psi, Hfull.dot(psi))
        Qexp = sp.vdot(psi, Qfull.dot(psi))
        
        if i % res_every == 0:
            if resQ is not None:
                resQ.put([pid, i / res_every, Szs])
            else:
                print pid, i / res_every, Hexp.real, Hexp.real-pHExp.real, Szs
        
        dpsi = dt * (Qfull.dot(psi) - Qexp * psi) #norm-preservation
        #error is about 10* more than expected for Euler integration
        
        for L in linds_full:
            u = sp.random.normal(0, sp.sqrt(dt), (2,))
            W = (u[0] + 1.j * u[1]) / sp.sqrt(2)
            Lexp = sp.vdot(psi, L.dot(psi))
            dpsi += (W + sp.conj(Lexp) * dt) * (L.dot(psi) - Lexp * psi)
        
        #print (sp.vdot(psi, Qfull.dot(Qfull).dot(psi)) - Qexp**2).real * dt**2 
        #error is from this. **why not in mps case?**... the variance is quite large...
        psi += dpsi
        psi /= sp.sqrt(sp.vdot(psi, psi))

        
def compute_ground(s, tol=1E-6, step=0.05):
    j = 0
    eta = 1000
    while eta > tol:
        s.update()
        if j % 10 == 9:
            s.vari_opt_ss_sweep()
        else:
            s.take_step(step)
        eta = s.eta.real
        print eta
        j += 1
        
def go(load_from=None, N_steps=100, resQ=None, pid=None):
    sp.random.seed(pid)
    
    s_start = tdvp.EvoMPS_TDVP_Generic_Dissipative(N, D, q, get_ham(N, lam))
    #s_start = tdvp.EvoMPS_TDVP_Generic(N, D, q, get_ham(N, lam))
    print s_start.D

    if load_from is not None:
        s_start.load_state(load_from)
    else:
        s_start.randomize()

    s = s_start

    linds = get_linds(N, eps)

    eta = 1
    Hexp = 0
    for i in xrange(N_steps):
        s.update()

        """
        Compute expectation values!
        """
        Szs = sp.array([s.expect_1s(Sz, n).real for n in xrange(1, s.N + 1)])
        pHexp = Hexp
        Hexp = s.H_expect
        if i % res_every == 0:
            if resQ is not None:
                resQ.put([pid, i / res_every, Szs])
            else:
                print pid, i / res_every, Hexp.real, Hexp.real - pHexp.real, Szs

        """
        Carry out next step!
        """
        s.take_step_dissipative(dt, linds)

        
if __name__ == '__main__':
    #
    #s_start_pure = tdvp.EvoMPS_TDVP_Generic_Dissipative(N, D, q, get_ham(N, lam))
    #s_start_pure.zero_tol = zero_tol
    #s_start_pure.sanity_checks = sanity_checks
    #compute_ground(s_start_pure)
    #s_start_pure.save_state("XXZ_start_s.npy")
    
    #go_exact(load_from=None, N_steps=N_steps, pid=1)
    #go(load_from=None, N_steps=N_steps, pid=1)
    #exit()
    
    N_proc = N_samp
    pp = Pool(processes=N_proc)
    resQ = Manager().Queue()
    workers = [pp.apply_async(go_exact, kwds={'load_from': None, 'N_steps': N_steps, 
                                        'resQ': resQ, 'pid': n}) for n in xrange(N_proc)]
    
    save_every = 10
    
    N_res = N_steps / res_every
    res_array = sp.zeros((N_res, N))
    res_count = sp.zeros((N_res), int)
    while True:
        try:
            pid, i, res = resQ.get(True, 5)
            res_array[i] += sp.array(res)
            res_count[i] += 1

            if res_count[i] == N_samp:
                print i, res_array[i] / N_samp
                
                if i % save_every or i == N_res - 1:
                    sp.save("XXZ_res.npy", res_array)
                    sp.save("XXZ_res_count.npy", res_count)
                
                if i == N_res - 1:
                    print "**all results in**"
                    break
                
        except qu.Empty:
            continue
    
    
    import matplotlib.pyplot as plt
    plt.plot((res_array / N_samp)[-1, :])
    plt.show()