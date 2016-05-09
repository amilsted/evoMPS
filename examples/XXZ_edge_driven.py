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
import Queue as qu
from multiprocessing import Pool, Queue, Manager
from multiprocessing.pool import ThreadPool
import copy

"""
First, we set up some global variables to be used as parameters.
"""
N = 101                       #The length of the finite spin chain.
bond_dim = 32                 #The maximum bond dimension

N_samp = 4                    #Number of samples

lam = 1.0
eps = 1.0

N_steps = 20
res_every = 1
dt = 0.01                     #Real-time step size

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
                     
"""
A nearest-neighbour Hamiltonian is a sequence of 4-dimensional arrays, one for
each pair of sites.
For each term, the indices 0 and 1 are the 'bra' indices for the first and
second sites and the indices 2 and 3 are the 'ket' indices:

  ham[n][s,t,u,v] = <st|h|uv> (for sites n and n+1)

The following function will return a Hamiltonian for the chain, given the
length N and the parameters J and h.
"""
def get_ham(N, lam):
    h = (2. * sp.kron(Sp, Sm) + 2. * sp.kron(Sm, Sp)
         + lam * sp.kron(Sz, Sz)).reshape(2, 2, 2, 2)
    return [h] * N
    
def get_linds(N, eps):
    Sp1 = (sp.kron(Sp, sp.eye(2))).reshape(2, 2, 2, 2)
    Sm2 = (sp.kron(sp.eye(2), Sm)).reshape(2, 2, 2, 2)
    
    L1 = [None] * N
    L2 = [None] * N
    
    L1[1] = sp.sqrt(eps) * Sp1
    L2[N - 1] = sp.sqrt(eps) * Sm2
    
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

def go(load_from=None, N_thread_samp=1, N_steps=100, N_threads=1, resQ=None, pid=None):

    s_start = tdvp.EvoMPS_TDVP_Generic_Dissipative(N, D, q, get_ham(N, lam))

    if load_from is not None:
        s_start.load_state(load_from)

    ss = [s_start]
    ss += [copy.deepcopy(s_start) for j in xrange(N_thread_samp - 1)]

    pool = ThreadPool(N_threads)

    linds = get_linds(N, eps)

    eta = 1
    for i in xrange(N_steps):
        pool.map(lambda s: s.update(), ss)

        #Hs = sp.array(pool.map(lambda s: s.H_expect.real, ss))
        
        """
        Compute expectation values!
        """
        Szs = sp.array(pool.map(lambda s: sp.array([s.expect_1s(Sz, n).real for n in xrange(1, s.N + 1)]), ss))

        if i % res_every == 0 and resQ is not None:
            resQ.put([pid, i / res_every, sp.mean(Szs, axis=0)])

        """
        Carry out next step!
        """
        pool.map(lambda s: s.take_step_dissipative(dt, linds), ss)

        
if __name__ == '__main__':
    #
    #s_start_pure = tdvp.EvoMPS_TDVP_Generic_Dissipative(N, D, q, get_ham(N, lam))
    #s_start_pure.zero_tol = zero_tol
    #s_start_pure.sanity_checks = sanity_checks
    #compute_ground(s_start_pure)
    #s_start_pure.save_state("XXZ_start.npy")
    
    N_proc = N_samp
    pp = Pool(processes=N_proc)
    resQ = Manager().Queue()
    workers = [pp.apply_async(go, kwds={'load_from': "XXZ_start.npy", 'N_steps': N_steps, 
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
                
                if i % save_every:
                    sp.save("XXZ_res.npy", res_array)
                    sp.save("XXZ_res_count.npy", res_count)
                
                if i == N_res - 1:
                    print "**all results in**"
                    break
                
        except qu.Empty:
            continue
    
    
    print res_array