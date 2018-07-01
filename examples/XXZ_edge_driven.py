#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
A demonstration of simulating open system dynamics by sampling
over randomized pure state trajectories.

This simulates the dynamics of an edge-driven XXZ chain,
which has an analytic solution for lam = 1.0, eps = 2.0.
See: T. Prosen, Phys. Rev. Lett. 107, 137201 (2011)

For comparison, with short spin chains this script also
evolves the full density matrix corresponding to the sample
trajectories using Euler integration of the Lindblad master equation.

The starting state is a random state composed of N_samp random pure states.

@author: Ashley Milsted
"""
from __future__ import absolute_import, division, print_function

import math as ma
import scipy as sp
import evoMPS.tdvp_gen_diss as tdvp
import time
import copy
import multiprocessing as mp

"""
First, we set up some global variables to be used as parameters.
"""
N = 6                         #The length of the finite spin chain.
bond_dim = 64                 #The maximum bond dimension

Nmax_fullrho = 8              #Maximum chain length for computing the full density matrix

num_procs = mp.cpu_count()    #Number of parallel processes to use

live_plot_mode = True         #Attempt to show a live plot of expectation values during evolution
                              #If False, save results to a file instead!
                              
plot_saved_data = False       #Do not simulate. Instead load and plot saved data.
plot_res = -1                 #Which result to plot. -1 means the last result.

#Set number of sample trajectories based on available cores if live plotting
if live_plot_mode:
    if N <= Nmax_fullrho:
        N_samp = num_procs - 1  #the density matrix computation requires a process too
    else:
        N_samp = num_procs
else:
    N_samp = 20 #number of samples to compute when saving data to a file

#System parameters
lam = 1.0
eps = 2.0

dt = 0.001                   #Time-step size for Euler integration
N_steps = 1000               #Number of steps to compute
res_every = 20               #Number of steps to wait between computation of results (expectation values)

random_seed = 1              #Seed used to generate the pseudo-random starting state.
                             #The same random number generator is used to simulate the Wiener processes needed to
                             #integrate the stochastic differential equation.

"""
Next, we define the operators used, including the Hamiltonian and the Lindblad operators.
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
evoMPS will adjust it to the maximum useful value near the ends of the chain.
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
        for n in range(N, 0, -1):
            A = s.A[n][ind[n-1]].dot(A)

        psi[ind] = A[0,0]
    psi = psi.ravel()

    return psi
    
def get_full_op(op):
    fop = sp.zeros((2**N, 2**N), dtype=sp.complex128)
    for n in range(1, min(len(op), N + 1)):
        if op[n] is None:
            continue
            
        n_sites = len(op[n].shape) / 2
        opn = op[n].reshape(2**n_sites, 2**n_sites)
        fop_n = sp.kron(sp.eye(2**(n - 1)), sp.kron(opn, sp.eye(2**(N - n - n_sites + 1))))

        assert fop.shape == fop_n.shape
        fop += fop_n
    return fop

def go(load_from=None, N_steps=100, resQ=None, pid=None, stateQ=None):
    sp.random.seed(random_seed + pid)
    
    s_start = tdvp.EvoMPS_TDVP_Generic_Dissipative(N, D, q, get_ham(N, lam), get_linds(N, eps))
    print("Starting MPS sample", pid)

    if load_from is not None:
        s_start.load_state(load_from)
    else:
        s_start.randomize()
        
    s = s_start
    
    #Send the state to the density matrix simulator
    if N <= Nmax_fullrho and stateQ is not None:
        psi = get_full_state(s)
        stateQ.put([pid, psi])

    eta = 1
    Hexp = 0
    for i in range(N_steps + 1):
        s.update()

        Szs = [s.expect_1s(Sz, n).real for n in range(1, s.N + 1)]
        pHexp = Hexp
        Hexp = s.H_expect
        if i % res_every == 0:
            if resQ is not None:
                resQ.put([pid, i, Szs])
            else:
                print(pid, i / res_every, Hexp.real, Hexp.real - pHexp.real, Szs)

        s.take_step_dissipative(dt)
    
def fullrho(qu, squ):
    print("#Starting full density matrix simulation!")
    
    #Build state from pure states received from the MPS processes
    rho = sp.zeros((2**N, 2**N), dtype=sp.complex128)
    psis = [None] * N_samp
    for n in range(N_samp):
        pnum, psi = squ.get()
        psis[pnum] = psi
        rho += sp.outer(psi, psi.conj())
        squ.task_done()
        
    rho /= sp.trace(rho)

    Hfull = get_full_op(get_ham(N, lam))

    linds = get_linds(N, eps)
    linds = [(n, L.reshape(tuple([sp.prod(L.shape[:sp.ndim(L)/2])]*2))) for (n, L) in linds]
    linds_full = [sp.kron(sp.eye(2**(n-1)), sp.kron(L, sp.eye(2**(N - n + 1) / L.shape[0]))) for (n, L) in linds]
    for L in linds_full:
        assert L.shape == Hfull.shape

    Qfull = -1.j * Hfull - 0.5 * sp.sum([L.conj().T.dot(L) for L in linds_full], axis=0)

    szs = [None] + [sp.kron(sp.kron(sp.eye(2**(n - 1)), Sz), sp.eye(2**(N - n))) for n in range(1, N + 1)]
    
    for i in range(N_steps + 1):
        rho /= sp.trace(rho)
        esz = []
        for n in range(1, N + 1):
            esz.append(sp.trace(szs[n].dot(rho)).real)

        if i % res_every == 0:
            if qu is None:
                print(esz)
            else:
                qu.put([-1, i, esz])
                qu.put([-2, i, [sp.NaN] * N]) #this slot is reserved for a second "exact" result
        
        #Do Euler steps, approximately integrating the Lindblad master equation
        rho += dt * (Qfull.dot(rho) + rho.dot(Qfull.conj().T) +
                       sum([L.dot(rho).dot(L.conj().T) for L in linds_full]))
    
def plotter(q):
    import matplotlib
    matplotlib.use("wxagg")
    import matplotlib.pyplot as plt

    fig = plt.figure()
    lns = [plt.plot([0]*N, ':')[0] for n in range(N_samp)]
    av = plt.plot([0]*N, 'k-', linewidth=2.0)[0]
    av_err1 = plt.plot([0]*N, 'k-', linewidth=1.0)[0]
    av_err2 = plt.plot([0]*N, 'k-', linewidth=1.0)[0]
    exa = plt.plot([0]*N, 'r-', linewidth=2.0)[0]
    #exa_s = plt.plot([0]*N, 'm--', linewidth=2.0)[0]
    
    plt.legend([exa, av], ["Density matrix", "Sample average"])
    plt.xlabel(r"$n$")
    plt.ylabel(r"$\langle \sigma^z_n \rangle$")
    
    plt.ylim((-1, 1))
    plt.xlim((0, N - 1))
    plt.ion()
    plt.show()
    i_buf = 0
    data_buf = [[None] * (N_samp + 2)]
    
    if N <= Nmax_fullrho:
        effbuflen = (N_samp + 2)
    else:
        effbuflen = N_samp
    
    while True:
        data = q.get()
        if data is None:
            break
        num = data[0]
        i = data[1]
        ys = data[2]
        
        i_off = (i - i_buf) / res_every
        if i_off >= len(data_buf):
            for j in range(len(data_buf), i_off + 1):
                data_buf.append([None] * (N_samp + 2))
        data_buf[i_off][num] = ys
        
        if not None in data_buf[0][:effbuflen]:
            print("Plotting results for step", i_buf, "buffer length", len(data_buf))

            for j in range(N_samp):
                lns[j].set_ydata(data_buf[0][j])
            av_ys = sp.zeros_like(ys)
            for da in data_buf[0][:-2]:
                av_ys += da
            av_ys /= N_samp
            av.set_ydata(av_ys)
            
            #Compute stdev and use it to display error
            av_ys_var = 1./(N_samp - 1) / N_samp * sp.sum([(da - av_ys)**2 for da in data_buf[0][:-2]], axis=0)
            av_ys_e1 = av_ys + sp.sqrt(av_ys_var)
            av_ys_e2 = av_ys - sp.sqrt(av_ys_var)

            av_err1.set_ydata(av_ys_e1)
            av_err2.set_ydata(av_ys_e2)

            exa.set_ydata(data_buf[0][-1])

            #exa_s.set_ydata(data_buf[0][-2])
            
            fig.canvas.draw()

            data_buf.pop(0)
            i_buf += res_every
            plt.pause(0.01)
            
        q.task_done()
        
    plt.ioff()
    plt.show()

def get_filename():
    return 'sample_data_eps%g_lam%g_N%u_ns%u_ts%u_dt%g_resev%u_maxD%u.bin' % (eps, lam, N, N_samp, 
                                                                              N_steps, dt, res_every, 
                                                                              bond_dim)
                       
def writer(q):
    df = sp.memmap(get_filename(), dtype=sp.float64, mode='w+', shape=(N_samp + 2, N_steps / res_every + 1, N))
    while True:
        data = q.get()
        if data is None:
            break
        num = data[0]
        i = data[1]
        ys = data[2]

        df[num, i / res_every, :] = ys

        if i == N_steps:
            df.flush()
            print("Sample", num, "finished. Data saved.")
            
    del df

def plot_saved():
    import matplotlib
    import matplotlib.pyplot as plt

    data = sp.memmap(get_filename(), dtype=sp.float64, mode='r', shape=(N_samp + 2, N_steps / res_every + 1, N))
    
    exa = data[-1]
    
    data = data[:-2]
    
    fins = sp.array([d[plot_res] for d in data if not sp.all(d[plot_res] == 0)])
    nsam = len(fins)
    print("Samples:", nsam)
    av = fins.sum(axis=0) / nsam

    av_var = 1./(nsam - 1) / nsam * sp.sum((fins/nsam - av)**2, axis=0) 
    av_e1 = av + sp.sqrt(av_var)
    av_e2 = av - sp.sqrt(av_var)

    
    plt.figure()
    pav = plt.plot(av, 'k-')[0]
    plt.plot(av_e1, 'k--')
    plt.plot(av_e2, 'k--')

    if not sp.all(exa[-1] == 0):
        pexa = plt.plot(exa[-1], 'r-')[0]
        plt.legend([pexa, pav], ["Density matrix", "Sample average"])
    
    plt.ylim((-1, 1))
    
    plt.xlabel(r"$n$")
    plt.ylabel(r"$\langle \sigma^z_n \rangle$")
    
    plt.show()
    

def f(args):
    pid, resQ, stateQ = args
    go(load_from=None, N_steps=N_steps, resQ=resQ, pid=pid, stateQ=stateQ)
        
if __name__ == "__main__":
    if plot_saved_data:
        plot_saved()
    else:
        mpm = mp.Manager()
        qu = mpm.Queue()
        state_q = mpm.Queue()
        
        if live_plot_mode:
            res_handler = plotter
        else:
            res_handler = writer
        
        resp = mp.Process(target=res_handler, args=(qu,))
        resp.start()
        
        p = mp.Pool(num_procs)
        
        if N <= Nmax_fullrho:
            exa = p.apply_async(fullrho, args=(qu, state_q))
        
        p.map(f, list(zip(list(range(N_samp)), [qu] * N_samp, [state_q] * N_samp)))

        if N <= Nmax_fullrho:
            exa.get()
        
        qu.put(None)
        resp.join()
