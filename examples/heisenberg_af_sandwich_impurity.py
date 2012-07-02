# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 09:29:41 2012

@author: ash

Finds the ground state of the anti-ferromagnetic Heisenberg model
on an infinite chain with an impurity.

HOW TO USE:
    1. Run heisenberg_af_uniform to find the uniform ground state.
    2. Run this script to solve the impurity problem.

NOTE: Depending on the bond dimension (set in heisenberg_af_uniform.py),
      N may need to be adjusted in order to accurately represent the ground
      state.
"""

import evoMPS.tdvp_sandwich as sw
import heisenberg_af_uniform as hu
import matplotlib.pyplot as plt

"""
The number of sites in the nonuniform region:
"""
N = 161

"""
The "impurity factor" lambda.
"""
lam = -2

"""
The impurity position:
"""
imp_pos = 81

"""
State tolerance target:
"""
tol = 1E-6

"""
Maximum steps:
"""
max_steps = 10000

"""
Step size:
"""
dtau = 0.1

sim = sw.EvoMPS_TDVP_Sandwich(N, hu.s)

def h_nn(n,s,t,u,v):
    if n == imp_pos + sim.grown_left - sim.shrunk_left:
        return (1 + lam) * hu.h_nn(s,t,u,v)
    else:
        return hu.h_nn(s,t,u,v)

sim.h_nn = h_nn
sim.gen_h_matrix()
        
def Sx(n,s,t):
    return hu.x_ss(s,t)
    
def Sy(n,s,t):
    return hu.y_ss(s,t)
    
def Sz(n,s,t):
    return hu.z_ss(s,t)
    
def Sp(n,s,t):
    return Sx(s,t) + 1.j * Sy(s,t)
    
def Sm(n,s,t):
    return Sx(s,t) - 1.j * Sy(s,t)
    
base_name = "heis_af_sandwich_impurity_N%u_m%u_D%u_lam%g" % (sim.N, imp_pos, 
                                                             hu.s.D, lam)

if __name__ == "__main__":
    
    load_step = 0
    
    if load_step > 0:
        sim.load_state("data/" + base_name + "_%u.npy" % load_step)
    
    sw.go(sim, dtau, max_steps, tol=tol, 
          op=Sx, op_save_as=base_name + "_Sx.txt", op_every=3,
          append_saved=False,
          counter_start=load_step,
          csv_file=base_name + ".csv",
          autogrow=False)
           
    plt.plot(map(lambda n: sim.expect_1s(Sx, n).real, range(-10, sim.N + 11)))
    plt.plot(map(lambda n: sim.expect_1s(Sy, n).real, range(-10, sim.N + 11)))
    plt.plot(map(lambda n: sim.expect_1s(Sz, n).real, range(-10, sim.N + 11)))
    plt.show()
    
    plt.plot(map(lambda n: sim.expect_2s(sim.h_nn, n).real, range(-10, sim.N + 11)))
    plt.show()
    