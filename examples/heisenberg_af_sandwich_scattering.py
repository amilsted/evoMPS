# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 09:29:41 2012

@author: ash

A demonstration of non-uniform dynamics on an infinite chain.

HOW TO USE:
    1. Run the heisenberg_af_uniform script separately to obtain a uniform
       ground state (setting D as desired).
    2. Set use_damping and N (size of the nonuniform region) as desired.
    3. Run this script!

If use_damping is False, dynamic expansion of the nonuniform region will be
used.
"""

import evoMPS.tdvp_sandwich as sw
import scipy as sp

import heisenberg_af_uniform as hu

import matplotlib.pyplot as plt

"""
Steps to simulate:
"""
steps = 300

"""
Step size:
"""
dt = 0.04

"""
The (initial) number of sites in the nonuniform region:
"""
N = 201

"""
Whether to use damping near the boundaries via "optical potential" terms.
Note: Auto-grow cannot currently be used with damping.
"""
use_damping = False

sim = sw.EvoMPS_TDVP_Sandwich(N, hu.s)

if use_damping:
    left_damp = 15
    right_damp = 15
        
    for n in xrange(left_damp):
        e = (left_damp - n) * 1.0/left_damp
        sim.h_nn[n] *= (1 - e * 1.j)

    for n in xrange(sim.N - right_damp, len(sim.h_nn)):
        e = (right_damp - (sim.N - n)) * 1.0/right_damp
        sim.h_nn[n] *= (1 - e * 1.j)
        
Sx = hu.x_ss    
Sy = hu.y_ss    
Sz = hu.z_ss
    
Sp = Sx + 1.j * Sy    
Sm = Sx - 1.j * Sy
    
base_name = "heis_af_sandwich_scattering_N%u_D%u" % (sim.N, hu.s.D)

if __name__ == "__main__":
    mid = int(round(11/20. * sim.N))
    
    sim.apply_op_1s(Sp, mid - 15 - 5)
    sim.apply_op_1s(Sm, mid - 15 + 5)

    sim.apply_op_1s(Sm, mid + 15 - 5)
    sim.apply_op_1s(Sp, mid + 15 + 5)
    
    load_step = 0
    
    if load_step > 0:
        sim.load_state("data/" + base_name + "_%u.npy" % load_step)
    
    op, en, S, OL = sw.go(sim, dt*1.j, steps, RK4=True, 
                          autogrow=not use_damping, autogrow_amount=4,
                          op=Sz, op_save_as=base_name + "_Sz.txt", op_every=1,
                          en_save_as=base_name + "_h.txt",
                          append_saved=False,
                          #save_as="data/" + base_name, save_every=10,
                          counter_start=load_step,
                          csv_file=base_name + ".csv")
                           
    op = sp.array(op)
    
    plt.imshow(op, origin="lower", interpolation="none", 
               aspect="auto", extent=(-10, sim.N + 9, 0, op.shape[0] - 1))
    plt.show()