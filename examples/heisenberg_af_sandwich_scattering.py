# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 09:29:41 2012

@author: ash

A demonstration of non-uniform dynamics on an infinite chain.

HOW TO USE:
    1. Run the heisenberg_af_uniform script separately to obtain a uniform
       ground state (setting D as desired).
    2. Set N (size of the nonuniform region) as desired.
    3. Run this script!

"""
from __future__ import absolute_import, division, print_function

import evoMPS.tdvp_sandwich as sw
import scipy as sp
import heisenberg_af_uniform as hu

N = 192                            #Number of sites in the non-uniform region

dt = 0.01                          #Imaginary time step size
steps = 1000                       #Maximum number of steps

plot_results = True

sim = sw.EvoMPS_TDVP_Sandwich(N, hu.s)

Sx = hu.Sx
Sy = hu.Sy
Sz = hu.Sz

Sp = Sx + 1.j * Sy
Sm = Sx - 1.j * Sy

base_name = "heis_af_sandwich_scattering_N%u_D%u" % (sim.N, hu.s.D)

if __name__ == "__main__":
    mid = N//2

    sim.apply_op_1s(Sp, mid - 15 - 5)
    sim.apply_op_1s(Sm, mid - 15 + 5)

    sim.apply_op_1s(Sm, mid + 15 - 5)
    sim.apply_op_1s(Sp, mid + 15 + 5)

    op, en, S, OL = sw.go(sim, dt*1.j, steps, RK4=True, op=Sz, op_every=1,
                          autogrow=True, autogrow_amount=4//hu.s.L)

    if plot_results:
        import matplotlib.pyplot as plt

        op = sp.array(op)
        plt.imshow(op, origin="lower", interpolation="none",
                   aspect="auto", extent=(-10, sim.N + 9, 0, (op.shape[0] - 1) * dt))
        plt.xlabel('site')
        plt.ylabel('t')
        cb = plt.colorbar()
        cb.set_label('Sz')
        plt.show()