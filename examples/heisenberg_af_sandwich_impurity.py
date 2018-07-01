# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 09:29:41 2012

@author: ash

Finds the ground state of the anti-ferromagnetic Heisenberg model
on an infinite chain with an impurity.

HOW TO USE:
    1. Run heisenberg_af_uniform to find the uniform ground state.
    2. Run this script to solve the impurity problem.

NOTE: Depending on the bond dimension D (set in heisenberg_af_uniform.py),
      N may need to be adjusted in order to accurately represent the ground
      state. Higher bond-dimensions may converge quicker (D=64 is quite good).
"""
from __future__ import absolute_import, division, print_function

import evoMPS.tdvp_sandwich as sw
import heisenberg_af_uniform as hu

N = 120                             #Number of sites in the non-uniform region

imp_pos = N//2                       #Position (of the first site) of the impurity
lam = -2.3                          #Factor by which to multiply the Ham. term at imp_pos

dtau = 0.08                         #Imaginary time step size
max_steps = 10000                   #Maximum number of steps
tol = 5E-8                          #Ground state tolerance

plot_results = True



def get_h_nn(n):
    if n < 0:
        return sim.h_nn[0]
    elif n >= len(sim.h_nn):
        return sim.h_nn[-1]
    else:
        return sim.h_nn[n]

sim = sw.EvoMPS_TDVP_Sandwich(N, hu.s)
sim.h_nn[imp_pos] = (1 + lam) * sim.h_nn[1]

Sx = hu.Sx
Sy = hu.Sy
Sz = hu.Sz

Sp = Sx + 1.j * Sy
Sm = Sx - 1.j * Sy

if __name__ == "__main__":

    sim.add_noise()

    sw.go(sim, dtau, max_steps, tol=tol)

    if plot_results:
        import matplotlib.pyplot as plt

        plt.plot(list(range(-10, sim.N + 11)), [sim.expect_1s(Sx, n).real for n in range(-10, sim.N + 11)], label='Sx')
        plt.plot(list(range(-10, sim.N + 11)), [sim.expect_1s(Sy, n).real for n in range(-10, sim.N + 11)], label='Sy')
        plt.plot(list(range(-10, sim.N + 11)), [sim.expect_1s(Sz, n).real for n in range(-10, sim.N + 11)], label='Sz')
        plt.xlabel('site')
        plt.legend()

        plt.figure()
        plt.plot([sim.expect_2s(get_h_nn(n), n).real for n in range(-10, sim.N + 11)])
        plt.xlabel('site')
        plt.ylabel('h')
        plt.show()
