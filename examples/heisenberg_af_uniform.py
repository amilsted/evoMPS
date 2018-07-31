#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
A demonstration of evoMPS: Calculation of approximate excitation spectrum
for the Heisenberg model.

@author: Ashley Milsted
"""
from __future__ import absolute_import, division, print_function

import math as ma
import scipy as sp
import evoMPS.tdvp_uniform as tdvp
import evoMPS.dynamics as dy

"""
First, we set up some global variables to be used as parameters.
"""
S = 1                         #Spin: Can be 0.5 or 1.
block_length = 1              #Translation-invariant block length
bond_dim = 32                 #The maximum bond dimension

Jx = 1.00                     #Interaction factors (Jx == Jy == Jz > 0 is the antiferromagnetic Heisenberg model)
Jy = 1.00
Jz = 1.00

tol = 1E-7                    #Ground state tolerance (norm of projected evolution vector)

step = 0.04                   #Imaginary time step size
max_steps = 10000             #Maximum number of iterations

load_saved_ground = False     #Whether to load a saved ground state (if it exists)

auto_truncate = True          #Whether to reduce the bond-dimension if any Schmidt coefficients fall below a tolerance.
zero_tol = 1E-20              #Zero-tolerance for the Schmidt coefficients squared (right canonical form)

num_excitations = 24          #The number of excitations to obtain
num_momenta = 20              #Number of points on momentum axis

plot_results = True

sanity_checks = False         #Whether to perform additional (verbose) sanity checks

use_CUDA = False              #Whether to use CUDA to accelerate certain parts of evoMPS.

random_seed = None            #Random seed to initialize state. Set to None to use default.

#rl = tdvp.logging.getLogger()
#rl.setLevel(tdvp.logging.DEBUG)

"""
Next, we define our Hamiltonian and some observables.
"""
Sx_s1 = ma.sqrt(0.5) * sp.array([[0, 1, 0],
                                 [1, 0, 1],
                                 [0, 1, 0]])
Sy_s1 = ma.sqrt(0.5) * 1.j * sp.array([[0, 1, 0],
                                       [-1, 0, 1],
                                       [0, -1, 0]])
Sz_s1 = sp.array([[1, 0, 0],
                  [0, 0, 0],
                  [0, 0, -1]])

Sx_pauli = sp.array([[0, 1],
                     [1, 0]])
Sy_pauli = 1.j * sp.array([[0, -1],
                           [1, 0]])
Sz_pauli = sp.array([[1, 0],
                     [0, -1]])

if S == 0.5:
    qn = 2
    Sz = Sz_pauli
    Sy = Sy_pauli
    Sx = Sx_pauli
elif S == 1:
    qn = 3
    Sz = Sz_s1
    Sy = Sy_s1
    Sx = Sx_s1
else:
    print("Only S = 1 or S = 1/2 are supported!")
    exit()

"""
A translation invariant (uniform) nearest-neighbour Hamiltonian is a
4-dimensional array defining the nearest-neighbour interaction.
The indices 0 and 1 are the 'bra' indices for the first and
second sites and the indices 2 and 3 are the 'ket' indices:

  ham[s,t,u,v] = <st|h|uv>

The following function will return a Hamiltonian for the chain, given the
the parameters J and h.
"""
def get_ham(Jx, Jy, Jz):
    h = (Jx * sp.kron(Sx, Sx) + Jy * sp.kron(Sy, Sy)
         + Jz * sp.kron(Sz, Sz)).reshape(qn, qn, qn, qn)
    return h

"""
Now we are ready to create an instance of the evoMPS class.
"""
def create(D, seed=None):
    if not seed is None:
        sp.random.seed(seed)
    s = tdvp.EvoMPS_TDVP_Uniform(D, qn, get_ham(Jx, Jy, Jz), L=block_length)
    s.symm_gauge = True
    s.zero_tol = zero_tol
    s.sanity_checks = sanity_checks
    s.ev_arpack_CUDA = use_CUDA
    s.PPinv_use_CUDA = use_CUDA
    return s
    
s = create(bond_dim, seed=random_seed)

"""
The following loads a ground state from a file.
The ground state will be saved automatically when it is declared found.
"""
grnd_fname = "heis_af_uni_L%d_D%d_q%d_S%g_Jx%g_Jy%g_Jz%g_s%g_dtau%g_ground.npy" % (block_length, bond_dim, qn, S, Jx, Jy, Jz, tol, step)

if load_saved_ground:
    try:
        a_file = open(grnd_fname, 'rb')
        s.load_state(a_file)
        a_file.close
        real_time = True
        loaded = True
        print('Using saved ground state: ' + grnd_fname)
    except IOError as e:
        real_time = False
        loaded = False
        print('No existing ground state could be opened.')
else:
    real_time = False
    loaded = False


if __name__ == '__main__':
    """
    Print a table header.
    """
    print("Bond dimensions: " + str(s.D))
    print()
    col_heads = ["Step", "t", "<h>", "d<h>",
                 "Sz", "eta"]
    print("\t".join(col_heads))
    print()
    
    """
    Define a function to print information after each step.
    """
    prev_h = 0
    def cbf(s, i, **kwargs):
        global prev_h
        
        h = s.h_expect.real

        row = [str(i)]

        row.append("%.15g" % h)

        dh = h - prev_h
        prev_h = h

        row.append("%.2e" % (dh))

        """
        Compute expectation values!
        """
        exSzs = []
        for k in range(s.L):
            exSzs.append("%.3g" % s.expect_1s(Sz, k=k).real)
        row += exSzs
        
        row.append("%.6g" % s.eta.real)

        row.append(str(kwargs))

        print("\t".join(row))

    """
    Find the ground state using the TDVP and the nonlinear conjugate gradient method
    """
    if not loaded:
        s = dy.find_ground(s, tol=tol, h_init=step, cb_func=cbf, max_itr=max_steps)
        s.save_state(grnd_fname)

    """
    Find excitations if we have the ground state.
    """
    print('Finding excitations!')

    ex_ev = []
    ex_p = []
    for p in sp.linspace(0, sp.pi, num=num_momenta):
        print("p = ", p)
        ex_ev.append(s.excite_top_triv(p, nev=num_excitations, ncv=num_excitations * 4))
        ex_p.append([p] * num_excitations)

    """
    Simple plots of the results.
    """
    if plot_results:
        import matplotlib.pyplot as plt

        plt.figure()
        ex_p = sp.array(ex_p).ravel()
        ex_ev = sp.array(ex_ev).ravel()
        plt.plot(ex_p, ex_ev, 'bo', label='top. trivial')
        plt.title('Excitation spectrum')
        plt.xlabel('p')
        plt.ylabel('dE')
        plt.ylim(0, ex_ev.max() * 1.1)
        plt.legend()

        plt.show()
