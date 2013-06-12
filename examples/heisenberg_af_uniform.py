#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
A demonstration of evoMPS: Calculation of approximate excitation spectrum
for the transverse Ising model.

@author: Ashley Milsted
"""

import math as ma
import scipy as sp
import evoMPS.tdvp_uniform as tdvp

"""
First, we set up some global variables to be used as parameters.
"""

S = 1                         #Spin: Can be 0.5 or 1.
bond_dim = 16                 #The maximum bond dimension

Jx = 1.00                     #Interaction factors (Jx == Jy == Jz > 0 is the antiferromagnetic Heisenberg model)
Jy = 1.00
Jz = 1.00

tol_im = 1E-10                #Ground state tolerance (norm of projected evolution vector)

step = 0.1                    #Imaginary time step size

load_saved_ground = True      #Whether to load a saved ground state (if it exists)

auto_truncate = False         #Whether to reduce the bond-dimension if any Schmidt coefficients fall below a tolerance.
zero_tol = 1E-20              #Zero-tolerance for the Schmidt coefficients squared (right canonical form)

num_excitations = 24          #The number of excitations to obtain
num_momenta = 20              #Number of points on momentum axis

plot_results = True

sanity_checks = False         #Whether to perform additional (verbose) sanity checks

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
    print "Only S = 1 or S = 1/2 are supported!"
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
s = tdvp.EvoMPS_TDVP_Uniform(bond_dim, qn, get_ham(Jx, Jy, Jz))
s.zero_tol = zero_tol
s.sanity_checks = sanity_checks

"""
The following loads a ground state from a file.
The ground state will be saved automatically when it is declared found.
"""
grnd_fname = "heis_af_uni_D%d_q%d_S%g_Jx%g_Jy%g_Jz%g_s%g_dtau%g_ground.npy" % (bond_dim, qn, S, Jx, Jy, Jz, tol_im, step)

if load_saved_ground:
    try:
        a_file = open(grnd_fname, 'rb')
        s.load_state(a_file)
        a_file.close
        real_time = True
        loaded = True
        print 'Using saved ground state: ' + grnd_fname
    except IOError as e:
        real_time = False
        loaded = False
        print 'No existing ground state could be opened.'
else:
    real_time = False
    loaded = False


if __name__ == '__main__':
    """
    Prepare some loop variables and some vectors to hold data from each step.
    """
    t = 0

    T = []
    H = []

    """
    Print a table header.
    """
    print "Bond dimensions: " + str(s.D)
    print
    col_heads = ["Step", "t", "<h>", "d<h>",
                 "Sx", "Sy", "Sz",
                 "eta"] #These last three are for testing the midpoint method.
    print "\t".join(col_heads)
    print

    eta = 1
    i = 0
    while True:
        T.append(t)

        s.update(auto_truncate=auto_truncate)

        H.append(s.h_expect.real)

        row = [str(i)]
        row.append(str(t))
        row.append("%.15g" % H[-1])

        if len(H) > 1:
            dH = H[-1] - H[-2]
        else:
            dH = 0

        row.append("%.2e" % (dH.real))

        """
        Compute expectation values!
        """
        exSx = s.expect_1s(Sx)
        exSy = s.expect_1s(Sy)
        exSz = s.expect_1s(Sz)
        row.append("%.3g" % exSx.real)
        row.append("%.3g" % exSy.real)
        row.append("%.3g" % exSz.real)

        """
        Carry out next step!
        """
        s.take_step(step)
        t += 1.j * step

        eta = s.eta.real
        row.append("%.6g" % eta)

        print "\t".join(row)

        i += 1

        """
        Switch to real time evolution if we have the ground state.
        """
        if eta < tol_im or loaded:
            s.save_state(grnd_fname)
            print 'Finding excitations!'

            ex_ev = []
            ex_p = []
            for p in sp.linspace(0, sp.pi, num=num_momenta):
                print "p = ", p
                ex_ev.append(s.excite_top_triv(p, k=num_excitations, ncv=num_excitations * 4))
                ex_p.append([p] * num_excitations)
            break
    """
    Simple plots of the results.
    """
    if plot_results:
        import matplotlib.pyplot as plt

        if not loaded: #Plot imaginary time evolution of K1 and Mx
            tau = sp.array(T).imag

            fig1 = plt.figure(1)
            H_tau = fig1.add_subplot(111)
            H_tau.set_xlabel('tau')
            H_tau.set_ylabel('H')
            H_tau.set_title('Imaginary time evolution: Energy')

            H_tau.plot(tau, H)

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
