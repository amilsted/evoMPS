#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
A demonstration of evoMPS: Calculation of approximate excitation spectrum
for the transverse Ising model.

@author: Ashley Milsted
"""

import copy
import scipy as sp
import scipy.linalg as la
import scipy.special as spe
import evoMPS.tdvp_uniform as tdvp

"""
First, we set up some global variables to be used as parameters.
"""

bond_dim = 8                  #The maximum bond dimension

J = 1.00                      #Interaction factor
h = 0.50                      #Transverse field factor

tol_im = 1E-10                #Ground state tolerance (norm of projected evolution vector)

step = 0.08                   #Imaginary time step size

load_saved_ground = True      #Whether to load a saved ground state (if it exists)

auto_truncate = False         #Whether to reduce the bond-dimension if any Schmidt coefficients fall below a tolerance.
zero_tol = 1E-20              #Zero-tolerance for the Schmidt coefficients squared (right canonical form)

num_excitations = 24          #The number of excitations to obtain
num_momenta = 20              #Number of points on momentum axis
top_non_triv = True           #Also look for topologically non-trivial excitations (only useful for h < J)

plot_results = True

"""
Next, we define our Hamiltonian and some observables.
"""
x_ss = sp.array([[0, 1],
                 [1, 0]])
y_ss = 1.j * sp.array([[0, -1],
                       [1, 0]])
z_ss = sp.array([[1, 0],
                 [0, -1]])

"""
A translation invariant (uniform) nearest-neighbour Hamiltonian is a
4-dimensional array defining the nearest-neighbour interaction.
The indices 0 and 1 are the 'bra' indices for the first and
second sites and the indices 2 and 3 are the 'ket' indices:

  ham[s,t,u,v] = <st|h|uv>

The following function will return a Hamiltonian for the chain, given the
the parameters J and h.
"""
def get_ham(J, h):
    ham = -J * (sp.kron(x_ss, x_ss) + h * sp.kron(z_ss, sp.eye(2))).reshape(2, 2, 2, 2)
    return ham

lam = J / h
print "Exact energy = ", -h * 2 / sp.pi * (1 + lam) * spe.ellipe((4 * lam / (1 + lam)**2))

"""
Now we are ready to create an instance of the evoMPS class.
"""
s = tdvp.EvoMPS_TDVP_Uniform(bond_dim, 2, get_ham(J, h))
s.zero_tol = zero_tol

"""
The following loads a ground state from a file.
The ground state will be saved automatically when it is declared found.
"""
grnd_fname = "t_ising_uni_D%d_J%g_h%g_s%g_dtau%g_ground.npy" % (bond_dim, J, h, tol_im, step)

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
    M = []

    """
    Print a table header.
    """
    print "Bond dimensions: " + str(s.D)
    print
    col_heads = ["Step", "t", "<h>", "d<h>",
                 "sig_x_3", "sig_y_3", "sig_z_3",
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
        Sx = s.expect_1s(x_ss)
        Sy = s.expect_1s(y_ss)
        Sz = s.expect_1s(z_ss)
        row.append("%.3g" % Sx.real)
        row.append("%.3g" % Sy.real)
        row.append("%.3g" % Sz.real)

        M.append(Sz.real)

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
            if top_non_triv:
                s2 = copy.deepcopy(s)
                flip_x = la.expm(1.j * sp.pi / 2. * z_ss)
                s2.apply_op_1s(flip_x)
                s2.update()
            ex_ev = []
            ex_ev_nt = []
            ex_p = []
            for p in sp.linspace(0, sp.pi, num=num_momenta):
                print "p = ", p
                ex_ev.append(s.excite_top_triv(p, k=num_excitations, ncv=num_excitations * 4))
                if top_non_triv:
                    ex_ev_nt.append(s.excite_top_nontriv(s2, p, k=num_excitations, ncv=num_excitations * 4))
                else:
                    ex_ev_nt.append([0])
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
            fig2 = plt.figure(2)
            H_tau = fig1.add_subplot(111)
            H_tau.set_xlabel('tau')
            H_tau.set_ylabel('H')
            H_tau.set_title('Imaginary time evolution: Energy')
            M_tau = fig2.add_subplot(111)
            M_tau.set_xlabel('tau')
            M_tau.set_ylabel('M')
            M_tau.set_title('Imaginary time evolution: Magnetization')

            H_tau.plot(tau, H)
            M_tau.plot(tau, M)

        plt.figure()
        ex_p = sp.array(ex_p).ravel()
        ex_ev = sp.array(ex_ev).ravel()
        ex_ev_nt = sp.array(ex_ev_nt).ravel()
        plt.plot(ex_p, ex_ev, 'bo', label='top. trivial')
        if top_non_triv:
            plt.plot(ex_p, ex_ev_nt, 'ro', label='top. non-trivial')
        plt.title('Excitation spectrum')
        plt.xlabel('p')
        plt.ylabel('dE')
        plt.ylim(0, max(ex_ev.max(), ex_ev_nt.max()) * 1.1)
        plt.legend()

        plt.show()
