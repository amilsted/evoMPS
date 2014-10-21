#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
A demonstration of evoMPS by simulation of quench dynamics
for the Heisenberg model.

@author: Ashley Milsted
"""
import math as ma
import scipy as sp
import evoMPS.tdvp_gen as tdvp
import time

"""
First, we set up some global variables to be used as parameters.
"""
S = 1                         #Spin: Can be 0.5 or 1.
N = 10                        #The length of the finite spin chain.
bond_dim = 32                 #The maximum bond dimension

Jx = 1.00                     #Interaction factors (Jx == Jy == Jz > 0 is the antiferromagnetic Heisenberg model)
Jy = 1.00
Jz = 1.00

Jx_quench = 1.                #Factors after quench
Jy_quench = 1.
Jz_quench = 2.

tol_im = 1E-6                 #Ground state tolerance (norm of projected evolution vector)

step = 0.1                    #Imaginary time step size
realstep = 0.01               #Real time step size
real_steps = 1000             #Number of real time steps to simulate

use_split_step = True         #Use the one-site split step integrator from http://arxiv.org/abs/1408.5056 (requires building evoMPS extension modules)

load_saved_ground = True      #Whether to load a saved ground state

auto_truncate = False         #Whether to reduce the bond-dimension if any Schmidt coefficients fall below a tolerance.
zero_tol = 0                  #Zero-tolerance for the Schmidt coefficients squared (right canonical form)

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
A nearest-neighbour Hamiltonian is a sequence of 4-dimensional arrays, one for
each pair of sites.
For each term, the indices 0 and 1 are the 'bra' indices for the first and
second sites and the indices 2 and 3 are the 'ket' indices:

  ham[n][s,t,u,v] = <st|h|uv> (for sites n and n+1)

The following function will return a Hamiltonian for the chain, given the
length N and the parameters J and h.
"""
def get_ham(N, Jx, Jy, Jz):
    h = (Jx * sp.kron(Sx, Sx) + Jy * sp.kron(Sy, Sy)
         + Jz * sp.kron(Sz, Sz)).reshape(qn, qn, qn, qn)
    return [h] * N

"""
The bond dimension for each site is given as a vector, length N + 1.
Here we set the bond dimension = bond_dim for all sites.
"""
D = [bond_dim] * (N + 1)

"""
The site Hilbert space dimension is also given as a vector, length N + 1.
Here, we set all sites to dimension = 2.
"""
q = [qn] * (N + 1)

"""
Now we are ready to create an instance of the evoMPS class.
"""
s = tdvp.EvoMPS_TDVP_Generic(N, D, q, get_ham(N, Jx, Jy, Jz))
s.zero_tol = zero_tol
s.sanity_checks = sanity_checks
#s.canonical_form = 'left'
#s.gauge_fixing = 'left'

"""
The following loads a ground state from a file.
The ground state will be saved automatically when it is declared found.
"""
grnd_fname = "heis_af_N%d_D%d_q%d_S%g_Jx%g_Jy%g_Jz%g_s%g_dtau%g_ground.npy" % (N, bond_dim, qn, S, Jx, Jy, Jz, tol_im, step)

if load_saved_ground:
    try:
        a_file = open(grnd_fname, 'rb')
        s.load_state(a_file)
        a_file.close
        real_time = True
        loaded = True
        s.ham = get_ham(N, Jx_quench, Jy_quench, Jz_quench)
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

    Tim = []
    Him = []
    exSim = []

    Tre = []
    Hre = []
    exSre = []

    """
    Print a table header.
    """
    print "Bond dimensions: " + str(s.D)
    print
    col_heads = ["Step", "t", "<H>", "d<H>",
                 "Sx_3", "Sy_3", "Sz_3",
                 "eta"] #These last three are for testing the midpoint method.
    print "\t".join(col_heads)
    print

    if real_time:
        T = Tre
        H = Hre
        exS = exSre
    else:
        T = Tim
        H = Him
        exS = exSim

    eta = 1
    i = 0
    t0 = time.clock()
    while True:
        if not use_split_step or not real_time:
            s.update(auto_truncate=auto_truncate)
        
        T.append(t)

        H.append(s.H_expect.real)

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
        Sx_3 = s.expect_1s(Sx, 3) #Spin observables for site 3.
        Sy_3 = s.expect_1s(Sy, 3)
        Sz_3 = s.expect_1s(Sz, 3)
        row.append("%.3g" % Sx_3.real)
        row.append("%.3g" % Sy_3.real)
        row.append("%.3g" % Sz_3.real)

        m_n = map(lambda n: s.expect_1s(Sz, n).real, xrange(1, N + 1)) #Magnetization
        exS.append(m_n)

        """
        Switch to real time evolution if we have the ground state.
        """
        if (not real_time and eta < tol_im):
#            print "time:", time.clock() - t0
#            exit()
            real_time = True
            s.save_state(grnd_fname)
            s.ham = get_ham(N, Jx_quench, Jy_quench, Jz_quench)
            T = Tre
            H = Hre
            exS = exSre
            i = 0
            t = 0
            print 'Starting real time evolution!'

        """
        Carry out next step!
        """
        if not real_time:
            if i % 10 == 9 and eta > 1E-5:
                print "Doing DMRG-style sweep."
                s.vari_opt_ss_sweep()
                eta = 100
            else:
                s.take_step(step)
                eta = s.eta.real
            t += 1.j * step
        else:
            if use_split_step:
                s.take_step_split(realstep * 1.j)
            else:
                s.take_step_RK4(realstep * 1.j)
            t += realstep
            eta = s.eta.real
            
        row.append("%.6g" % eta)

        print "\t".join(row)

        i += 1
        if real_time and i > real_steps:
            break

    """
    Simple plots of the results.
    """
    if plot_results:
        import matplotlib.pyplot as plt

        if len(Tim) > 0: #Plot imaginary time evolution of K1 and Mx
            tau = sp.array(Tim).imag

            fig1 = plt.figure(1)
            fig2 = plt.figure(2)
            H_tau = fig1.add_subplot(111)
            H_tau.set_xlabel('tau')
            H_tau.set_ylabel('H')
            H_tau.set_title('Imaginary time evolution: Energy')
            M_tau = fig2.add_subplot(111)
            M_tau.set_xlabel('n')
            M_tau.set_ylabel('tau')
            M_tau.set_title('Imaginary time evolution: Sz')

            H_tau.plot(tau, Him)

            exSim = sp.array(exSim)
            img = M_tau.imshow(exSim, origin='lower', aspect='auto',
                               extent=[1, N, 0, tau[-1]],
                               vmin=exSim[-1].min(), vmax=exSim[-1].max())
            cb = fig2.colorbar(img)
            cb.set_label('Sz')

        #Now plot the real time evolution of K1 and Mx
        t = Tre
        fig3 = plt.figure(3)
        fig4 = plt.figure(4)

        H_t = fig3.add_subplot(111)
        H_t.set_xlabel('t')
        H_t.set_ylabel('H')
        H_t.set_title('Real time evolution: Energy')
        M_t = fig4.add_subplot(111)
        M_t.set_xlabel('n')
        M_t.set_ylabel('t')
        M_t.set_title('Real time evolution: Sz')

        H_t.plot(t, Hre)

        exSre = sp.array(exSre)
        img = M_t.imshow(exSre, origin='lower', aspect='auto',
                         extent=[1, N, 0, t[-1]])
        cb = fig4.colorbar(img)
        cb.set_label('Sz')

        plt.show()
