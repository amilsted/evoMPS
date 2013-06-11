#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
A demonstration of evoMPS by simulation of quench dynamics
for the transverse Ising model.

@author: Ashley Milsted
"""

import scipy as sp
import evoMPS.tdvp_gen as tdvp

"""
First, we set up some global variables to be used as parameters.
"""

N = 20                        #The length of the finite spin chain.
bond_dim = 32                 #The maximum bond dimension

J = 1.00                      #Interaction factor
h = 0.50                      #Transverse field factor
h_quench = -0.5               #Field factor after quench

tol_im = 1E-10                #Ground state tolerance (norm of projected evolution vector)

step = 0.08                   #Imaginary time step size
realstep = 0.01               #Real time step size
real_steps = 100              #Number of real time steps to simulate

load_saved_ground = True      #Whether to load a saved ground state

auto_truncate = True          #Whether to reduce the bond-dimension if any Schmidt coefficients fall below a tolerance.
zero_tol = 1E-10              #Zero-tolerance for the Schmidt coefficients squared (right canonical form)

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
A nearest-neighbour Hamiltonian is a sequence of 4-dimensional arrays, one for
each pair of sites.
For each term, the indices 0 and 1 are the 'bra' indices for the first and
second sites and the indices 2 and 3 are the 'ket' indices:

  ham[n][s,t,u,v] = <st|h|uv> (for sites n and n+1)

The following function will return a Hamiltonian for the chain, given the
length N and the parameters J and h.
"""
def get_ham(N, J, h):
    ham = -J * (sp.kron(x_ss, x_ss) + h * sp.kron(z_ss, sp.eye(2))).reshape(2, 2, 2, 2)
    ham_end = ham + h * sp.kron(sp.eye(2), z_ss).reshape(2, 2, 2, 2)
    return [None] + [ham] * (N - 2) + [ham_end]

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

"""
Print the exact ground state energy
"""
if h == J:
    E = - 2 * abs(sp.sin(sp.pi * (2 * sp.arange(N) + 1) / (2 * (2 * N + 1)))).sum()
    print "Exact ground state energy = %.15g" % E

"""
Now we are ready to create an instance of the evoMPS class.
"""
s = tdvp.EvoMPS_TDVP_Generic(N, D, q, get_ham(N, J, h))
s.zero_tol = zero_tol

"""
The following loads a ground state from a file.
The ground state will be saved automatically when it is declared found.
"""
grnd_fname = "t_ising_N%d_D%d_J%g_h%g_s%g_dtau%g_ground.npy" % (N, bond_dim, J, h, tol_im, step)

if load_saved_ground:
    try:
        a_file = open(grnd_fname, 'rb')
        s.load_state(a_file)
        a_file.close
        real_time = True
        loaded = True
        s.ham = get_ham(N, J, h_quench)
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
    Mim = []

    Tre = []
    Hre = []
    Mre = []

    """
    Print a table header.
    """
    print "Bond dimensions: " + str(s.D)
    print
    col_heads = ["Step", "t", "<H>", "d<H>",
                 "sig_x_3", "sig_y_3", "sig_z_3",
                 "M_x", "eta"] #These last three are for testing the midpoint method.
    print "\t".join(col_heads)
    print

    if real_time:
        T = Tre
        H = Hre
        M = Mre
    else:
        T = Tim
        H = Him
        M = Mim

    eta = 1
    i = 0
    while True:
        T.append(t)

        s.update(auto_truncate=auto_truncate)

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
        Sx_3 = s.expect_1s(x_ss, 3) #Spin observables for site 3.
        Sy_3 = s.expect_1s(y_ss, 3)
        Sz_3 = s.expect_1s(z_ss, 3)
        row.append("%.3g" % Sx_3.real)
        row.append("%.3g" % Sy_3.real)
        row.append("%.3g" % Sz_3.real)

        m_n = map(lambda n: s.expect_1s(z_ss, n).real, xrange(1, N + 1)) #Magnetization
        m = sp.sum(m_n)

        row.append("%.9g" % m)
        M.append(m)

        """
        Switch to real time evolution if we have the ground state.
        """
        if (not real_time and eta < tol_im):
            real_time = True
            s.save_state(grnd_fname)
            s.ham = get_ham(N, J, h_quench)
            T = Tre
            H = Hre
            M = Mre
            i = 0
            t = 0
            print 'Starting real time evolution!'

        """
        Carry out next step!
        """
        if not real_time:
            s.take_step(step)
            t += 1.j * step
        else:
            s.take_step_RK4(realstep * 1.j)
            t += realstep

        eta = s.eta.real.sum()
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
            M_tau.set_xlabel('tau')
            M_tau.set_ylabel('M_x')
            M_tau.set_title('Imaginary time evolution: Magnetization')

            H_tau.plot(tau, Him)
            M_tau.plot(tau, Mim)

        #Now plot the real time evolution of K1 and Mx
        t = Tre
        fig3 = plt.figure(3)
        fig4 = plt.figure(4)

        H_t = fig3.add_subplot(111)
        H_t.set_xlabel('t')
        H_t.set_ylabel('H')
        H_t.set_title('Real time evolution: Energy')
        M_t = fig4.add_subplot(111)
        M_t.set_xlabel('t')
        M_t.set_ylabel('M_x')
        M_t.set_title('Real time evolution: Magnetization')

        H_t.plot(t, Hre)
        M_t.plot(t, Mre)

        plt.show()
