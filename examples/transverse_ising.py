#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
A demonstration of evoMPS by simulation of quench dynamics
for the transverse Ising model.

@author: Ashley Milsted
"""

import scipy as sp
import matplotlib.pyplot as plt

import evoMPS.tdvp_gen as tdvp

"""
First, we define our Hamiltonian and some observables.
"""
x_ss = sp.array([[0, 1], 
                 [1, 0]])
y_ss = 1.j * sp.array([[0, -1], 
                       [1, 0]])
z_ss = sp.array([[1, 0], 
                 [0, -1]])

def get_ham(N, J, h):
    ham = -J * (sp.kron(x_ss, x_ss) + h * sp.kron(z_ss, sp.eye(2))).reshape(2, 2, 2, 2)
    ham_end = ham + h * sp.kron(sp.eye(2), z_ss).reshape(2, 2, 2, 2)
    return [None] + [ham] * (N - 2) + [ham_end] 

"""
Next, we set up some global variables to be used as parameters to 
the evoMPS class.
"""

N = 7 #The length of the finite spin chain.


"""
The bond dimension for each site is given as a vector, length N + 1.
Here we set the bond dimension = bond_dim for all sites.
"""
bond_dim = 8 #The maximum bond dimension

D = sp.empty(N + 1, dtype=sp.int32)
D.fill(bond_dim)


"""
The site Hilbert space dimension is also given as a vector, length N + 1.
Here, we set all sites to dimension = qn.
"""
qn = 2 #The site dimension

q = sp.empty(N + 1, dtype=sp.int32)
q.fill(qn)

"""
Set the initial Hamiltonian parameters.
"""
h = 1.00
J = 1.00

"""
Now we are ready to create an instance of the evoMPS class.
"""
s = tdvp.EvoMPS_TDVP_Generic(N, D, q, get_ham(N, J, h))

if h == J:
    E = - 2 * abs(sp.sin(sp.pi * (2 * sp.arange(N) + 1) / (2 * (2 * N + 1)))).sum()
    print "Exact ground state energy = %.15g" % E


"""
We're going to simulate a quench after we find the ground state.
Set the new J parameter for the real time evolution here.
"""
J_real = 2

"""
Now set the step sizes for the imaginary and the real time evolution.
These are currently fixed.
"""
step = 0.1
realstep = 0.01

"""
Now set the tolerance for the imaginary time evolution.
When the change in the energy falls below this level, the
real time simulation of the quench will begin.
"""
tol_im = 1E-10
total_steps = 1000

"""
The following handles loading the ground state from a file.
The ground state will be saved automatically when it is declared found.
If this script is run again with the same settings, an existing
ground state will be loaded, if present.
"""
grnd_fname = "t_ising_N%d_D%d_q%d_J%g_h%g_s%g_dtau%g_ground.npy" % (N, qn, bond_dim, J, h, tol_im, step)

real_time = False
loaded = False

if False:
    try:
        a_file = open(grnd_fname, 'rb')
        s.load_state(a_file)
        a_file.close
        real_time = True
        loaded = True
        print 'Using saved ground state: ' + grnd_fname
    except IOError as e:
        print 'No existing ground state could be opened.'


"""
Prepare some loop variables and some vectors to hold data from each step.
"""
t = 0. + 0.j
imsteps = 0

reCF = []
reNorm = []

T = sp.zeros((total_steps), dtype=sp.complex128)
H = sp.zeros((total_steps), dtype=sp.complex128)
lN = sp.zeros((total_steps), dtype=sp.complex128)

Sx_3 = sp.zeros((total_steps), dtype=sp.complex128) #Observables for site 3.
Sy_3 = sp.zeros((total_steps), dtype=sp.complex128)
Sz_3 = sp.zeros((total_steps), dtype=sp.complex128)

Mx = sp.zeros((total_steps), dtype=sp.complex128)   #Magnetization in x-direction.
   
   
"""
Print a table header.
"""
print "Bond dimensions: " + str(s.D)
print
col_heads = ["Step", "t", "eta", "<H>", "d<H>", 
             "sig_x_3", "sig_y_3", "sig_z_3",
             "E_vn_3,4", "M_x", "Next step",
             "(itr", "delta", "delta_chk)"] #These last three are for testing the midpoint method.
print "\t".join(col_heads)
print

for i in xrange(total_steps):
    T[i] = t
    
    row = [str(i)]
    row.append(str(t))
    
    s.update()
    
    eta = s.eta.real.sum()    
    row.append("%.6g" % eta)
        
    H[i] = s.H_expect
    row.append("%.15g" % H[i].real)
    
    if i > 0:        
        dH = H[i].real - H[i - 1].real
    else:
        dH = H[i]
    
    row.append("%.2e" % (dH.real))
        
    """
    Compute obserables!
    """
    
    Sx_3[i] = s.expect_1s(x_ss, 3) #Spin observables for site 3.
    Sy_3[i] = s.expect_1s(y_ss, 3)
    Sz_3[i] = s.expect_1s(z_ss, 3)
    row.append("%.3g" % Sx_3[i].real)
    row.append("%.3g" % Sy_3[i].real)
    row.append("%.3g" % Sz_3[i].real)
    
    m = 0   #x-Magnetization
    for n in xrange(1, N + 1):
        m += s.expect_1s(x_ss, n) 
        
    row.append("%.9g" % m.real)
    Mx[i] = m
    
    """
    Switch to real time evolution if we have the ground state.
    """
    if loaded or (not real_time and i > 0 and eta < tol_im):
        real_time = True
        s.save_state(grnd_fname)
        s.h_nn = get_ham(N, J_real, h)
        step = realstep * 1.j
        loaded = False
        print 'Starting real time evolution!'
    
    row.append(str(1.j * sp.conj(step)))
    
    """
    Carry out next step!
    """
    if not real_time:
        print "\t".join(row)
        s.take_step(step)     
        imsteps += 1
    else:
        print "\t".join(row)
        s.take_step_RK4(step)
    
    t += 1.j * sp.conj(step)

"""
Simple plots of the results.
"""

if imsteps > 0: #Plot imaginary time evolution of K1 and Mx
    tau = T.imag[0:imsteps]
    
    fig1 = plt.figure(1)
    fig2 = plt.figure(2) 
    H_tau = fig1.add_subplot(111)
    H_tau.set_xlabel('tau')
    H_tau.set_ylabel('H')
    M_tau = fig2.add_subplot(111)
    M_tau.set_xlabel('tau')
    M_tau.set_ylabel('M_x')    
    
    H_tau.plot(tau, H.real[0:imsteps])
    M_tau.plot(tau, Mx.real[0:imsteps])

#Now plot the real time evolution of K1 and Mx
t = T.real[imsteps + 1:]
fig3 = plt.figure(3)
fig4 = plt.figure(4)

H_t = fig3.add_subplot(111)
H_t.set_xlabel('t')
H_t.set_ylabel('H')
M_t = fig4.add_subplot(111)
M_t.set_xlabel('t')
M_t.set_ylabel('M_x')

H_t.plot(t, H.real[imsteps + 1:])
M_t.plot(t, Mx.real[imsteps + 1:])

plt.show()
