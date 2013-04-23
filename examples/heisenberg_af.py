#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
A demonstration of evoMPS by simulation of quench dynamics
for the transverse Ising model.

@author: Ashley Milsted
"""

import scipy as sp
import scipy.linalg as la
import matplotlib.pyplot as plt
import math as ma

import evoMPS.tdvp_gen as tdvp_gen

x_ss_s1 = ma.sqrt(0.5) * sp.array([[0, 1, 0], 
                                   [1, 0, 1], 
                                   [0, 1, 0]])
y_ss_s1 = ma.sqrt(0.5) * 1.j * sp.array([[0, 1, 0], 
                                         [-1, 0, 1], 
                                         [0, -1, 0]])
z_ss_s1 = sp.array([[1, 0, 0], 
                    [0, 0, 0], 
                    [0, 0, -1]])
                    
x_ss_pauli = sp.array([[0, 1], 
                       [1, 0]])
y_ss_pauli = 1.j * sp.array([[0, -1], 
                             [1, 0]])
z_ss_pauli = sp.array([[1, 0], 
                       [0, -1]])

def get_ham(S, Jx, Jy, Jz):
    if S == 1:
        return (Jx * sp.kron(x_ss_s1, x_ss_s1) 
                + Jy * sp.kron(y_ss_s1, y_ss_s1)
                + Jz * sp.kron(z_ss_s1, z_ss_s1)).reshape(3, 3, 3, 3)
    elif S == 0.5:
        return (Jx * sp.kron(x_ss_pauli, x_ss_pauli) 
                + Jy * sp.kron(y_ss_pauli, y_ss_pauli)
                + Jz * sp.kron(z_ss_pauli, z_ss_pauli)).reshape(2, 2, 2, 2)
    else:
        return None

"""
Choose spin-1 or spin-1/2.
"""
q = 0
S = 1

if S == 0.5:
    q = 2
    z_ss = z_ss_pauli
    y_ss = y_ss_pauli
    x_ss = x_ss_pauli
elif S == 1:
    q = 3
    z_ss = z_ss_s1
    y_ss = y_ss_s1
    x_ss = x_ss_s1
else:
    print "Only S = 1 or S = 1/2 are supported!"
    exit()

"""
Next, we set up some global variables to be used as parameters to 
the evoMPS class.
"""

N = 16 #The length of the finite spin chain.


"""
The bond dimension for each site is given as a vector, length N.
Here we set the bond dimension = bond_dim for all sites.
"""
bond_dim = 32 #The maximum bond dimension


"""
Set the initial Hamiltonian parameters.
"""
J = 1

"""
Now we are ready to create an instance of the evoMPS class.
"""
s = tdvp_gen.EvoMPS_TDVP_Generic(N, [bond_dim] * (N + 1), 
                                 [q] * (N + 1), 
                                 [get_ham(S, J, J, J)] * N)

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
tol_im = 1E-4
total_steps = 2000

"""
The following handles loading the ground state from a file.
The ground state will be saved automatically when it is declared found.
If this script is run again with the same settings, an existing
ground state will be loaded, if present.
"""
grnd_fname = "heis_af_N%d_D%d_q%d_J%g_s%g_dtau%g_ground.npy" % (N, q, bond_dim, J, tol_im, step)

try:
   a_file = open(grnd_fname, 'rb')
   s.load_state(a_file)
   a_file.close
   real_time = True
   loaded = True
   print 'Using saved ground state: ' + grnd_fname
except IOError as e:
   print 'No existing ground state could be opened.'
   real_time = False
   loaded = False

"""
Prepare some loop variables and some vectors to hold data from each step.
"""
t = 0. + 0.j
imsteps = 0

reCF = []
reNorm = []

T = sp.zeros((total_steps), dtype=sp.complex128)
K1 = sp.zeros((total_steps), dtype=sp.complex128)
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
col_heads = ["Step", "t", "H", "dH", 
             "sig_x_3", "sig_y_3", "sig_z_3",
             "E_vn_3,4", "M_x", "eta"] #These last three are for testing the midpoint method.
print "\t".join(col_heads)
print
eta = 1
for i in xrange(total_steps):
    T[i] = t
    
    row = [str(i)]
    row.append(str(t))
    
    s.update()
        
    K1[i] = s.H_expect
    row.append("%.15g" % K1[i].real)
    
    if i > 0:        
        dK1 = K1[i].real - K1[i - 1].real
    else:
        dK1 = K1[i]
    
    row.append("%.2e" % (dK1.real))
        
    """
    Compute obserables!
    """
    
    Sx_3[i] = s.expect_1s(x_ss, 10) #Spin observables for site 3.
    Sy_3[i] = s.expect_1s(y_ss, 10)
    Sz_3[i] = s.expect_1s(z_ss, 10)
    row.append("%.3g" % Sx_3[i].real)
    row.append("%.3g" % Sy_3[i].real)
    row.append("%.3g" % Sz_3[i].real)
    
#    rho_34 = s.density_2s(3, 4) #Reduced density matrix for sites 3 and 4.
#    E_v = -sp.trace(sp.dot(rho_34, la.logm(rho_34)/sp.log(2))) #The von Neumann entropy.
    
#    row.append("%.9g" % E_v.real)
    
    m = 0   #x-Magnetization
    for n in xrange(1, N + 1):
        m += s.expect_1s(x_ss, n) 
        
    row.append("%.9g" % m.real)
    Mx[i] = m
    
    """
    Switch to real time evolution if we have the ground state.
    """
    if loaded or (not real_time and eta < tol_im):
        real_time = True
        s.save_state(grnd_fname)
        s.ham = [get_ham(S, J, J_real, J)] * N
        step = realstep * 1.j
        loaded = False
        print 'Starting real time evolution!'

    
    """
    Carry out next step!
    """
    if not real_time:
        s.take_step(step)     
        imsteps += 1
    else:
        s.take_step_RK4(step)
    
    eta = s.eta.real.sum()
    row.append(str(eta))
    print "\t".join(row)
    
    t += 1.j * sp.conj(step)

"""
Simple plots of the results.
"""

if imsteps > 0: #Plot imaginary time evolution of K1 and Mx
    tau = T.imag[0:imsteps]
    
    fig1 = plt.figure(1)
    fig2 = plt.figure(2) 
    K1_tau = fig1.add_subplot(111)
    K1_tau.set_xlabel('tau')
    K1_tau.set_ylabel('H')
    M_tau = fig2.add_subplot(111)
    M_tau.set_xlabel('tau')
    M_tau.set_ylabel('M_x')    
    
    K1_tau.plot(tau, K1.real[0:imsteps])
    M_tau.plot(tau, Mx.real[0:imsteps])

#Now plot the real time evolution of K1 and Mx
t = T.real[imsteps + 1:]
fig3 = plt.figure(3)
fig4 = plt.figure(4)

K1_t = fig3.add_subplot(111)
K1_t.set_xlabel('t')
K1_t.set_ylabel('H')
M_t = fig4.add_subplot(111)
M_t.set_xlabel('t')
M_t.set_ylabel('M_x')

K1_t.plot(t, K1.real[imsteps + 1:])
M_t.plot(t, Mx.real[imsteps + 1:])

plt.show()
