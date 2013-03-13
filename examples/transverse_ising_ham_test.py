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

import evoMPS.tdvp_gen as tdvp

"""
First, we define our Hamiltonian and some observables.
"""    
        
x_ss = sp.array([[0, 1], [1, 0]])
y_ss = 1.j * sp.array([[0, -1], [1, 0]])
z_ss = sp.array([[1, 0], [0, -1]])

def get_ham(J, h):
    return (-J * sp.kron(x_ss, x_ss) - h * sp.kron(z_ss, sp.eye(2))
           ).reshape(2, 2, 2, 2)

def get_ham_end(J, h):
    return get_ham(J, h) - h * sp.kron(sp.eye(2), z_ss).reshape(2, 2, 2, 2)

def get_ham_3s(J, h):
    res = (-J * sp.kron(sp.kron(x_ss, x_ss), sp.eye(2)) 
            - h * sp.kron(sp.kron(z_ss, sp.eye(2)), sp.eye(2))
           )
    return res.reshape(2, 2, 2, 2, 2, 2)

def get_ham_end_3s(J, h):
    return get_ham_3s(J, h) + (-h * sp.kron(sp.kron(sp.eye(2), z_ss), sp.eye(2))
                            -h * sp.kron(sp.kron(sp.eye(2), sp.eye(2)), z_ss)
                            -J * sp.kron(sp.kron(sp.eye(2), x_ss), x_ss)
                           ).reshape(2, 2, 2, 2, 2, 2)

"""
Next, we set up some global variables to be used as parameters to 
the evoMPS class.
"""

N = 10 #The length of the finite spin chain.


"""
The bond dimension for each site is given as a vector, length N.
Here we set the bond dimension = bond_dim for all sites.
"""
bond_dim = 32 #The maximum bond dimension

D = sp.empty(N + 1, dtype=sp.int32)
D.fill(bond_dim)


"""
The site Hilbert space dimension is also given as a vector, length N.
Here, we set all sites to dimension = qn.
"""
qn = 2 #The site dimension

q = sp.empty(N + 1, dtype=sp.int32)
q.fill(qn)

"""
Set the initial Hamiltonian parameters.
"""
h = -1.00
J = 1.0

if h == -J:
    E = 0
    for n in xrange(N):
        E += 2 * abs(sp.sin(sp.pi * (2 * n + 1) / (2 * (2 * N + 1))))
    print "Exact energy = %.15g" % E

#ham = [get_ham(J, h)] * (N - 1) + [get_ham_end(J, h)]
ham = [get_ham_3s(J, h)] * (N - 2) + [get_ham_end_3s(J, h)]

"""
Now we are ready to create an instance of the evoMPS class.
"""
s = tdvp.EvoMPS_TDVP_Generic(N, D, q, ham)

s.randomize()

step = 0.1
tol = 1E-10

"""
The following handles loading the ground state from a file.
The ground state will be saved automatically when it is declared found.
If this script is run again with the same settings, an existing
ground state will be loaded, if present.
"""
grnd_fname = "t_ising_N%d_D%d_q%d_J%g_h%g_s%g_dtau%g_ground.npy" % (N, qn, bond_dim, J, h, tol, step)

loaded = False

if False:
    try:
        a_file = open(grnd_fname, 'rb')
        s.load_state(a_file)
        a_file.close
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


print "Bond dimensions: " + str(s.D)
print
col_heads = ["t", "eta", "H", "dH", 
             "sig_x_3", "sig_y_3", "sig_z_3",
             "M_x", "Next step"]
print "\t".join(col_heads)
print

eta = 10000
prevH = 0
while (eta > tol):    
    row = [str(t)]
    
    s.update()    
    
    row.append("%.8g" % eta)
        
    row.append("%.15g" % s.H_expect.real)
    
    row.append("%.2e" % (s.H_expect.real - prevH))
    
    prevH = s.H_expect.real
        
    """
    Compute obserables!
    """
    
    Sx_3 = s.expect_1s(x_ss, 3) #Spin observables for site 3.
    Sy_3 = s.expect_1s(y_ss, 3)
    Sz_3 = s.expect_1s(z_ss, 3)
    row.append("%.3g" % Sx_3.real)
    row.append("%.3g" % Sy_3.real)
    row.append("%.3g" % Sz_3.real)
    
    
    m = 0   #x-Magnetization
    for n in xrange(1, N + 1):
        m += s.expect_1s(x_ss, n) 
        
    row.append("%.9g" % m.real)    
    
    row.append(str(1.j * sp.conj(step)))
    
    print "\t".join(row)
    s.take_step(step)
    
    eta = s.eta.real.sum()
    
    imsteps += 1

    t += 1.j * sp.conj(step)

