#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
A demonstration of evoMPS by simulation of quench dynamics
for the transverse Ising model.

@author: Ashley Milsted
"""

from scipy import *
import scipy.linalg as la
import matplotlib.pyplot as plt

from evoMPS import *

"""
First, we define our Hamiltonian and some observables.
"""

def h_nn(n, s, t, u, v):
    """The nearest neighbour Hamiltonian representing the interaction.

    The global variable J determines the strength.
    """
    res = x_ss(n, s, u) * x_ss(n, t, v)
    res += y_ss(n, s, u) * y_ss(n, t, v)
    res += z_ss(n, s, u) * z_ss(n, t, v)    
        
    return J * 1.0 * res
    
def z_ss(n, s, t):
    """Spin observable: z-direction
    """
    if s == t:
        return (-1.0)**s
    else:
        return 0
        
def x_ss(n, s, t):
    """Spin observable: x-direction
    """
    if s == t:
        return 0
    else:
        return 1.0
        
def y_ss(n, s, t):
    """Spin observable: y-direction
    """
    if s == t:
        return 0
    else:
        return (1.j * (-1.0)**t)

"""
Next, we set up some global variables to be used as parameters to 
the evoMPS class.
"""

N = 20 #The length of the finite spin chain.


"""
The bond dimension for each site is given as a vector, length N.
Here we set the bond dimension = bond_dim for all sites.
"""
bond_dim = 16 #The maximum bond dimension

D = empty(N + 1, dtype=int32)
D.fill(bond_dim)


"""
The site Hilbert space dimension is also given as a vector, length N.
Here, we set all sites to dimension = qn.
"""
qn = 2 #The site dimension

q = empty(N + 1, dtype=int32)
q.fill(qn)

"""
Now we are ready to create an instance of the evoMPS class.
"""
s = tdvp_gen.evoMPS_TDVP_Generic(N, D, q)

"""
Tell evoMPS about our Hamiltonian.
"""
s.h_nn = h_nn

"""
Set the initial Hamiltonian parameters.
"""
J = 1

"""
We're going to simulate a quench after we find the ground state.
Set the new J parameter for the real time evolution here.
"""
J_real = 2

"""
Now set the step sizes for the imaginary and the real time evolution.
These are currently fixed.
"""
step = 0.05
realstep = 0.01

"""
Now set the tolerance for the imaginary time evolution.
When the change in the energy falls below this level, the
real time simulation of the quench will begin.
"""
tol_im = 5E-15
total_steps = 500

"""
The following handles loading the ground state from a file.
The ground state will be saved automatically when it is declared found.
If this script is run again with the same settings, an existing
ground state will be loaded, if present.
"""
grnd_fname = "heis_af_N%d_D%d_q%d_J%g_s%g_dtau%g_ground.npy" % (N, qn, bond_dim, J, tol_im, step)

try:
   a_file = open(grnd_fname, 'rb')
   s.LoadState(a_file)
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

T = zeros((total_steps), dtype=complex128)
K1 = zeros((total_steps), dtype=complex128)
lN = zeros((total_steps), dtype=complex128)

Sx_3 = zeros((total_steps), dtype=complex128) #Observables for site 3.
Sy_3 = zeros((total_steps), dtype=complex128)
Sz_3 = zeros((total_steps), dtype=complex128)

Mx = zeros((total_steps), dtype=complex128)   #Magnetization in x-direction.
   
   
"""
Print a table header.
"""
print "Bond dimensions: " + str(s.D)
print
col_heads = ["Step", "t", "l[N]", "Restore CF?", "Renorm?", "K[1]", "dK[1]", 
             "sig_x_3", "sig_y_3", "sig_z_3",
             "E_vn_3,4", "M_x", "Next step",
             "(itr", "delta", "delta_chk)"] #These last three are for testing the midpoint method.
print "\t".join(col_heads)
print

for i in xrange(total_steps):
    T[i] = t
    
    row = [str(i)]
    row.append(str(t))
    
    s.Upd_r()
    s.Upd_l()

    lN[i] = s.l[N][0, 0]
    row.append("%.3g" % lN[i].real)

    restoreCF = (i % 4 == 0) #Restore canonical form every 4 steps.
    reCF.append(restoreCF)
    if restoreCF:
        s.Restore_ON_R()
        row.append("Yes")
    else:
        row.append("No")
    
    #Renormalize if the norm is drifting.
    reNormalize = not allclose(s.l[N][0, 0], 1., atol=s.eps, rtol=0)
    reNorm.append(reNormalize)
    if reNormalize:
        row.append("True")
        s.Simple_renorm()
    else:
        row.append("False")
    
    s.BuildC()    
    s.CalcK()
        
    K1[i] = s.K[1][0, 0]    
    row.append("%.15g" % K1[i].real)
    
    if i > 0:        
        dK1 = K1[i].real - K1[i - 1].real
    else:
        dK1 = K1[i]
    
    row.append("%.2e" % (dK1.real))
        
    """
    Compute obserables!
    """
    
    Sx_3[i] = s.Expect_SS(x_ss, 10) #Spin observables for site 3.
    Sy_3[i] = s.Expect_SS(y_ss, 10)
    Sz_3[i] = s.Expect_SS(z_ss, 10)
    row.append("%.3g" % Sx_3[i].real)
    row.append("%.3g" % Sy_3[i].real)
    row.append("%.3g" % Sz_3[i].real)
    
    rho_34 = s.DensityMatrix_2S(3, 4) #Reduced density matrix for sites 3 and 4.
    E_v = -trace(dot(rho_34, la.logm(rho_34)/log(2))) #The von Neumann entropy.
    
    row.append("%.9g" % E_v.real)
    
    m = 0   #x-Magnetization
    for n in xrange(1, N + 1):
        m += s.Expect_SS(x_ss, n) 
        
    row.append("%.9g" % m.real)
    Mx[i] = m
    
    """
    Switch to real time evolution if we have the ground state.
    """
#    if loaded or (not real_time and abs(dK1) < tol_im):
#        real_time = True
#        s.SaveState(grnd_fname)
#        J = J_real
#        step = realstep * 1.j
#        loaded = False
#        print 'Starting real time evolution!'
    
    row.append(str(1.j * conj(step)))
    
    """
    Carry out next step!
    """
    if not real_time:
        print "\t".join(row)
        s.TakeStep(step)     
        imsteps += 1
    elif False: #Midpoint method. Currently disabled. Change to True to test!
        itr, delta, delta_check = s.TakeStep_BEuler(step)
        row.append(str(itr))
        row.append("%.3g" % delta.real)
        row.append("%.3g" % delta_check.real)
        print "\t".join(row)
    else:
        print "\t".join(row)
        s.TakeStep_RK4(step)
    
    t += 1.j * conj(step)

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
