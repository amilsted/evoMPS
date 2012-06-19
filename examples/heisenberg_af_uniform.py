#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
A demonstration of evoMPS by simulation of quench dynamics
for the transverse Ising model.

@author: Ashley Milsted
"""

import math as ma
import scipy as sp
import matplotlib.pyplot as plt

import evoMPS.tdvp_uniform as tdvp

"""
First, we define our Hamiltonian and some observables.
"""

def h_nn(s, t, u, v):
    """The nearest neighbour Hamiltonian representing the interaction.

    The global variable J determines the strength.
    """        
    res = Jx * x_ss(s, u) * x_ss(t, v)
    res += Jy * y_ss(s, u) * y_ss(t, v)
    res += Jz * z_ss(s, u) * z_ss(t, v)               
        
    return res        
    

def z_ss_pauli(s, t):
    """Spin observable: z-direction
    """
    if s == t:
        return (-1.0)**s
    else:
        return 0
        
def x_ss_pauli(s, t):
    """Spin observable: x-direction
    """
    if s == t:
        return 0
    else:
        return 1.0
        
def y_ss_pauli(s, t):
    """Spin observable: y-direction
    """
    if s == t:
        return 0
    else:
        return (1.j * (-1.0)**t)
        
def z_ss_s1(s, t):
    """Spin observable: z-direction
    """
    if s == t:
        if s == 1:
            return 0
        else:
            return 1 * (-1.0)**(s/2)
    else:
        return 0
        
def x_ss_s1(s, t):
    """Spin observable: x-direction
    """
    if s == t or abs(s - t) == 2:
        return 0
    else:
        return ma.sqrt(0.5)
        
def y_ss_s1(s, t):
    """Spin observable: y-direction
    """
    if s == t or abs(s - t) == 2:
        return 0
    elif s > t:
        return -1.j * ma.sqrt(0.5)
    else:
        return 1.j * ma.sqrt(0.5)

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

D = 32


s = tdvp.EvoMPS_TDVP_Uniform(D, q)


s.h_nn = h_nn

"""
Set the Hamiltonian parameters.
"""
Jx = Jy = Jz = 1

"""
Set the parameters for the quench.
"""
Jx_2 = Jy_2 = 1
Jz_2 = -1

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
total_steps = 100

s.itr_atol = 1E-12
s.itr_rtol = 1E-11

"""
The following handles loading the ground state from a file.
The ground state will be saved automatically when it is declared found.
If this script is run again with the same settings, an existing
ground state will be loaded, if present.
"""
grnd_fname_fmt = "heis_af_uni_D%d_q%d_Jx%g_Jy%g_Jz%g_s%g_dtau%g_ground.npy"

grnd_fname = grnd_fname_fmt % (D, q, Jx, Jy, Jz, tol_im, step)

load_state = True
expand = False
real_time = False

if load_state:
    try:
       a_file = open(grnd_fname, 'rb')
       s.load_state(a_file)
       a_file.close
       real_time = not expand
       loaded = True
       print 'Using saved ground state: ' + grnd_fname
    except IOError as e:
       print 'No existing ground state could be opened.'
       real_time = False
       loaded = False
else:
    loaded = False
        
if __name__ == "__main__":    
    step = 0.1
    
    """
    Prepare some loop variables and some vectors to hold data from each step.
    """
    t = 0. + 0.j
    imsteps = 0
    
    reCF = []
    reNorm = []
    
    T = sp.zeros((total_steps), dtype=sp.complex128)
    E = sp.zeros((total_steps), dtype=sp.complex128)
    lN = sp.zeros((total_steps), dtype=sp.complex128)
    
    Sx = sp.zeros((total_steps), dtype=sp.complex128)
    Sy = sp.zeros((total_steps), dtype=sp.complex128)
    Sz = sp.zeros((total_steps), dtype=sp.complex128)
    
    """
    Print a table header.
    """
    print "Bond dimensions: " + str(s.D)
    print
    col_heads = ["Step", "t", "eta", "Restore CF?", "energy: h/J", 
                 "difference", 
                 "sig_x", "sig_y", "sig_z", "conv_l", "conv_r", 
                 "Next step"]
    print "\t".join(col_heads)
    print
    
    s.symm_gauge = True
    
    for i in xrange(total_steps):
        T[i] = t
        
        row = [str(i)]
        row.append(str(t))
        
        eta = s.eta.real
        row.append("%.4g" % eta)
        
        s.update()  
            
        E[i] = s.h / Jx #Note: J_x sets the scale!
        row.append("%.15g" % E[i].real)
        
        if i > 0:        
            dE = E[i].real - E[i - 1].real
        else:
            dE = E[i]
        
        row.append("%.2e" % (dE.real))
            
        """
        Compute obserables!
        """
        
        Sx[i] = s.expect_1s(x_ss) #Spin observables for site 3.
        Sy[i] = s.expect_1s(y_ss)
        Sz[i] = s.expect_1s(z_ss)
        row.append("%.3g" % Sx[i].real)
        row.append("%.3g" % Sy[i].real)
        row.append("%.3g" % Sz[i].real)
        
        row.append(str(s.conv_l))
        row.append(str(s.conv_r))
        
        """
        Switch to real time evolution if we have the ground state.
        """
        if expand and (loaded or (not real_time and i > 1 and eta < tol_im)):
            grnd_fname = grnd_fname_fmt % (D, q, Jx, Jy, Jz, tol_im, step)        
            
            if not loaded:
                s.save_state(grnd_fname)
            
            if i > 0:
                D = D * 2
                print "***MOVING TO D = " + str(D) + "***"
                s.expand_D(D)
                s.update()
            
            loaded = False
        elif loaded or (not real_time and i > 1 and eta < tol_im):
            real_time = True
            
            s.save_state(grnd_fname)
            
            Jx = Jx_2
            Jy = Jy_2
            Jz = Jz_2
            
            step = realstep * 1.j
            loaded = False
            print 'Quenched parameters, starting real time evolution!'
        
        row.append(str(1.j * sp.conj(step)))
        
        if i > 2 and max(1E-14, abs(dE * Jx)) < s.itr_rtol * 10:
            s.itr_atol = abs(dE * Jx) / 50
            s.itr_rtol = abs(dE * Jx) / 50    
        
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
        K1_tau = fig1.add_subplot(111)
        K1_tau.set_xlabel('tau')
        K1_tau.set_ylabel('H')
        S_tau = fig2.add_subplot(111)
        S_tau.set_xlabel('tau')
        S_tau.set_ylabel('S_x')    
        
        K1_tau.plot(tau, E.real[0:imsteps])
        S_tau.plot(tau, Sx.real[0:imsteps])
    
    #Now plot the real time evolution of K1 and Mx
    t = T.real[imsteps + 1:]
    fig3 = plt.figure(3)
    fig4 = plt.figure(4)
    
    K1_t = fig3.add_subplot(111)
    K1_t.set_xlabel('t')
    K1_t.set_ylabel('H')
    S_t = fig4.add_subplot(111)
    S_t.set_xlabel('t')
    S_t.set_ylabel('S_x')
    
    K1_t.plot(t, E.real[imsteps + 1:])
    S_t.plot(t, Sx.real[imsteps + 1:])
    
    plt.show()
