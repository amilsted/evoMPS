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

import evoMPS.tdvp_uniform as tdvp

"""
First, we define our Hamiltonian and some observables.
"""

x_ss = sp.array([[0, 1], 
                 [1, 0]])
y_ss = 1.j * sp.array([[0, -1], 
                       [1, 0]])
z_ss = sp.array([[1, 0], 
                 [0, -1]])

def get_ham(J, h):
    ham = -J * (sp.kron(x_ss, x_ss) + h * sp.kron(z_ss, sp.eye(2))).reshape(2, 2, 2, 2)
    return ham

"""
Next, we set up some global variables to be used as parameters to 
the evoMPS class.
"""

D = 32 #The bond dimension
q = 2 #The site dimension

"""
Set the initial Hamiltonian parameters.
"""
h = -0.50
J = 1.00

"""
Now we are ready to create an instance of the evoMPS class.
"""
s = tdvp.EvoMPS_TDVP_Uniform(D, q, get_ham(J, h))

"""
We're going to simulate a quench after we find the ground state.
Set the new J parameter for the real time evolution here.
"""
h_real = -1.5

"""
Now set the step sizes for the imaginary and the real time evolution.
These are currently fixed.
"""
step = 0.1
realstep = 0.01

"""
Now set the tolerance for the imaginary time evolution.
When the state tolerance falls below this level, the
real time simulation of the quench will begin.
"""
tol_im = 1E-9
num_realtime_steps = 200

"""
The following handles loading the ground state from a file.
The ground state will be saved automatically when it is declared found.
If this script is run again with the same settings, an existing
ground state will be loaded, if present.
"""

broken_left = False

grnd_fname_fmt = "t_ising_uni_D%d_q%d_J%g_h%g_s%g_dtau%g_ground_%u.npy"

grnd_fname = grnd_fname_fmt % (D, q, J, h, tol_im, step, int(broken_left))

expand = False

if False:
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
    real_time = False
    
s.sanity_checks = False
s.symm_gauge = True

if __name__ == "__main__":
    """
    Prepare some loop variables and some vectors to hold data from each step.
    """
    t = 0. + 0.j
    
    reCF = []
    reNorm = []
    
    T = []
    E = []
    lN = []
    
    Sx = []
    Sy = []
    Sz = []
    
    Mx = []   #Magnetization in x-direction.
       
       
    """
    Print a table header.
    """
    print "Bond dimensions: " + str(s.D)
    print
    col_heads = ["Step", "t", "eta", "H", "dH", 
                 "sig_x", "sig_y", "sig_z", "entr.",
                 "Next step"] #These last three are for testing the midpoint method.
    print "\t".join(col_heads)
    print
    
    im_steps = 0
    realtime_steps = 0
    while realtime_steps < num_realtime_steps:
        T.append(t)
        
        row = [str(im_steps + realtime_steps)]
        row.append(str(t))
        
        eta = s.eta.real
        row.append("%.4g" % eta)
        
        s.update(auto_truncate=not real_time)
        
        E.append(s.h)
        row.append("%.15g" % E[-1].real)
        
        if len(E) > 1:
            dE = E[-1].real - E[-2].real
        else:
            dE = E[-1]
        
        row.append("%.2e" % (dE.real))
            
        """
        Compute obserables!
        """
        
        Sx.append(s.expect_1s(x_ss))
        Sy.append(s.expect_1s(y_ss))
        Sz.append(s.expect_1s(z_ss))
        row.append("%.3g" % Sx[-1].real)
        row.append("%.3g" % Sy[-1].real)
        row.append("%.3g" % Sz[-1].real)
        
        entr = s.S_hc
        row.append("%.3g" % entr.real)
        
        """
        Switch to real time evolution if we have the ground state.
        """
        if expand and (loaded or (not real_time and im_steps > 1 and eta < tol_im)):
            grnd_fname = grnd_fname_fmt % (D, q, J, h, tol_im, step)        
            
            if not loaded:
                s.save_state(grnd_fname)
            
            D = D * 2
            print "***MOVING TO D = " + str(D) + "***"
            s.expand_D(D)
            s.update()
            
            loaded = False
        elif loaded or (not real_time and im_steps > 1 and eta < tol_im):
            real_time = True
            
            if abs(h/J) < 1:
                broken_left = Sx[-1] > 0
                print "Broken left: " + str(broken_left)
            
            grnd_fname = grnd_fname_fmt % (D, q, J, h, tol_im, step, int(broken_left))  
            
            s.save_state(grnd_fname)
            s.ham = get_ham(J, h_real)
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
            im_steps += 1
        else:
            print "\t".join(row)
            s.take_step_RK4(step)
            realtime_steps += 1
        
        t += 1.j * sp.conj(step)
    
    """
    Simple plots of the results.
    """
    
    T = sp.array(T)
    E = sp.array(E)
    Sx = sp.array(Sx)
    
    if im_steps > 0: #Plot imaginary time evolution of K1 and Mx
        tau = T.imag[0:im_steps]
        
        fig1 = plt.figure(1)
        fig2 = plt.figure(2) 
        K1_tau = fig1.add_subplot(111)
        K1_tau.set_xlabel('tau')
        K1_tau.set_ylabel('H')
        M_tau = fig2.add_subplot(111)
        M_tau.set_xlabel('tau')
        M_tau.set_ylabel('M_x')    
        
        K1_tau.plot(tau, E.real[0:im_steps])
        M_tau.plot(tau, Sx.real[0:im_steps])
    
    #Now plot the real time evolution of K1 and Mx
    t = T.real[im_steps + 1:]
    fig3 = plt.figure(3)
    fig4 = plt.figure(4)
    
    K1_t = fig3.add_subplot(111)
    K1_t.set_xlabel('t')
    K1_t.set_ylabel('H')
    M_t = fig4.add_subplot(111)
    M_t.set_xlabel('t')
    M_t.set_ylabel('M_x')
    
    K1_t.plot(t, E.real[im_steps + 1:])
    M_t.plot(t, Sx.real[im_steps + 1:])
    
    plt.show()
