#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
A demonstration of evoMPS by simulation of quench dynamics
for the transverse Ising model.

@author: Ashley Milsted
"""

import scipy as sp
import evoMPS.tdvp_gen as tdvp
import unittest

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

class TestOps(unittest.TestCase):
    def test_ising_crit_im_tdvp(self):
        N = 7 
        
        s = tdvp.EvoMPS_TDVP_Generic(N, [1024] * (N + 1), [2] * (N + 1), get_ham(N, 1.0, 1.0))
        
        E = - 2 * abs(sp.sin(sp.pi * (2 * sp.arange(N) + 1) / (2 * (2 * N + 1)))).sum()
        print "Exact ground state energy = %.15g" % E
        
        tol = 1E-7
        
        eta = 1
        while eta > tol:
            s.update()
            H = s.H_expect
            s.take_step(0.1)
            eta = s.eta.real.sum()
            
        print "Imaginary time result: ", H.real
        self.assertTrue(sp.allclose(E, H))