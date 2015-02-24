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

def get_E_crit(N):
    return - 2 * abs(sp.sin(sp.pi * (2 * sp.arange(N) + 1) / (2 * (2 * N + 1)))).sum()

class TestOps(unittest.TestCase):
    def test_ising_crit_im_tdvp(self):
        N = 5
        D = 8
        
        s = tdvp.EvoMPS_TDVP_Generic(N, [D] * (N + 1), [2] * (N + 1), get_ham(N, 1.0, 1.0))
        
        E = get_E_crit(N)
        
        tol = 1E-5 #Should result in correct energy to ~1E-12
        
        eta = 1
        while eta > tol:
            s.update()
            H = s.H_expect
            s.take_step(0.1)
            eta = s.eta.real.sum()
            
        s.update()
            
        self.assertTrue(sp.allclose(E, H))
        
        self.assertLessEqual(s.expect_1s(x_ss, 1), 10 * tol)
        
        self.assertLessEqual(s.expect_1s(y_ss, 1), 10 * tol)
        
    def test_ising_crit_im_tdvp_RK4(self):
        N = 5
        D = 8
        
        s = tdvp.EvoMPS_TDVP_Generic(N, [D] * (N + 1), [2] * (N + 1), get_ham(N, 1.0, 1.0))
        
        E = get_E_crit(N)
        
        tol = 1E-5 #Should result in correct energy to ~1E-12
        
        eta = 1
        while eta > tol:
            s.update()
            H = s.H_expect
            s.take_step_RK4(0.1)
            eta = s.eta.real.sum()
            
        self.assertTrue(sp.allclose(E, H))
        
    def test_ising_crit_im_tdvp_split_step(self):
        N = 5
        D = 8
        
        s = tdvp.EvoMPS_TDVP_Generic(N, [D] * (N + 1), [2] * (N + 1), get_ham(N, 1.0, 1.0))
        
        E = get_E_crit(N)
        
        tol = 1E-5 #Should result in correct energy to ~1E-12
        
        eta = 1
        while eta > tol:
            s.update()
            H = s.H_expect
            if eta < 1E-2:
                s.take_step_split(0.1)
            else:
                s.take_step(0.1)
            eta = s.eta.real.sum()
            
        self.assertTrue(sp.allclose(E, H))
        
    def test_ising_crit_im_tdvp_DMRG(self):
        N = 5
        D = 8
        
        s = tdvp.EvoMPS_TDVP_Generic(N, [D] * (N + 1), [2] * (N + 1), get_ham(N, 1.0, 1.0))
        
        E = get_E_crit(N)
        
        tol = 1E-5 #Should result in correct energy to ~1E-12
        
        eta = 1
        itr = 0
        while eta > tol:
            s.update()
            H = s.H_expect
            if itr % 10 == 9:
                s.vari_opt_ss_sweep()
            else:
                s.take_step(0.1)            
            eta = s.eta.real.sum()
            itr += 1
            
        self.assertTrue(sp.allclose(E, H))
        
    def test_ising_crit_im_tdvp_auto_trunc(self):
        N = 10
        D = 16
        
        s = tdvp.EvoMPS_TDVP_Generic(N, [D] * (N + 1), [2] * (N + 1), get_ham(N, 1.0, 0.5))
        
        tol = 1E-5 #Should result in correct energy to ~1E-12
        
        eta = 1
        itr = 0
        while eta > tol:
            s.update(auto_truncate=True)
            if eta < 1E-3 and itr % 10 == 9:
                s.vari_opt_ss_sweep(ncv=4)
            else:
                s.take_step(0.08)
            eta = s.eta.real.sum()
            itr += 1
            
        self.assertTrue(s.D[N/2] < D)
        
    def test_ising_crit_im_tdvp_dynexp(self):
        N = 8
        D = 1
        
        s = tdvp.EvoMPS_TDVP_Generic(N, [D] * (N + 1), [2] * (N + 1), get_ham(N, 1.0, 1.0))
        
        E = get_E_crit(N)
        
        tol = 1E-5 #Should result in correct energy to ~1E-12
        
        eta = 1
        itr = 0
        while eta > tol:
            s.update()
            H = s.H_expect
            s.take_step(0.1, dynexp=True, dD_max=1)
            eta = s.eta.real.sum()
            itr += 1
            
        self.assertTrue(sp.allclose(E, H))

        
if __name__ == '__main__':
    unittest.main()