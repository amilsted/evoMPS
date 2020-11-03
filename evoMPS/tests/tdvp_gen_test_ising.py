#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
A demonstration of evoMPS by simulation of quench dynamics
for the transverse Ising model.

@author: Ashley Milsted
"""
from __future__ import absolute_import, division, print_function

import scipy as sp
import evoMPS.tdvp_gen as tdvp
import unittest
import copy

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
        for N, D in ((5, 8),(8,8)): #exact state, and also truncation  
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

            rtstep = 0.01
            s.ham = get_ham(N, 1.0, 1.1)
            s.update()
            H1 = s.H_expect
            for j in range(10):
                s.take_step(rtstep*1.j)
                s.update()
            H2 = s.H_expect
            print("N=",N," D=",D," E1=",H1," E2=",H2," difference=",abs(H1-H2))
            self.assertTrue(abs(H1-H2) < rtstep)

    def test_ising_crit_im_tdvp_RK4(self):
        for N, D in ((5, 8),(8,8)): #exact state, and also truncation  
            s = tdvp.EvoMPS_TDVP_Generic(N, [D] * (N + 1), [2] * (N + 1), get_ham(N, 1.0, 1.0))
            
            E = get_E_crit(N)
            
            tol = 1E-5 #Should result in correct energy to ~1E-12
            
            eta = 1
            while eta > tol:
                s.update()
                H = s.H_expect
                s.take_step_RK4(0.1)
                eta = s.eta.real.sum()
                
            print("N=",N," D=",D," E=",E," E_mps=",H," difference=",abs(E-H))
            self.assertTrue(sp.allclose(E, H))

            rtstep = 0.1
            s.ham = get_ham(N, 1.0, 1.1)
            s.update()
            H1 = s.H_expect
            for j in range(10):
                s.take_step_RK4(rtstep*1.j)
                s.update()
            H2 = s.H_expect
            print("N=",N," D=",D," E1=",H1," E2=",H2," difference=",abs(H1-H2))
            self.assertTrue(abs(H1-H2) < rtstep**4)

        
    def test_ising_crit_im_tdvp_split_step(self):
        for N, D in ((5, 8),(8,8)): #exact state, and also truncation    
            s = tdvp.EvoMPS_TDVP_Generic(N, [D] * (N + 1), [2] * (N + 1), get_ham(N, 1.0, 1.0))
            
            E = get_E_crit(N)
            
            tol = 1E-5 #Should result in correct energy to ~1E-12
            
            itr = 1
            eta = 1
            while eta > tol:
                s.update()
                B = s.calc_B() #ensure eta is set!
                eta = s.eta.real.sum()
                H = s.H_expect
                print(itr,": eta = ",eta, " E_mps = ", H)
                if eta < 1E-2:
                    s.take_step_split(0.1)
                else:
                    s.take_step(0.1, B=B)
                
                itr += 1

            print("N=",N," D=",D," E=",E," E_mps=",H," difference=",abs(E-H))
            self.assertTrue(sp.allclose(E, H))

            s.ham = get_ham(N, 1.0, 1.1)
            s.update()
            s_ref = copy.deepcopy(s)
            H1 = s.H_expect

            for j in range(20): #Loc. err. dt**4. Global ??. Here: Loc. 1E-4. Only a few time steps...
                s.take_step_split(0.1j)
            s.update()

            for j in range(20000): #Euler. loc. err. dt**2, glob. dt. Here: loc. 1E-8, glob. 1E-4
                s_ref.take_step(0.0001 * 1.j)
                s_ref.update()

            H2 = s.H_expect
            ent = s.entropy(N//2)
            ent_ref = s_ref.entropy(N//2)

            print("N=",N," D=",D," E1=",H1," E2=",H2," difference=",abs(H1-H2))
            self.assertTrue(abs(H1-H2) < 1e-12) #split-step preserves energy to solver precision

            print("N=",N," D=",D," ent=",ent," ent_ref=",ent_ref," difference=",abs(ent-ent_ref))
            self.assertTrue(abs(ent-ent_ref) < 1E-3)
        
    def test_ising_crit_im_tdvp_DMRG(self):
        for N, D in ((5, 8),(8,8)): #exact state, and also truncation  
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
            
            print("N=",N," D=",D," E=",E," E_mps=",H," difference=",abs(E-H))
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
            
        self.assertTrue(s.D[N//2] < D)
        
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
        
        print("N=",N," D=",D," E=",E," E_mps=",H," difference=",abs(E-H))
        self.assertTrue(sp.allclose(E, H))

        
if __name__ == '__main__':
    unittest.main()