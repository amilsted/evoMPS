# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 18:11:02 2013

@author: ash
"""

import scipy as sp
import scipy.linalg as la
import evoMPS.matmul as mm
import evoMPS.tdvp_common as tc
import evoMPS.mps_uniform_pinv as upin
import evoMPS.mps_uniform as uni
import unittest

class TestOps(unittest.TestCase):

    def setUp(self):
        self.d = 3
        self.D = 8
                
        self.A = [sp.rand(self.d, self.D, self.D) + 1.j * sp.rand(self.d, self.D, self.D)]
        self.B = [sp.rand(self.d, self.D, self.D) + 1.j * sp.rand(self.d, self.D, self.D)]
        
        s = uni.EvoMPS_MPS_Uniform(self.D, self.d)
        s.A = self.A
        s.calc_lr()
        self.A = s.A
        self.Al = s.l[-1]
        self.Ar = s.r[-1]
        
        s = uni.EvoMPS_MPS_Uniform(self.D, self.d)
        s.A = self.B
        s.calc_lr()
        self.B = s.A
        self.Bl = s.l[-1]
        self.Br = s.r[-1]
        
        self.x = sp.rand(self.D, self.D) + 1.j * sp.rand(self.D, self.D)
        
        self.y = sp.rand(self.D, self.D) + 1.j * sp.rand(self.D, self.D)
        
        self.solver_tol = 1E-14
    
#    def test_check_spec(self):
#        ev, EVL, EVR = la.eig(self.E_AA, left=True, right=True)
#        
#        self.assertAlmostEqual(ev.max(), 1.0)
#        
#        self.assertTrue(sp.allclose(self.Ar.ravel() / la.norm(self.Ar.ravel()), EVR[:, ev.argmax()]))
#        
#        self.assertTrue(sp.allclose(self.Al.ravel() / la.norm(self.Al.ravel()), EVL[:, ev.argmax()]))
    
    def test_pinv_linop(self):
        p = 2.3
        
        pinvE = upin.pinv_1mE_brute(self.A, self.B, self.x, self.y, p=p, pseudo=True)
        pinvELOPR = upin.pinv_1mE_brute_LOP(self.A, self.B, self.x, self.y, p=p, pseudo=True, left=False)
        pinvELOPL = upin.pinv_1mE_brute_LOP(self.A, self.B, self.x, self.y, p=p, pseudo=True, left=True)
        
        self.assertTrue(sp.allclose(pinvE, pinvELOPR, rtol=1E-15, atol=1E-15))
        self.assertTrue(sp.allclose(pinvE, pinvELOPL, rtol=1E-15, atol=1E-15))
        
        pinvE = upin.pinv_1mE_brute(self.A, self.B, self.x, self.y, p=p, pseudo=False)
        pinvELOPR = upin.pinv_1mE_brute_LOP(self.A, self.B, self.x, self.y, p=p, pseudo=False, left=False)
        pinvELOPL = upin.pinv_1mE_brute_LOP(self.A, self.B, self.x, self.y, p=p, pseudo=False, left=True)
        
        self.assertTrue(sp.allclose(pinvE, pinvELOPR, rtol=1E-15, atol=1E-15))
        self.assertTrue(sp.allclose(pinvE, pinvELOPL, rtol=1E-15, atol=1E-15))
    
    def test_pinv_AA_right_p0(self):
        res = upin.pinv_1mE(self.x, self.A, self.A, self.Al, self.Ar, p=0, left=False, pseudo=True, tol=self.solver_tol)
        
        pinvE = upin.pinv_1mE_brute(self.A, self.A, self.Al, self.Ar, p=0, pseudo=True)
        res_brute = pinvE.dot(self.x.ravel()).reshape(self.D, self.D)
        
        self.assertLess(la.norm(res_brute - res), self.solver_tol * la.norm(res_brute))
        
    def test_pinv_AA_left_p0(self):
        res = upin.pinv_1mE(self.x, self.A, self.A, self.Al, self.Ar, p=0, left=True, pseudo=True, tol=self.solver_tol)
        
        pinvE = upin.pinv_1mE_brute(self.A, self.A, self.Al, self.Ar, p=0, pseudo=True)
        
        res_brute = pinvE.conj().T.dot(self.x.ravel()).reshape(self.D, self.D)
        
        self.assertLess(la.norm(res_brute - res), self.solver_tol * la.norm(res_brute))
        
    def test_pinv_AA_right_p2(self):
        res = upin.pinv_1mE(self.x, self.A, self.A, self.Al, self.Ar, p=2, left=False, pseudo=True, tol=self.solver_tol)
        
        pinvE = upin.pinv_1mE_brute(self.A, self.A, self.Al, self.Ar, p=2, pseudo=True)
        
        res_brute = pinvE.dot(self.x.ravel()).reshape(self.D, self.D)
        
        self.assertLess(la.norm(res_brute - res), self.solver_tol * la.norm(res_brute))
        
    def test_pinv_AA_left_p2(self):
        res = upin.pinv_1mE(self.x, self.A, self.A, self.Al, self.Ar, p=2, left=True, pseudo=True, tol=self.solver_tol)
        
        pinvE = upin.pinv_1mE_brute(self.A, self.A, self.Al, self.Ar, p=2, pseudo=True)
        
        res_brute = pinvE.conj().T.dot(self.x.ravel()).reshape(self.D, self.D)
        
        self.assertLess(la.norm(res_brute - res), self.solver_tol * la.norm(res_brute))
        
    def test_inv_AB_right_p0(self):
        res = upin.pinv_1mE(self.x, self.A, self.B, None, None, p=0, left=False, pseudo=False, tol=self.solver_tol)
        
        pinvE = upin.pinv_1mE_brute(self.A, self.B, None, None, p=0, pseudo=False)
        res_brute = pinvE.dot(self.x.ravel()).reshape(self.D, self.D)
        
        self.assertLess(la.norm(res_brute - res), self.solver_tol * la.norm(res_brute))

    def test_inv_AB_left_p0(self):
        res = upin.pinv_1mE(self.x, self.A, self.B, None, None, p=0, left=True, pseudo=False, tol=self.solver_tol)
        
        pinvE = upin.pinv_1mE_brute(self.A, self.B, None, None, p=0, pseudo=False)
        res_brute = pinvE.conj().T.dot(self.x.ravel()).reshape(self.D, self.D)
        
        self.assertLess(la.norm(res_brute - res), self.solver_tol * la.norm(res_brute))
        
    def test_inv_AB_right_p2(self):
        res = upin.pinv_1mE(self.x, self.A, self.B, None, None, p=2, left=False, pseudo=False, tol=self.solver_tol)
        
        pinvE = upin.pinv_1mE_brute(self.A, self.B, None, None, p=2, pseudo=False)
        res_brute = pinvE.dot(self.x.ravel()).reshape(self.D, self.D)
        
        self.assertLess(la.norm(res_brute - res), self.solver_tol * la.norm(res_brute))

    def test_inv_AB_left_p2(self):
        res = upin.pinv_1mE(self.x, self.A, self.B, None, None, p=2, left=True, pseudo=False, tol=self.solver_tol)
        
        pinvE = upin.pinv_1mE_brute(self.A, self.B, None, None, p=2, pseudo=False)
        res_brute = pinvE.conj().T.dot(self.x.ravel()).reshape(self.D, self.D)
        
        self.assertLess(la.norm(res_brute - res), self.solver_tol * la.norm(res_brute))
        
    def test_pinv_AA_right_p0_rank_def(self):
        A0 = sp.zeros((self.d, self.D + 2, self.D + 2), dtype=self.A[0].dtype)
        A0[:, :self.D, :self.D] = self.A[0]
        Al = sp.zeros((self.D + 2, self.D + 2), dtype=self.A[0].dtype)
        Al[:self.D, :self.D] = self.Al
        Ar = sp.zeros((self.D + 2, self.D + 2), dtype=self.A[0].dtype)
        Ar[:self.D, :self.D] = self.Ar
        
        x = sp.zeros((self.D + 2, self.D + 2), dtype=self.A[0].dtype)
        x[:self.D, :self.D] = self.x
        
        res = upin.pinv_1mE(x, [A0], [A0], Al, Ar, p=0, left=False, pseudo=True, tol=self.solver_tol)
        
        pinvE = upin.pinv_1mE_brute([A0], [A0], Al, Ar, p=0, pseudo=True)
        res_brute = pinvE.dot(x.ravel()).reshape(self.D + 2, self.D + 2)
        
        self.assertLess(la.norm(res_brute - res), self.solver_tol * la.norm(res_brute))

if __name__ == '__main__':
    unittest.main()