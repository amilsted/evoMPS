# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 18:11:02 2013

@author: ash
"""
from __future__ import absolute_import, division, print_function

import scipy as sp
import evoMPS.matmul as mm
import evoMPS.tdvp_common as tc
import unittest

def make_E_noop(A, B):
    res = sp.zeros((A.shape[1]**2, A.shape[2]**2), dtype=A.dtype)
    for s in range(A.shape[0]):
        res += sp.kron(A[s], B[s].conj())
    return res
    
def make_E_1s(A, B, op):
    res = sp.zeros((A.shape[1]**2, A.shape[2]**2), dtype=A.dtype)
    for s in range(A.shape[0]):
        for t in range(A.shape[0]):
            res += sp.kron(A[s], B[t].conj()) * op[t, s]
    return res

def make_E_2s(A1, A2, B1, B2, op):
    res = sp.zeros((A1.shape[1]**2, A2.shape[2]**2), dtype=A1.dtype)
    for s in range(A1.shape[0]):
        for t in range(A2.shape[0]):
            for u in range(A1.shape[0]):
                for v in range(A2.shape[0]):
                    res += sp.kron(A1[s].dot(A2[t]), B1[u].dot(B2[v]).conj()) * op[u, v, s, t]
    return res

class TestOps(unittest.TestCase):

    def setUp(self): #TODO: Test rectangular x as well
        self.d = [2, 3]
        self.D = [2, 4, 3]
        
        self.l0 = sp.rand(self.D[0], self.D[0]) + 1.j * sp.rand(self.D[0], self.D[0])
        self.r1 = sp.rand(self.D[1], self.D[1]) + 1.j * sp.rand(self.D[1], self.D[1])
        self.r2 = sp.rand(self.D[2], self.D[2]) + 1.j * sp.rand(self.D[2], self.D[2])
        
        self.ld0 = mm.simple_diag_matrix(sp.rand(self.D[0]) + 1.j * sp.rand(self.D[0]))
        self.rd1 = mm.simple_diag_matrix(sp.rand(self.D[1]) + 1.j * sp.rand(self.D[1]))
        self.rd2 = mm.simple_diag_matrix(sp.rand(self.D[2]) + 1.j * sp.rand(self.D[2]))
        
        self.eye0 = mm.eyemat(self.D[0], dtype=sp.complex128)
        self.eye1 = mm.eyemat(self.D[1], dtype=sp.complex128)
        self.eye2 = mm.eyemat(self.D[2], dtype=sp.complex128)
        
        self.A0 = sp.rand(self.d[0], self.D[0], self.D[0]) + 1.j * sp.rand(self.d[0], self.D[0], self.D[0])
        self.A1 = sp.rand(self.d[0], self.D[0], self.D[1]) + 1.j * sp.rand(self.d[0], self.D[0], self.D[1])
        self.A2 = sp.rand(self.d[1], self.D[1], self.D[2]) + 1.j * sp.rand(self.d[1], self.D[1], self.D[2])
        self.A3 = sp.rand(self.d[1], self.D[2], self.D[2]) + 1.j * sp.rand(self.d[1], self.D[2], self.D[2])
        
        self.B1 = sp.rand(self.d[0], self.D[0], self.D[1]) + 1.j * sp.rand(self.d[0], self.D[0], self.D[1])
        self.B2 = sp.rand(self.d[1], self.D[1], self.D[2]) + 1.j * sp.rand(self.d[1], self.D[1], self.D[2])
        
        self.E1_AB = make_E_noop(self.A1, self.B1)
        self.E2_AB = make_E_noop(self.A2, self.B2)
        
        self.op1s_1 = sp.rand(self.d[0], self.d[0]) + 1.j * sp.rand(self.d[0], self.d[0])
        self.E1_op_AB = make_E_1s(self.A1, self.B1, self.op1s_1)
        
        self.op2s = sp.rand(self.d[0], self.d[1], self.d[0], self.d[1]) + 1.j * sp.rand(self.d[0], self.d[1], self.d[0], self.d[1])
        self.E12_op = make_E_2s(self.A1, self.A2, self.B1, self.B2, self.op2s)
        
        self.AA12 = tc.calc_AA(self.A1, self.A2)
        self.BB12 = tc.calc_AA(self.B1, self.B2)
        
        self.C_A12 = tc.calc_C_mat_op_AA(self.op2s, self.AA12)
        self.C_conj_B12 = tc.calc_C_conj_mat_op_AA(self.op2s, self.BB12)
        
        self.C01 = sp.rand(self.d[0], self.d[0], self.D[0], self.D[1]) + 1.j * sp.rand(self.d[0], self.d[0], self.D[0], self.D[1])

    def test_eps_l_noop(self):
        l1 = tc.eps_l_noop(self.l0, self.A1, self.B1)
        
        l1_ = self.E1_AB.conj().T.dot(self.l0.ravel()).reshape(self.D[1], self.D[1])
        
        self.assertTrue(sp.allclose(l1, l1_))
        
    def test_eps_l_noop_diag(self):
        l1 = tc.eps_l_noop(self.ld0, self.A1, self.B1)
        
        l1_ = self.E1_AB.conj().T.dot(self.ld0.A.ravel()).reshape(self.D[1], self.D[1])
        
        self.assertTrue(sp.allclose(l1, l1_))
        
    def test_eps_l_noop_eye(self):
        l1 = tc.eps_l_noop(self.eye0, self.A1, self.B1)
        
        l1_ = self.E1_AB.conj().T.dot(self.eye0.A.ravel()).reshape(self.D[1], self.D[1])
        
        self.assertTrue(sp.allclose(l1, l1_))

    def test_eps_l_noop_inplace(self):
        l1 = sp.zeros_like(self.r1)
        l1_ = tc.eps_l_noop_inplace(self.l0, self.A1, self.B1, l1)
        
        self.assertTrue(l1 is l1_)
        
        l1__ = tc.eps_l_noop(self.l0, self.A1, self.B1)
        
        self.assertTrue(sp.allclose(l1, l1__))

    def test_eps_r_noop(self):
        r0 = tc.eps_r_noop(self.r1, self.A1, self.B1)
        
        r0_ = self.E1_AB.dot(self.r1.ravel()).reshape(self.D[0], self.D[0])
        
        self.assertTrue(sp.allclose(r0, r0_))
        
        r1 = tc.eps_r_noop(self.r2, self.A2, self.B2)
        
        r1_ = self.E2_AB.dot(self.r2.ravel()).reshape(self.D[1], self.D[1])
        
        self.assertTrue(sp.allclose(r1, r1_))
        
    def test_eps_r_noop_diag(self):
        r0 = tc.eps_r_noop(self.rd1, self.A1, self.B1)
        
        r0_ = self.E1_AB.dot(self.rd1.A.ravel()).reshape(self.D[0], self.D[0])
        
        self.assertTrue(sp.allclose(r0, r0_))
        
    def test_eps_r_noop_eye(self):
        r0 = tc.eps_r_noop(self.eye1, self.A1, self.B1)
        
        r0_ = self.E1_AB.dot(self.eye1.A.ravel()).reshape(self.D[0], self.D[0])
        
        self.assertTrue(sp.allclose(r0, r0_))
        
    def test_eps_r_noop_multi(self): 
        r0 = tc.eps_r_noop(tc.eps_r_noop(self.r2, self.A2, self.B2), self.A1, self.B1)
        
        r0_ = tc.eps_r_noop_multi(self.r2, [self.A1, self.A2], [self.B1, self.B2])
        
        self.assertTrue(sp.allclose(r0, r0_))
        
        r0__ = tc.eps_r_noop_multi(self.r2, [self.AA12], [self.BB12])
        
        self.assertTrue(sp.allclose(r0, r0__))
        
        r0C = tc.eps_r_op_2s_C12(self.r2, self.C_A12, self.B1, self.B2)
        r0C_ = tc.eps_r_noop_multi(self.r2, [self.C_A12], [self.B1, self.B2])
        
        self.assertTrue(sp.allclose(r0C, r0C_))
        
        r0C2 = tc.eps_r_op_2s_C12_AA34(self.r2, self.C_A12, self.BB12)
        r0C2_ = tc.eps_r_noop_multi(self.r2, [self.C_A12], [self.BB12])
        
        self.assertTrue(sp.allclose(r0C2, r0C2_))
        
        r0CA2 = tc.eps_r_op_2s_C12(tc.eps_r_noop(self.r2, self.A2, self.B2), 
                                   self.C01, self.A0, self.B1)
        r0CA2_ = tc.eps_r_noop_multi(self.r2, [self.C01, self.A2], [self.A0, self.BB12])
        
        self.assertTrue(sp.allclose(r0CA2, r0CA2_))

        
    def test_eps_r_noop_inplace(self):
        r0 = sp.zeros_like(self.l0)
        r0_ =tc.eps_r_noop_inplace(self.r1, self.A1, self.B1, r0)
        
        self.assertTrue(r0 is r0_)
        
        r0__ = tc.eps_r_noop(self.r1, self.A1, self.B1)
        
        self.assertTrue(sp.allclose(r0, r0__))
        
    def test_eps_l_op_1s(self):
        l1 = tc.eps_l_op_1s(self.l0, self.A1, self.B1, self.op1s_1)
        
        l1_ = self.E1_op_AB.conj().T.dot(self.l0.ravel()).reshape(self.D[1], self.D[1])
        
        self.assertTrue(sp.allclose(l1, l1_))

    def test_eps_r_op_1s(self):
        r0 = tc.eps_r_op_1s(self.r1, self.A1, self.B1, self.op1s_1)
        
        r0_ = self.E1_op_AB.dot(self.r1.ravel()).reshape(self.D[0], self.D[0])
        
        self.assertTrue(sp.allclose(r0, r0_))
        
    def test_eps_r_op_2s_A(self):
        r0 = tc.eps_r_op_2s_A(self.r2, self.A1, self.A2, self.B1, self.B2, self.op2s)
        
        r0_ = self.E12_op.dot(self.r2.ravel()).reshape(self.D[0], self.D[0])
        
        self.assertTrue(sp.allclose(r0, r0_))
        
    def test_eps_r_op_2s_AA12(self):
        r0 = tc.eps_r_op_2s_AA12(self.r2, self.AA12, self.B1, self.B2, self.op2s)
        
        r0_ = tc.eps_r_op_2s_A(self.r2, self.A1, self.A2, self.B1, self.B2, self.op2s)
        
        self.assertTrue(sp.allclose(r0, r0_))
        
    def test_eps_r_op_2s_AA_func_op(self):
        r0 = tc.eps_r_op_2s_AA_func_op(self.r2, self.AA12, self.BB12, lambda s,t,u,v: self.op2s[s,t,u,v])
        
        r0_ = tc.eps_r_op_2s_A(self.r2, self.A1, self.A2, self.B1, self.B2, self.op2s)
        
        self.assertTrue(sp.allclose(r0, r0_))
        
    def test_eps_r_op_2s_C12(self):
        r0 = tc.eps_r_op_2s_C12(self.r2, self.C_A12, self.B1, self.B2)
        
        r0_ = tc.eps_r_op_2s_A(self.r2, self.A1, self.A2, self.B1, self.B2, self.op2s)
        
        self.assertTrue(sp.allclose(r0, r0_))
        
    def test_eps_r_op_2s_C34(self):
        r0 = tc.eps_r_op_2s_C34(self.r2, self.A1, self.A2, self.C_conj_B12)
        
        r0_ = tc.eps_r_op_2s_A(self.r2, self.A1, self.A2, self.B1, self.B2, self.op2s)
        
        self.assertTrue(sp.allclose(r0, r0_))
        
    def test_eps_r_op_2s_C12_AA34(self):
        r0 = tc.eps_r_op_2s_C12_AA34(self.r2, self.C_A12, self.BB12)
        
        r0_ = tc.eps_r_op_2s_A(self.r2, self.A1, self.A2, self.B1, self.B2, self.op2s)
        
        self.assertTrue(sp.allclose(r0, r0_))
        
    def test_eps_r_op_2s_AA12_C34(self):
        r0 = tc.eps_r_op_2s_AA12_C34(self.r2, self.AA12, self.C_conj_B12)
        
        r0_ = tc.eps_r_op_2s_A(self.r2, self.A1, self.A2, self.B1, self.B2, self.op2s)
        
        self.assertTrue(sp.allclose(r0, r0_))
        
    def test_eps_l_op_2s_AA12_C34(self):
        l2 = tc.eps_l_op_2s_AA12_C34(self.l0, self.AA12, self.C_conj_B12)
        
        l2_ = self.E12_op.conj().T.dot(self.l0.ravel()).reshape(self.D[2], self.D[2])
        
        self.assertTrue(sp.allclose(l2, l2_))
        
    def test_calc_C_func_op(self):
        C = tc.calc_C_func_op(lambda s,t,u,v: self.op2s[s,t,u,v], self.A1, self.A2)
        
        self.assertTrue(sp.allclose(C, self.C_A12))
        
    def test_calc_C_func_op_AA(self):
        C = tc.calc_C_func_op_AA(lambda s,t,u,v: self.op2s[s,t,u,v], self.AA12)
        
        self.assertTrue(sp.allclose(C, self.C_A12))

if __name__ == '__main__':
    unittest.main()