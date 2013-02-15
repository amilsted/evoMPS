# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 11:52:18 2012

@author: ash
"""

import cython
cimport numpy as np
#cimport matmul as mm

ctypedef np.complex128_t npcmp

@cython.locals(s = cython.int, out = np.ndarray)
cpdef np.ndarray eps_l_noop(x, np.ndarray A1, np.ndarray A2)

@cython.locals(s = cython.int)
cpdef np.ndarray eps_l_noop_inplace(x, np.ndarray A1, np.ndarray A2, np.ndarray out)

@cython.locals(s = cython.int, t = cython.int, o_st = npcmp, out = np.ndarray)
cpdef np.ndarray eps_l_op_1s(x, np.ndarray A1, np.ndarray A2, np.ndarray op)

@cython.locals(s = cython.int, out = np.ndarray)
cpdef np.ndarray eps_r_noop(x, np.ndarray A1, np.ndarray A2)

@cython.locals(s = cython.int)
cpdef np.ndarray eps_r_noop_inplace(x, np.ndarray A1, np.ndarray A2, np.ndarray out)

@cython.locals(s = cython.int, t = cython.int, o_st = npcmp, out = np.ndarray)
cpdef np.ndarray eps_r_op_1s(x, np.ndarray A1, np.ndarray A2, np.ndarray op)

@cython.locals(s = cython.int, t = cython.int, u = cython.int, v = cython.int, opval = npcmp, res = np.ndarray, subres = np.ndarray)
cpdef np.ndarray eps_r_op_2s_A(x, np.ndarray A1, np.ndarray A2, np.ndarray A3, np.ndarray A4, np.ndarray op)

@cython.locals(s = cython.int, t = cython.int, u = cython.int, v = cython.int, opval = npcmp, res = np.ndarray, subres = np.ndarray)
cpdef np.ndarray eps_r_op_2s_AA12(x, np.ndarray AA12, np.ndarray A3, np.ndarray A4, np.ndarray op)

@cython.locals(s = cython.int, t = cython.int, u = cython.int, v = cython.int, opval = npcmp, res = np.ndarray, subres = np.ndarray)
cpdef np.ndarray eps_r_op_2s_AA_func_op(x, np.ndarray AA12, np.ndarray AA34, op)

@cython.locals(s = cython.int, t = cython.int, u = cython.int, v = cython.int, res = np.ndarray)
cpdef np.ndarray eps_r_op_2s_C12(x, np.ndarray C12, np.ndarray A3, np.ndarray A4)

@cython.locals(s = cython.int, t = cython.int, u = cython.int, v = cython.int, res = np.ndarray)
cpdef np.ndarray eps_r_op_2s_C34(x, np.ndarray A1, np.ndarray A2, np.ndarray C34)

@cython.locals(s = cython.int, t = cython.int, u = cython.int, v = cython.int, res = np.ndarray)
cpdef np.ndarray eps_r_op_2s_C12_AA34(x, np.ndarray C12, np.ndarray AA34)

@cython.locals(s = cython.int, t = cython.int, u = cython.int, v = cython.int, res = np.ndarray)
cpdef np.ndarray eps_r_op_2s_AA12_C34(x, np.ndarray AA12, np.ndarray C34)

@cython.locals(Dp1 = cython.int, Dm1 = cython.int, q = cython.int, qp1 = cython.int, s = cython.int, t = cython.int, u = cython.int, v = cython.int, res = np.ndarray)
cpdef np.ndarray eps_l_op_2s_AA12_C34(x, np.ndarray AA12, np.ndarray C34)

@cython.locals(u = cython.int, v = cython.int, AA = np.ndarray)
cpdef np.ndarray calc_AA(np.ndarray A, np.ndarray Ap1)

@cython.locals(q = cython.int, qp1 = cython.int, s = cython.int, t = cython.int, u = cython.int, v = cython.int, h_nn_stuv = npcmp, AAuv = np.ndarray, C = np.ndarray)
cpdef np.ndarray calc_C_func_op(op, np.ndarray A, np.ndarray Ap1)

@cython.locals(q = cython.int, qp1 = cython.int, s = cython.int, t = cython.int, u = cython.int, v = cython.int, h_nn_stuv = npcmp, AAuv = np.ndarray, C = np.ndarray)
cpdef np.ndarray calc_C_func_op_AA(op, np.ndarray AA)

@cython.locals(Dm1 = cython.int, q = cython.int, qp1 = cython.int, s = cython.int, t = cython.int, Ash = np.ndarray, K = np.ndarray, Hr = np.ndarray, op_expect = npcmp)
cpdef np.ndarray calc_K(np.ndarray Kp1, np.ndarray C, lm1, rp1, np.ndarray A, np.ndarray Ap1, bint sanity_checks=*)

@cython.locals(D = cython.int, Dm1 = cython.int, q = cython.int, qp1 = cython.int, qm1 = cython.int, s = cython.int, t = cython.int, x = np.ndarray, x_part = np.ndarray, x_subpart = np.ndarray, x_subsubpart = np.ndarray)
cpdef np.ndarray calc_x(np.ndarray Kp1, np.ndarray C, np.ndarray Cm1, rp1, lm2, np.ndarray Am1, np.ndarray A, np.ndarray Ap1, lm1_s, lm1_si, r_s, r_si, np.ndarray Vsh)

@cython.locals(s = cython.int, x = np.ndarray)
cpdef restore_RCF_r(np.ndarray A, lm1, np.ndarray Gm1, bint sanity_checks=*)

@cython.locals(s = cython.int)
cpdef restore_RCF_l(np.ndarray A, r, np.ndarray G_n_i, bint sanity_checks=*)