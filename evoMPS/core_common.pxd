# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 11:52:18 2012

@author: ash
"""
#cython: boundscheck=False

import cython
cimport numpy as np
#cimport matmul as mm

ctypedef fused flts:
    np.complex128_t
    np.complex64_t
    np.float64_t
    np.float32_t
ctypedef np.complex128_t npcmp
ctypedef cython.long[:] arint1D
ctypedef flts[:, :] ndar2D
ctypedef flts[:, :, :] ndar3D
ctypedef flts[:, :, :, :] ndar4D
ctypedef flts[:, :, :, :, :] ndar5D

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

@cython.locals(s = cython.int, t = cython.int, S = cython.long, ind = cython.int,
               nA1 = cython.int, nA2 = cython.int,
               A1dims_prod = arint1D, A2dims_prod = arint1D,
               A1dims = arint1D, A2dims = arint1D,
               A1s_prod = np.ndarray, A2s_prod = np.ndarray, out = np.ndarray)
cpdef np.ndarray eps_r_noop_multi(x, A1, A2)

@cython.locals(s = cython.int, t = cython.int, o_st = npcmp, out = np.ndarray)
cpdef np.ndarray eps_r_op_1s(x, np.ndarray A1, np.ndarray A2, np.ndarray op)

@cython.locals(s = cython.int, t = cython.int, u = cython.int, v = cython.int, opval = npcmp, res = np.ndarray, subres = np.ndarray)
cpdef np.ndarray eps_r_op_2s_A(x, np.ndarray A1, np.ndarray A2, np.ndarray A3, np.ndarray A4, np.ndarray op)

@cython.locals(s = cython.int, t = cython.int, u = cython.int, v = cython.int, opval = npcmp, res = np.ndarray, subres = np.ndarray)
cpdef np.ndarray eps_r_op_2s_AA12(x, np.ndarray AA12, np.ndarray A3, np.ndarray A4, np.ndarray op)

@cython.locals(s = cython.int, t = cython.int, u = cython.int, v = cython.int, opval = npcmp, res = np.ndarray, subres = np.ndarray)
cpdef np.ndarray eps_r_op_2s_AA_func_op(x, np.ndarray AA12, np.ndarray AA34, op)

@cython.locals(u = cython.int, v = cython.int, res = np.ndarray)
cpdef np.ndarray eps_r_op_2s_C12(x, np.ndarray C12, np.ndarray A3, np.ndarray A4)

@cython.locals(u = cython.int, v = cython.int, res = np.ndarray)
cpdef np.ndarray eps_r_op_2s_C34(x, np.ndarray A1, np.ndarray A2, np.ndarray C34)
    
@cython.locals(Dp1 = cython.int, Dm1 = cython.int, q = cython.int, qp1 = cython.int, u = cython.int, v = cython.int, AA = np.ndarray)
cpdef np.ndarray calc_AA(np.ndarray A, np.ndarray Ap1)

@cython.locals(Dp1 = cython.int, Dp2 = cython.int, Dm1 = cython.int, q = cython.int, qp1 = cython.int, qp2 = cython.int, u = cython.int, v = cython.int, w = cython.int, AAA = np.ndarray)
cpdef np.ndarray calc_AAA(np.ndarray A, np.ndarray Ap1, np.ndarray Ap2)

@cython.locals(Dp2 = cython.int, Dm1 = cython.int, q = cython.int, qp1 = cython.int, qp2 = cython.int, u = cython.int, v = cython.int, w = cython.int, AAA = np.ndarray)
cpdef np.ndarray calc_AAA_AA(np.ndarray AAp1, np.ndarray Ap2)

@cython.locals(q = cython.int, qp1 = cython.int, s = cython.int, t = cython.int, u = cython.int, v = cython.int, h_nn_stuv = npcmp, AAuv = np.ndarray, C = np.ndarray)
cpdef np.ndarray calc_C_func_op(op, np.ndarray A, np.ndarray Ap1)

@cython.locals(q = cython.int, qp1 = cython.int, s = cython.int, t = cython.int, u = cython.int, v = cython.int, h_nn_stuv = npcmp, AAuv = np.ndarray, C = np.ndarray)
cpdef np.ndarray calc_C_func_op_AA(op, np.ndarray AA)
