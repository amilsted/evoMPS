# -*- coding: utf-8 -*-
"""
Created on Sun Nov 25 11:49:32 2012

@author: ash

TODO:
    - Sane implementation of sanity checks (return the info to the caller)
"""

import scipy as sp
import scipy.linalg as la
import numpy as np
import matmul as mm
import nullspace as ns   
import logging

log = logging.getLogger(__name__)

def eps_l_noop(x, A1, A2):
    """Implements the left epsilon map.
    
    For example, in the generic case: l[n] = eps_l_noop(l[n - 1], A[n], A[n])
    
    The input and output matrices are "column vectors" or "kets" and we
    implement multiplication from the left as (E^A1_A2).conj().T.dot(x.ravel()).
    In other words <x|E = <eps_l_noop(x, A, A)|.
    
    Note: E^A1_A2 = sum_over_s_of( kron(A1[s], A2[s].conj()) )

    Parameters
    ----------
    x : ndarray
        The argument matrix.
    A1: ndarray
        The MPS ket tensor for the current site.
    A2: ndarray
        The MPS bra tensor for the current site.    

    Returns
    -------
    res : ndarray
        The resulting matrix.
    """
    out = np.zeros((A1.shape[2], A2.shape[2]), dtype=A1.dtype)
    for s in xrange(A1.shape[0]):
        out += A1[s].conj().T.dot(x.dot(A2[s]))
    return out
    
def eps_l_noop_inplace(x, A1, A2, out):
    """Implements the left epsilon map for a pre-exisiting output matrix.
    
    The output must have shape (A1.shape[2], A2.shape[2]).
    
    See eps_l_noop().
    
    Parameters
    ----------
    x : ndarray
        The argument matrix.
    A1: ndarray
        The MPS ket tensor for the current site.
    A2: ndarray
        The MPS bra tensor for the current site.
    out: ndarray
        The output matrix (must have correct dimensions).
        
    Returns
    -------
    res : ndarray
        The resulting matrix.
    """
    out.fill(0)
    for s in xrange(A1.shape[0]):
        out += A1[s].conj().T.dot(x.dot(A2[s]))
    return out
        
def eps_l_op_1s(x, A1, A2, op):
    """Implements the left epsilon map with a non-trivial single-site operator.
    
    For example the expectation value of a single-site operator <op> is equal 
    to adot(eps_l_op_1s(l[n - 1], A[n], A[n], op), r[n]).
    
    This is (E_op(A1,A2)).conj().T.dot(x.ravel()) where x and the output are kets.
    
    Alternatively, this is the same as x.conj().T.ravel().dot(E_(op.conj().T)(A2, A1))
    where x is a bra.
    
    See eps_l_noop().

    Parameters
    ----------
    x : ndarray
        The argument matrix.
    A1: ndarray
        The MPS ket tensor for the current site.
    A2: ndarray
        The MPS bra tensor for the current site.
    op: ndarray
        Single-site operator matrix elements op[s, t] = <s|op|t>

    Returns
    ------
    res : ndarray
        The resulting matrix.
    """
    op = op.conj()
    out = np.zeros((A1.shape[2], A2.shape[2]), dtype=A1.dtype)
    for s in xrange(A1.shape[0]):
        for t in xrange(A1.shape[0]):
            o_st = op[t, s]
            if o_st != 0:
                out += o_st * A1[s].conj().T.dot(x.dot(A2[t]))
    return out
    
def eps_r_noop(x, A1, A2):
    """Implements the right epsilon map
    
    For example 

    Parameters
    ----------
    x : ndarray
        The argument matrix. For example, using l[n - 1] gives a result l[n]
    A1: ndarray
        The MPS ket tensor for the current site.
    A2: ndarray
        The MPS bra tensor for the current site. 

    Returns
    -------
    res : ndarray
        The resulting matrix.
    """
    out = np.zeros((A1.shape[1], A2.shape[1]), dtype=A1.dtype)
    for s in xrange(A1.shape[0]):
        out += A1[s].dot(x.dot(A2[s].conj().T))
    return out
    
def eps_r_noop_inplace(x, A1, A2, out):
    """Implements the right epsilon map for a pre-exisiting output matrix.
    
    The output must have shape (A1.shape[1], A2.shape[1]).
    
    See eps_r_noop().
    
    Parameters
    ----------
    x : ndarray
        The argument matrix.
    A1: ndarray
        The MPS ket tensor for the current site.
    A2: ndarray
        The MPS bra tensor for the current site. 
    out: ndarray
        The output matrix (must have correct dimensions).
        
    Returns
    -------
    res : ndarray
        The resulting matrix.
    """
    out.fill(0)
    for s in xrange(A1.shape[0]):
        out += A1[s].dot(x.dot(A2[s].conj().T))
    return out
    
def eps_r_noop_multi(x, A1, A2):
    """Implements the right epsilon map
    
    For example 

    Parameters
    ----------
    x : ndarray
        The argument matrix. For example, using l[n - 1] gives a result l[n]
    A1: ndarray
        The MPS ket tensor for the current site.
    A2: ndarray
        The MPS bra tensor for the current site. 

    Returns
    -------
    res : ndarray
        The resulting matrix.
    """
    #M = sum([len(A1t.shape) - 2 for A1t in A1])
    
    assert sp.all([len(At.shape) >= 3 for At in A1 + A2]), "Invalid input shapes"
    
    #Flatten site indices within each tensor
    A1 = [A1t.reshape((sp.prod(A1t.shape[:-2]), A1t.shape[-2], A1t.shape[-1])) for A1t in A1]
    A2 = [A2t.reshape((sp.prod(A2t.shape[:-2]), A2t.shape[-2], A2t.shape[-1])) for A2t in A2]
    
    A1dims = sp.array([1] + [A1t.shape[0] for A1t in reversed(A1)])
    A1dims_prod = sp.cumprod(A1dims)
    S = A1dims_prod[-1]
    #print A1dims, A1dims_prod, S
    
    A2dims = sp.array([1] + [A2t.shape[0] for A2t in reversed(A2)])
    A2dims_prod = sp.cumprod(A2dims)
    #print A2dims, A2dims_prod, S
    
    out = np.zeros((A1[0].shape[1], A2[0].shape[1]), dtype=A1[0].dtype)
    
    for s in xrange(S):
        A1ind = [(s / A1dims_prod[t]) % A1dims[t + 1] for t in xrange(len(A1))]
        A1s = [A1[t][A1ind[-(t + 1)]] for t in xrange(len(A1))]
        A1s_prod = reduce(sp.dot, A1s)
        
        A2ind = [(s / A2dims_prod[t]) % A2dims[t + 1] for t in xrange(len(A2))]
        A2s = [A2[t][A2ind[-(t + 1)]] for t in xrange(len(A2))]
        A2s_prod = reduce(sp.dot, A2s)
        
        #print A1s_prod.shape, x.shape, A2s_prod.conj().T.shape
        
        out += A1s_prod.dot(x.dot(A2s_prod.conj().T))
    return out
    
def eps_r_noop_multi_test():
    d = [2, 3]
    D = [2, 4, 3]
    
    l0 = sp.rand(D[0], D[0]) + 1.j * sp.rand(D[0], D[0])
    r1 = sp.rand(D[1], D[1]) + 1.j * sp.rand(D[1], D[1])
    r2 = sp.rand(D[2], D[2]) + 1.j * sp.rand(D[2], D[2])
    
    A0 = sp.rand(d[0], D[0], D[0]) + 1.j * sp.rand(d[0], D[0], D[0])
    A1 = sp.rand(d[0], D[0], D[1]) + 1.j * sp.rand(d[0], D[0], D[1])
    A2 = sp.rand(d[1], D[1], D[2]) + 1.j * sp.rand(d[1], D[1], D[2])
    A3 = sp.rand(d[1], D[2], D[2]) + 1.j * sp.rand(d[1], D[2], D[2])
    
    B1 = sp.rand(d[0], D[0], D[1]) + 1.j * sp.rand(d[0], D[0], D[1])
    B2 = sp.rand(d[1], D[1], D[2]) + 1.j * sp.rand(d[1], D[1], D[2])
    
    C12 = sp.rand(d[0], d[1], D[0], D[2]) + 1.j * sp.rand(d[0], d[1], D[0], D[2])
    C01 = sp.rand(d[0], d[0], D[0], D[1]) + 1.j * sp.rand(d[0], d[0], D[0], D[1])
    
    AA12 = calc_AA(A1, A2)
    BB12 = calc_AA(B1, B2)
    
    r0 = eps_r_noop(eps_r_noop(r2, A2, B2), A1, B1)
    
    r0_ = eps_r_noop_multi(r2, [A1, A2], [B1, B2])
    
    print la.norm(r0 - r0_)
    
    r0__ = eps_r_noop_multi(r2, [AA12], [BB12])
    
    print la.norm(r0 - r0__)
    
    r0C = eps_r_op_2s_C12(r2, C12, B1, B2)
    r0C_ = eps_r_noop_multi(r2, [C12], [B1, B2])
    
    print la.norm(r0C - r0C_)    
    
    r0C2 = eps_r_op_2s_C12_AA34(r2, C12, BB12)
    r0C2_ = eps_r_noop_multi(r2, [C12], [BB12])
    
    print la.norm(r0C2 - r0C2_)
    
    r0CA2 = eps_r_op_2s_C12(eps_r_noop(r2, A2, B2), C01, A0, B1)
    r0CA2_ = eps_r_noop_multi(r2, [C01, A2], [A0, BB12])
    
    print la.norm(r0CA2 - r0CA2_)
    
def eps_r_op_1s(x, A1, A2, op):
    """Implements the right epsilon map with a non-trivial single-site operator.
    
    For example the expectation value of a single-site operator <op> is equal 
    to adot(l[n - 1], eps_r_op_1s(r[n], A[n], A[n], op)).
    
    See eps_r_noop().

    Parameters
    ----------
    x : ndarray
        The argument matrix.
    A1: ndarray
        The MPS ket tensor for the current site.
    A2: ndarray
        The MPS bra tensor for the current site.
    op: ndarray
        Single-site operator matrix elements op[s, t] = <s|op|t>

    Returns
    ------
    res : ndarray
        The resulting matrix.
    """
    out = np.zeros((A1.shape[1], A1.shape[1]), dtype=A1.dtype)
    for s in xrange(A1.shape[0]):
        for t in xrange(A1.shape[0]):
            o_st = op[s, t]
            if o_st != 0:
                out += o_st * A1[t].dot(x.dot(A2[s].conj().T))
    return out
    
def eps_r_op_2s_A(x, A1, A2, A3, A4, op):
    """Implements the right epsilon map with a non-trivial nearest-neighbour operator.
    
    For example the expectation value of an operator <op> is equal 
    to adot(l[n - 2], eps_r_op_2s_A(r[n], A[n], A[n], A[n], A[n], op)).

    Parameters
    ----------
    x : ndarray
        The argument matrix.
    A1: ndarray
        The MPS ket tensor for the first site.
    A2: ndarray
        The MPS ket tensor for the second site.
    A3: ndarray
        The MPS bra tensor for the first site.
    A4: ndarray
        The MPS bra tensor for the second site.
    op: ndarray
        Nearest-neighbour operator matrix elements op[s, t, u, v] = <st|op|uv>

    Returns
    ------
    res : ndarray
        The resulting matrix.
    """
    res = np.zeros((A1.shape[1], A3.shape[1]), dtype=A1.dtype)
    zeros = np.zeros
    for u in xrange(A3.shape[0]):
        for v in xrange(A4.shape[0]):
            subres = zeros((A1.shape[1], A2.shape[2]), dtype=A1.dtype)
            for s in xrange(A1.shape[0]):
                for t in xrange(A2.shape[0]):
                    opval = op[u, v, s, t]
                    if opval != 0:
                        subres += opval * A1[s].dot(A2[t])
            res += subres.dot(x.dot((A3[u].dot(A4[v])).conj().T))
    return res
    
def eps_r_op_2s_AA12(x, AA12, A3, A4, op):
    """Implements the right epsilon map with a non-trivial nearest-neighbour operator.
    
    Uses a pre-multiplied tensor for the ket AA12[s, t] = A1[s].dot(A2[t]). 
    
    See eps_r_op_2s_A().

    Parameters
    ----------
    x : ndarray
        The argument matrix.
    AA12: ndarray
        The combined MPS ket tensor for the first and second sites.
    A3: ndarray
        The MPS bra tensor for the first site.
    A4: ndarray
        The MPS bra tensor for the second site.
    op: ndarray
        Nearest-neighbour operator matrix elements op[s, t, u, v] = <st|op|uv>

    Returns
    ------
    res : ndarray
        The resulting matrix.
    """
    res = np.zeros((AA12.shape[2], A3.shape[1]), dtype=A3.dtype)
    zeros_like = np.zeros_like
    for u in xrange(A3.shape[0]):
        for v in xrange(A4.shape[0]):
            subres = zeros_like(AA12[0, 0])
            for s in xrange(AA12.shape[0]):
                for t in xrange(AA12.shape[1]):
                    opval = op[u, v, s, t]
                    if opval != 0:
                        subres += opval * AA12[s, t]
            res += subres.dot(x.dot((A3[u].dot(A4[v])).conj().T))
    return res
    
def eps_r_op_2s_AA_func_op(x, AA12, AA34, op):
    """Implements the right epsilon map with a non-trivial nearest-neighbour operator.
    
    Uses pre-multiplied tensors for the ket AA12[s, t] = A1[s].dot(A2[t])
    and bra AA34[s, t] = A3[s].dot(A4[t]).
    
    See eps_r_op_2s_A().

    Parameters
    ----------
    x : ndarray
        The argument matrix.
    AA12: ndarray
        The combined MPS ket tensor for the first and second sites.
    AA34: ndarray
        The combined MPS bra tensor for the first and second sites.
    op: callable
        Nearest-neighbour operator matrix element function op(s, t, u, v) = <st|op|uv>

    Returns
    ------
    res : ndarray
        The resulting matrix.
    """
    res = np.zeros((AA12.shape[2], AA34.shape[2]), dtype=AA12.dtype)
    zeros_like = np.zeros_like
    for u in xrange(AA34.shape[0]):
        for v in xrange(AA34.shape[1]):
            subres = zeros_like(AA12[0, 0])
            for s in xrange(AA12.shape[0]):
                for t in xrange(AA12.shape[1]):
                    opval = op(u, v, s, t)
                    if opval != 0:
                        subres += opval * AA12[s, t]
            res += subres.dot(x.dot((AA34[u, v]).conj().T))
    return res
    
def eps_r_op_2s_C12(x, C12, A3, A4):
    """Implements the right epsilon map with a non-trivial nearest-neighbour operator.
    
    Uses pre-multiplied tensors for the ket and operator 
    C12 = calc_C_mat_op_AA(op, AA12) with AA12[s, t] = A1[s].dot(A2[t]).
    
    See eps_r_op_2s_A().

    Parameters
    ----------
    x : ndarray
        The argument matrix.
    C12: ndarray
        The combined MPS ket tensor for the first and second sites.
    A3: ndarray
        The MPS bra tensor for the first site.
    A4: ndarray
        The MPS bra tensor for the second site.

    Returns
    ------
    res : ndarray
        The resulting matrix.
    """
    res = np.zeros((C12.shape[2], A3.shape[1]), dtype=A3.dtype)
    for u in xrange(A3.shape[0]):
        for v in xrange(A4.shape[0]):
            res += C12[u, v].dot(x.dot((A3[u].dot(A4[v])).conj().T))
    return res
    
def eps_r_op_2s_C34(x, A1, A2, C34):
    res = np.zeros((A1.shape[1], C34.shape[2]), dtype=A1.dtype)
    for u in xrange(C34.shape[0]):
        for v in xrange(C34.shape[1]):
            res += A1[u].dot(A2[v]).dot(x.dot(C34[u, v].conj().T))
    return res

def eps_r_op_2s_C12_AA34(x, C12, AA34):
    res = np.zeros((C12.shape[2], AA34.shape[2]), dtype=AA34.dtype)
    for u in xrange(AA34.shape[0]):
        for v in xrange(AA34.shape[1]):
            res += C12[u, v].dot(x.dot(AA34[u, v].conj().T))
    return res
    
def eps_r_op_2s_AA12_C34(x, AA12, C34):
    res = np.zeros((AA12.shape[2], C34.shape[2]), dtype=C34.dtype)
    for u in xrange(C34.shape[0]):
        for v in xrange(C34.shape[1]):
            res += AA12[u, v].dot(x.dot(C34[u, v].conj().T))
    return res
    
def eps_l_op_2s_AA12_C34(x, AA12, C34):
    res = np.zeros((AA12.shape[3], C34.shape[3]), dtype=C34.dtype)
    for u in xrange(C34.shape[0]):
        for v in xrange(C34.shape[1]):
            res += AA12[u, v].conj().T.dot(x.dot(C34[u, v]))
    return res
    
def eps_l_op_2s_A1_A2_C34(x, A1, A2, C34):
    res = np.zeros((A2.shape[2], C34.shape[3]), dtype=C34.dtype)
    for u in xrange(C34.shape[0]):
        for v in xrange(C34.shape[1]):
            res += (A1[u].dot(A2[v])).conj().T.dot(x.dot(C34[u, v]))
    return res

def eps_r_op_3s_C123_AAA456(x, C123, AAA456):
    res = np.zeros((C123.shape[3], AAA456.shape[3]), dtype=AAA456.dtype)
    for u in xrange(AAA456.shape[0]):
        for v in xrange(AAA456.shape[1]):
            for w in xrange(AAA456.shape[2]):
                res += C123[u, v, w].dot(x.dot(AAA456[u, v, w].conj().T))
    return res
    
def eps_l_op_3s_AAA123_C456(x, AAA123, C456):
    res = np.zeros((AAA123.shape[4], C456.shape[4]), dtype=C456.dtype)
    for u in xrange(C456.shape[0]):
        for v in xrange(C456.shape[1]):
            for w in xrange(C456.shape[2]):
                res += AAA123[u, v, w].conj().T.dot(x.dot(C456[u, v, w]))
    return res

def calc_AA(A, Ap1):
    Dp1 = Ap1.shape[2]
    Dm1 = A.shape[1]
    q = A.shape[0]
    qp1 = Ap1.shape[0]
    
    AA = sp.zeros((q, qp1, Dm1, Dp1), dtype=A.dtype)
    for u in xrange(q):
        for v in xrange(qp1):
            AA[u, v] = A[u].dot(Ap1[v])
    
    return AA
    
    #This works too: (just for reference)
    #AA = np.array([dot(A[s], A[t]) for s in xrange(self.q) for t in xrange(self.q)])
    #self.AA = AA.reshape(self.q, self.q, self.D, self.D)

def calc_AAA(A, Ap1, Ap2):
    Dp2 = Ap2.shape[2]
    Dm1 = A.shape[1]
    q = A.shape[0]
    qp1 = Ap1.shape[0]
    qp2 = Ap2.shape[0]
    
    AAA = sp.zeros((q, qp1, qp2, Dm1, Dp2), dtype=A.dtype)
    for u in xrange(q):
        for v in xrange(qp1):
            for w in xrange(qp2):
                AAA[u, v, w] = A[u].dot(Ap1[v]).dot(Ap2[w])
    
    return AAA
    
def calc_AAA_AA(AAp1, Ap2):
    Dp2 = Ap2.shape[2]
    Dm1 = AAp1.shape[2]
    q = AAp1.shape[0]
    qp1 = AAp1.shape[1]
    qp2 = Ap2.shape[0]
    
    AAA = sp.zeros((q, qp1, qp2, Dm1, Dp2), dtype=AAp1.dtype)
    for u in xrange(q):
        for v in xrange(qp1):
            for w in xrange(qp2):
                AAA[u, v, w] = AAp1[u, v].dot(Ap2[w])
    
    return AAA

def calc_C_mat_op_AA(op, AA):
    return sp.tensordot(op, AA, ((2, 3), (0, 1)))
    
def calc_C_3s_mat_op_AAA(op, AAA):
    return sp.tensordot(op, AAA, ((3, 4, 5), (0, 1, 2)))

def calc_C_conj_mat_op_AA(op, AA):
    return sp.tensordot(op.conj(), AA, ((0, 1), (0, 1)))

def calc_C_func_op(op, A, Ap1):
    q = A.shape[0]
    qp1 = Ap1.shape[0]
    C = sp.zeros((A.shape[0], Ap1.shape[0], A.shape[1], Ap1.shape[2]), dtype=A.dtype)
    for u in xrange(q):
        for v in xrange(qp1):
            AAuv = A[u].dot(Ap1[v])
            for s in xrange(q):
                for t in xrange(qp1):
                    h_nn_stuv = op(s, t, u, v)
                    if h_nn_stuv != 0:
                        C[s, t] += h_nn_stuv * AAuv
    return C
    
def calc_C_func_op_AA(op, AA):
    q = AA.shape[0]
    qp1 = AA.shape[1]
    C = sp.zeros_like(AA)
    for u in xrange(q):
        for v in xrange(qp1):
            AAuv = AA[u, v]
            for s in xrange(q):
                for t in xrange(qp1):
                    h_nn_stuv = op(s, t, u, v)
                    if h_nn_stuv != 0:
                        C[s, t] += h_nn_stuv * AAuv
    return C
    
def calc_K(Kp1, C, lm1, rp1, A, AAp1):    
    K = eps_r_noop(Kp1, A, A)
    
    Hr = eps_r_op_2s_C12_AA34(rp1, C, AAp1)

    op_expect = mm.adot(lm1, Hr)
        
    K += Hr
    
    return K, op_expect
    
def calc_K_l(Km1, Cm1, lm2, r, A, Am1A):
    """Calculates the K_left using the recursive definition.
    
    This is the "bra-vector" K_left, which means (K_left.dot(r)).trace() = <K_left|r>.
    In other words, K_left ~ <K_left| and K_left.conj().T ~ |K_left>.
    """    
    K = eps_l_noop(Km1, A, A)
    
    Hl = eps_l_op_2s_AA12_C34(lm2, Am1A, Cm1)

    op_expect = mm.adot_noconj(Hl, r)
        
    K += Hl
    
    return K, op_expect
    
def calc_K_3s(Kp1, C, lm1, rp2, A, AAp1Ap2):    
    K = eps_r_noop(Kp1, A, A)
    
    Hr = eps_r_op_3s_C123_AAA456(rp2, C, AAp1Ap2)
        
    op_expect = mm.adot(lm1, Hr)
        
    K += Hr
    
    return K, op_expect
    
def calc_K_3s_l(Km1, Cm1, lm3, r, A, Am2Am1A):
    K = eps_l_noop(Km1, A, A)
    
    Hl = eps_l_op_3s_AAA123_C456(lm3, Am2Am1A, Cm1)  
    
    op_expect = mm.adot_noconj(Hl, r)
        
    K += Hl
    
    return K, op_expect
                   
   
def herm_sqrt_inv(x, zero_tol=1E-15, sanity_checks=False, return_rank=False, sc_data=''):
    if isinstance(x,  mm.eyemat):
        x_sqrt = x
        x_sqrt_i = x
        rank = x.shape[0]
    else:
        try:
            ev = x.diag #simple_diag_matrix
            EV = None
        except AttributeError:
            ev, EV = la.eigh(x)
        
        zeros = ev <= zero_tol #throw away negative results too!
        
        ev_sqrt = sp.sqrt(ev)
        
        err = sp.seterr(divide='ignore', invalid='ignore')
        try:
            ev_sqrt_i = 1 / ev_sqrt
            ev_sqrt[zeros] = 0
            ev_sqrt_i[zeros] = 0
        finally:
            sp.seterr(divide=err['divide'], invalid=err['invalid'])
        
        if EV is None:
            x_sqrt = mm.simple_diag_matrix(ev_sqrt, dtype=x.dtype)
            x_sqrt_i = mm.simple_diag_matrix(ev_sqrt_i, dtype=x.dtype)
        else:
            B = mm.mmul_diag(ev_sqrt, EV.conj().T)
            x_sqrt = EV.dot(B)
            
            B = mm.mmul_diag(ev_sqrt_i, EV.conj().T)
            x_sqrt_i = EV.dot(B)
            
        rank = x.shape[0] - np.count_nonzero(zeros)
        
        if sanity_checks:
            if ev.min() < -zero_tol:
                log.warning("Sanity Fail in herm_sqrt_inv(): Throwing away negative eigenvalues! %s %s",
                            ev.min(), sc_data)
            
            if not np.allclose(x_sqrt.dot(x_sqrt), x):
                log.warning("Sanity Fail in herm_sqrt_inv(): x_sqrt is bad! %s %s",
                            la.norm(x_sqrt.dot(x_sqrt) - x), sc_data)
            
            if EV is None: 
                nulls = sp.zeros(x.shape[0])
                nulls[zeros] = 1
                nulls = sp.diag(nulls)
            else: #if we did an EVD then we use the eigenvectors
                nulls = EV.copy()
                nulls[:, sp.invert(zeros)] = 0
                nulls = nulls.dot(nulls.conj().T)
                
            eye = np.eye(x.shape[0])
            if not np.allclose(x_sqrt.dot(x_sqrt_i), eye - nulls):
                log.warning("Sanity Fail in herm_sqrt_inv(): x_sqrt_i is bad! %s %s",
                            la.norm(x_sqrt.dot(x_sqrt_i) - eye + nulls), sc_data)
    
    if return_rank:
        return x_sqrt, x_sqrt_i, rank
    else:
        return x_sqrt, x_sqrt_i
   
def calc_l_r_roots(lm1, r, zero_tol=1E-15, sanity_checks=False, sc_data=''):
    l_sqrt, l_sqrt_i = herm_sqrt_inv(lm1, zero_tol=zero_tol, sanity_checks=sanity_checks, sc_data=(sc_data, 'l'))
    
    r_sqrt, r_sqrt_i = herm_sqrt_inv(r, zero_tol=zero_tol, sanity_checks=sanity_checks, sc_data=(sc_data, 'r'))
    
    return l_sqrt, l_sqrt_i, r_sqrt, r_sqrt_i
    
def calc_Vsh(A, r_s, sanity_checks=False):
    D = A.shape[2]
    Dm1 = A.shape[1]
    q = A.shape[0]
    
    if q * D - Dm1 <= 0:
        return None
    
    R = sp.zeros((D, q, Dm1), dtype=A.dtype, order='C')

    for s in xrange(q):
        R[:,s,:] = r_s.dot(A[s].conj().T)

    R = R.reshape((q * D, Dm1))
    Vconj = ns.nullspace_qr(R.conj().T).T

    if sanity_checks:
        if not sp.allclose(mm.mmul(Vconj.conj(), R), 0):
            log.warning("Sanity Fail in calc_Vsh!: VR != 0")
        if not sp.allclose(mm.mmul(Vconj, Vconj.conj().T), sp.eye(Vconj.shape[0])):
            log.warning("Sanity Fail in calc_Vsh!: V H(V) != eye")
        
    Vconj = Vconj.reshape((q * D - Dm1, D, q))

    Vsh = Vconj.T
    Vsh = sp.asarray(Vsh, order='C')

    if sanity_checks:
        Vs = sp.transpose(Vsh, axes=(0, 2, 1)).conj()
        M = eps_r_noop(r_s, Vs, A)
        if not sp.allclose(M, 0):
            log.warning("Sanity Fail in calc_Vsh!: Bad Vsh")

    return Vsh

def calc_Vsh_l(A, lm1_sqrt, sanity_checks=False):    
    D = A.shape[2]
    Dm1 = A.shape[1]
    q = A.shape[0]
    
    if q * Dm1 - D <= 0:
        return None
    
    L = sp.zeros((D, q, Dm1), dtype=A.dtype, order='C')

    for s in xrange(q):
        L[:,s,:] = lm1_sqrt.dot(A[s]).conj().T

    L = L.reshape((D, q * Dm1))
    V = ns.nullspace_qr(L)

    if sanity_checks:
        if not sp.allclose(L.dot(V), 0):
            log.warning("Sanity Fail in calc_Vsh_l!: LV != 0")
        if not sp.allclose(V.conj().T.dot(V), sp.eye(V.shape[1])):
            log.warning("Sanity Fail in calc_Vsh_l!: V H(V) != eye")
        
    V = V.reshape((q, Dm1, q * Dm1 - D))

    Vsh = sp.transpose(V.conj(), axes=(0, 2, 1))
    Vsh = sp.asarray(Vsh, order='C')

    if sanity_checks:
        M = eps_l_noop(lm1_sqrt, A, V)
        if not sp.allclose(M, 0):
            log.warning("Sanity Fail in calc_Vsh_l!: Bad Vsh")

    return Vsh

   
def calc_x(Kp1, C, Cm1, rp1, lm2, Am1, A, Ap1, lm1_s, lm1_si, r_s, r_si, Vsh):
    D = A.shape[2]
    Dm1 = A.shape[1]
    q = A.shape[0]
    
    x = np.zeros((Dm1, q * D - Dm1), dtype=A.dtype)
    x_part = np.empty_like(x)
    x_subpart = np.empty_like(A[0])
    
    assert not (C is None and not Kp1 is None) #existence of Kp1 implies existence of C
    if not C is None:
        x_part.fill(0)
        for s in xrange(q):            
            x_subpart = eps_r_noop_inplace(rp1, C[s], Ap1, x_subpart) #~1st line
            
            if not Kp1 is None:
                x_subpart += A[s].dot(Kp1) #~3rd line
    
            x_part += x_subpart.dot(r_si.dot(Vsh[s]))

        x += lm1_s.dot(x_part)

    if not lm2 is None:
        x_part.fill(0)
        for s in xrange(q):     #~2nd line
            x_subpart = eps_l_noop_inplace(lm2, Am1, Cm1[:, s], x_subpart)
            x_part += x_subpart.dot(r_s.dot(Vsh[s]))
        x += lm1_si.dot(x_part)

    return x
    
def calc_x_3s(Kp1, C, Cm1, Cm2, rp1, rp2, lm2, lm3, Am2Am1, Am1, A, Ap1, Ap1Ap2, 
              lm1_s, lm1_si, r_s, r_si, Vsh):
    D = A.shape[2]
    Dm1 = A.shape[1]
    q = A.shape[0]
    
    x = np.zeros((Dm1, q * D - Dm1), dtype=A.dtype)
    x_part = np.empty_like(x)
    
    assert not (C is None and not Kp1 is None)
    if not C is None:
        x_part.fill(0)
        for s in xrange(q):            
            x_subpart = eps_r_op_2s_C12_AA34(rp2, C[s], Ap1Ap2) #~1st line
            
            if not Kp1 is None:
                x_subpart += A[s].dot(Kp1) #~3rd line
    
            x_part += x_subpart.dot(r_si.dot(Vsh[s]))
    
        x += lm1_s.dot(x_part)

    if not lm2 is None and not Cm1 is None:
        x_subpart = np.empty((Am1.shape[2], D), dtype=A.dtype)
        x_subsubpart = np.empty((Cm1[0, 0].shape[1], D), dtype=A.dtype)
        qm1 = Am1.shape[0]
        x_part.fill(0)
        for t in xrange(q):     #~2nd line
            x_subpart.fill(0)
            for s in xrange(qm1):
                eps_r_noop_inplace(rp1, Cm1[s, t], Ap1, x_subsubpart)
                x_subpart += (lm2.dot(Am1[s])).conj().T.dot(x_subsubpart)
            x_part += x_subpart.dot(r_si.dot(Vsh[t]))
        x += lm1_si.dot(x_part)

    if not lm3 is None:
        x_part.fill(0)
        for u in xrange(q):     #~2nd line
            x_subpart = eps_l_op_2s_AA12_C34(lm3, Am2Am1, Cm2[:, :, u])
            x_part += x_subpart.dot(r_s.dot(Vsh[u]))
        x += lm1_si.dot(x_part)

    return x
    
def calc_x_l(Km1, C, Cm1, rp1, lm2, Am1, A, Ap1, lm1_s, lm1_si, r_s, r_si, Vsh):
    D = A.shape[2]
    Dm1 = A.shape[1]
    q = A.shape[0]
    
    x = sp.zeros((q * Dm1 - D, D), dtype=A.dtype)
    x_part = sp.empty_like(x)
    x_subpart = sp.empty_like(A[0])
    
    if not C is None:
        x_part.fill(0)
        for s in xrange(q):
            x_subpart = eps_r_noop_inplace(rp1, C[s], Ap1, x_subpart) #~1st line
            x_part += Vsh[s].dot(lm1_s.dot(x_subpart))
            
        try:
            x += r_si.dot_left(x_part)
        except AttributeError:
            x += x_part.dot(r_si)

    
    x_part.fill(0)
    for s in xrange(q):     #~2nd line
        x_subpart.fill(0)

        if not lm2 is None:
            x_subpart = eps_l_noop_inplace(lm2, Am1, Cm1[:, s], x_subpart)
        
        if not Km1 is None:
            x_subpart += Km1.dot(A[s]) #~3rd line
        
        x_part += Vsh[s].dot(lm1_si.dot(x_subpart))
    try:
        x += r_s.dot_left(x_part)
    except AttributeError:
        x += x_part.dot(r_s)

    return x

def calc_BB_Y_2s(C, Vlh, Vrh_p1, l_s_m1, r_s_p1):
    Vr_p1 = sp.transpose(Vrh_p1, axes=(0, 2, 1)).conj()

    Y = sp.zeros((Vlh.shape[1], Vrh_p1.shape[2]), dtype=C.dtype)
    for s in xrange(Vlh.shape[0]):
        Y += Vlh[s].dot(l_s_m1.dot(eps_r_noop(r_s_p1, C[s], Vr_p1)))

    etaBB_sq = mm.adot(Y, Y)
    
    return Y, etaBB_sq
    
def calc_BB_Y_2s_ham_3s(A_m1, A_p2, C, C_m1, Vlh, Vrh_p1, l_m2, r_p2, l_s_m1, l_si_m1, r_s_p1, r_si_p1):
    Vr_p1 = sp.transpose(Vrh_p1, axes=(0, 2, 1)).conj()
    
    Vrri_p1 = sp.zeros_like(Vr_p1)
    try:
        for s in xrange(Vrri_p1.shape[0]):
            Vrri_p1[s] = r_si_p1.dot_left(Vr_p1[s])
    except AttributeError:
        for s in xrange(Vrri_p1.shape[0]):
            Vrri_p1[s] = Vr_p1[s].dot(r_si_p1)
    
    Vl = sp.transpose(Vlh, axes=(0, 2, 1)).conj()        
    liVl = sp.zeros_like(Vl)            
    for s in xrange(liVl.shape[0]):
        liVl[s] = l_si_m1.dot(Vl[s])

    Y = sp.zeros((Vlh.shape[1], Vrh_p1.shape[2]), dtype=Vrh_p1.dtype)
    if not A_p2 is None:
        for s in xrange(C.shape[0]):
            Y += Vlh[s].dot(l_s_m1.dot(eps_r_op_2s_C12(r_p2, C[s], Vrri_p1, A_p2)))
    if not A_m1 is None:
        for u in xrange(C_m1.shape[2]):
            Y += eps_l_op_2s_A1_A2_C34(l_m2, A_m1, liVl, C_m1[:, :, u]).dot(r_s_p1.dot(Vrh_p1[u]))

    etaBB_sq = mm.adot(Y, Y)
    
    return Y, etaBB_sq
    
def calc_BB_2s(Y, Vlh, Vrh_p1, l_si_m1, r_si_p1, dD_max=16, sv_tol=1E-14):
    try:
        U, sv, Vh = la.svd(Y)
    except la.LinAlgError:
        return None, None, 0
    
    dDn = min(sp.count_nonzero(sv > sv_tol), dD_max)
    
    sv = mm.simple_diag_matrix(sv[:dDn])
    
    ss = sv.sqrt()
    
    Z1 = ss.dot_left(U[:, :dDn])
    
    Z2 = ss.dot(Vh[:dDn, :])
    
    BB12n = sp.zeros((Vlh.shape[0], l_si_m1.shape[0], dDn), dtype=Y.dtype)
    
    for s in xrange(Vlh.shape[0]):
        BB12n[s] = l_si_m1.dot(Vlh[s].conj().T).dot(Z1)
    
    BB21np1 = sp.zeros((Vrh_p1.shape[0], dDn, Vrh_p1.shape[1]), dtype=Y.dtype)
    
    try:
        for s in xrange(Vrh_p1.shape[0]):
            BB21np1[s] = r_si_p1.dot_left(Z2.dot(Vrh_p1[s].conj().T))
    except AttributeError:
        for s in xrange(Vrh_p1.shape[0]):
            BB21np1[s] = Z2.dot(Vrh_p1[s].conj().T).dot(r_si_p1)
        
    return BB12n, BB21np1, dDn

def herm_fac_with_inv(A, lower=False, zero_tol=1E-15, return_rank=False, 
                      calc_inv=True, force_evd=False, 
                      sanity_checks=False, sc_data=''):
    """Factorizes a Hermitian matrix using either Cholesky or eigenvalue decomposition.
    
    Decomposes a Hermitian A as A = X*X or, if lower == True, A = XX*.
    
    Tries Cholesky first by default, then falls back to EVD if the matrix is 
    not positive-definite. If Cholesky decomposition is used, X is upper (or lower)
    triangular. For the EVD decomposition, the inverse becomes a pseudo-inverse
    and all eigenvalues below the zero-tolerance are set to zero.
    
    Parameters
    ----------
    A : ndarray
        The Hermitian matrix to be factorized.
    lower : bool
        Refers to Cholesky factorization. If True, factorize as A = XX*, otherwise as A = X*X
    zero_tol : float
        Tolerance for detection of zeros in EVD case.
    return_rank : bool
        Whether to return the rank of A. The detected rank is affected by zero_tol.
    calc_inv : bool
        Whether to calculate (and return) the inverse of the factor.
    force_evd : bool
        Whether to force eigenvalue instead of Cholesky decomposition.
    sanity_checks : bool
        Whether to perform soem basic sanity checks.
    """    
    if not force_evd:
        try:
            x = la.cholesky(A, lower=lower)
            if calc_inv:
                xi = mm.invtr(x, lower=lower)
            else:
                xi = None
            
            nonzeros = A.shape[0]
        except sp.linalg.LinAlgError: #this usually means a is not pos. def.
            force_evd = True
            
    if force_evd:
        ev, EV = la.eigh(A, turbo=True) #wraps lapack routines, which return eigenvalues in ascending order
        
        if sanity_checks:
            assert np.all(ev == np.sort(ev)), "Sanity fail in herm_fac_with_inv(): Unexpected eigenvalue ordering"
            
            if ev.min() < -zero_tol:
                log.warning("Sanity fail in herm_fac_with_inv(): Discarding negative eigenvalues! %s %s",
                            ev.min(), sc_data)
        
        nonzeros = np.count_nonzero(ev > zero_tol) 

        ev_sq = sp.zeros_like(ev, dtype=A.dtype)
        ev_sq[-nonzeros:] = sp.sqrt(ev[-nonzeros:])
        ev_sq = mm.simple_diag_matrix(ev_sq, dtype=A.dtype)
        
        if calc_inv:
            #Replace almost-zero values with zero and perform a pseudo-inverse
            ev_sq_i = sp.zeros_like(ev, dtype=A.dtype)
            ev_sq_i[-nonzeros:] = 1. / ev_sq[-nonzeros:]
            
            ev_sq_i = mm.simple_diag_matrix(ev_sq_i, dtype=A.dtype)        
                   
        xi = None
        if lower:
            x = ev_sq.dot_left(EV)
            if calc_inv:
                xi = ev_sq_i.dot(EV.conj().T)
        else:
            x = ev_sq.dot(EV.conj().T)
            if calc_inv:
                xi = ev_sq_i.dot_left(EV)
            
    if sanity_checks:
        if not sp.allclose(A, A.conj().T, atol=1E-13, rtol=1E-13):
            log.warning("Sanity fail in herm_fac_with_inv(): A is not Hermitian! %s %s",
                        la.norm(A - A.conj().T), sc_data)
        
        eye = sp.zeros((A.shape[0]), dtype=A.dtype)
        eye[-nonzeros:] = 1
        eye = mm.simple_diag_matrix(eye)
        
        if lower:
            if calc_inv:
                if not sp.allclose(xi.dot(x), eye, atol=1E-13, rtol=1E-13):
                    log.warning("Sanity fail in herm_fac_with_inv(): Bad left inverse! %s %s",
                                la.norm(xi.dot(x) - eye), sc_data)
                                
                if not sp.allclose(xi.dot(A).dot(xi.conj().T), eye, atol=1E-13, rtol=1E-13):
                    log.warning("Sanity fail in herm_fac_with_inv(): Bad A inverse! %s %s",
                                la.norm(xi.conj().T.dot(A).dot(xi) - eye), sc_data)
    
            if not sp.allclose(x.dot(x.conj().T), A, atol=1E-13, rtol=1E-13):
                log.warning("Sanity fail in herm_fac_with_inv(): Bad decomp! %s %s",
                            la.norm(x.dot(x.conj().T) - A), sc_data)
        else:
            if calc_inv:
                if not sp.allclose(x.dot(xi), eye, atol=1E-13, rtol=1E-13):
                    log.warning("Sanity fail in herm_fac_with_inv(): Bad right inverse! %s %s",
                                la.norm(x.dot(xi) - eye), sc_data)
                if not sp.allclose(xi.conj().T.dot(A).dot(xi), eye, atol=1E-13, rtol=1E-13):
                    log.warning("Sanity fail in herm_fac_with_inv(): Bad A inverse! %s %s",
                                la.norm(xi.conj().T.dot(A).dot(xi) - eye), sc_data)

    
            if not sp.allclose(x.conj().T.dot(x), A, atol=1E-13, rtol=1E-13):
                log.warning("Sanity fail in herm_fac_with_inv(): Bad decomp! %s %s",
                            la.norm(x.conj().T.dot(x) - A), sc_data)
                    
    if calc_inv:
        if return_rank:
            return x, xi, nonzeros
        else:
            return x, xi
    else:
        if return_rank:
            return x, nonzeros
        else:
            return x
        
def restore_RCF_r_seq(A, r, GN=None, sanity_checks=False, sc_data=''):
    """Transforms a sequence of A[n]'s to obtain r[n - 1] = eye(D).
    
    Implements the condition for right-orthonormalization from sub-section
    3.1, theorem 1 of arXiv:quant-ph/0608197v2.
    
    Uses a reduced QR decomposition to avoid inverting anything explicity.
    
    Parameters
    ----------
    A : sequence of ndarray
        The parameter tensors for a sequence of sites [None, A1, A2,..., AN].
        The first entry is ignored so that the indices match up with r.
    r : sequence of ndarray or objects with array interface
        The matrices [r0, r1, r2,..., rN], where rN will not be changed, but is 
        used for sanity checks.
    GN : ndarray or scalar
        Initial right gauge transformation matrix for site N. Only needed when used
        as part of a larger transformation.
    sanity_checks : bool (False)
        Whether to perform additional sanity checks.
    sc_data : string
        A string to be appended to sanity check log messages.
    """
    assert len(A) == len(r), 'A and r must have the same length!'
    if GN is None:
        Gh = mm.eyemat(A[-1].shape[2], dtype=A[-1].dtype)
    else:
        Gh = GN.conj().T
    for n in xrange(len(A) - 1, 0, -1):
        q, Dm1, D = A[n].shape
        AG = sp.array([Gh.dot(As.conj().T) for As in A[n]]).reshape((q * D, Dm1))
        Q, R = la.qr(AG, mode='economic')
        A[n] = sp.transpose(Q.conj().reshape((q, D, Dm1)), axes=(0, 2, 1))
        Gh = R
        
        r[n - 1] = mm.eyemat(Dm1, dtype=A[n].dtype)
        
        if sanity_checks:
            r_nm1_ = eps_r_noop(r[n], A[n], A[n])
            if not sp.allclose(r_nm1_, r[n - 1].A, atol=1E-13, rtol=1E-13):
                log.warning("Sanity Fail in restore_RCF_r!: r is bad! %s %s",
                            la.norm(r_nm1_ - r[n - 1]), sc_data)
        
    return Gh.conj().T
    
def restore_RCF_l_seq(A, l, G0=None, sanity_checks=False, sc_data=''):
    """Transforms a sequence of A to obtain diagonal l.
    
    See restore_RCF_l.
    
    Parameters
    ----------
    A : sequence of ndarray
        The parameter tensors for a sequence of sites [None, A1, A2,..., AN].
        The first entry is ignored so that the indices match up with l.
    l : sequence of ndarray or objects with array interface
        The matrices [l0, l1, l2,..., lN], where l0 will not be changed, but is 
        used for sanity checks.
    G0 : ndarray
        Initial left gauge transformation matrix for site 1. Only needed when used
        as part of a larger transformation.
    sanity_checks : bool (False)
        Whether to perform additional sanity checks.
    sc_data : string
        A string to be appended to sanity check log messages.
    """
    assert len(A) == len(l), 'A and l must have the same length!'
    
    if G0 is None:
        G = sp.eye(A[1].shape[1], dtype=A[1].dtype)
    else:
        G = G0
        
    for n in xrange(1, len(A)):
        l[n], G, Gi = restore_RCF_l(A[n], l[n - 1], G, sanity_checks)
    
    return G
    

def restore_RCF_r(A, r, G_n_i, zero_tol=1E-15, sanity_checks=False, sc_data=''):
    """Transforms a single A[n] to obtain r[n - 1] = eye(D).

    Implements the condition for right-orthonormalization from sub-section
    3.1, theorem 1 of arXiv:quant-ph/0608197v2.

    This function must be called for each n in turn, starting at N + 1,
    passing the gauge transformation matrix from the previous step
    as an argument.

    Finds a G[n-1] such that orthonormalization is fulfilled for n.

    If rank-deficiency is encountered, the result fulfills the orthonormality
    condition in the occupied subspace with the zeros at the top-left
    (for example r = diag([0, 0, 1, 1, 1, 1, 1])).

    Parameters
    ----------
    A : ndarray
        The parameter tensor for the nth site A[n].
    r : ndarray or object with array interface
        The matrix r[n].
    G_n_i : ndarray
        The inverse gauge transform matrix for site n obtained in the previous step (for n + 1).
    sanity_checks : bool (False)
        Whether to perform additional sanity checks.
    zero_tol : float
        Tolerance for detecting zeros.
            
    Returns
    -------
    r_nm1 : ndarray or simple_diag_matrix or eyemat
        The new matrix r[n - 1].
    G_nm1 : ndarray
        The gauge transformation matrix for the site n - 1.
    G_n_m1_i : ndarray
        The inverse gauge transformation matrix for the site n - 1.
    """
    if G_n_i is None:
        GGh_n_i = r
    else:
        GGh_n_i = G_n_i.dot(r.dot(G_n_i.conj().T))

    M = eps_r_noop(GGh_n_i, A, A)
    
    X, Xi, new_D = herm_fac_with_inv(M, zero_tol=zero_tol, return_rank=True, 
                                     sanity_checks=sanity_checks)
                                     
    G_nm1 = Xi.conj().T
    G_nm1_i = X.conj().T

    if G_n_i is None:
        G_n_i = G_nm1_i

    if sanity_checks:     
        #GiG may not be equal to eye in the case of rank-deficiency,
        #but the rest should lie in the null space of A.
        GiG = G_nm1_i.dot(G_nm1)
        As = np.sum(A, axis=0)
        if not sp.allclose(GiG.dot(As).dot(G_n_i), 
                           As.dot(G_n_i), atol=1E-13, rtol=1E-13):
            log.warning("Sanity Fail in restore_RCF_r!: Bad GT! %s %s",
                        la.norm(GiG.dot(As).dot(G_n_i) - As.dot(G_n_i)), sc_data)

    for s in xrange(A.shape[0]):
        A[s] = G_nm1.dot(A[s]).dot(G_n_i)

    if new_D == A.shape[1]:
        r_nm1 = mm.eyemat(A.shape[1], A.dtype)
    else:
        r_nm1 = sp.zeros((A.shape[1]), dtype=A.dtype)
        r_nm1[-new_D:] = 1
        r_nm1 = mm.simple_diag_matrix(r_nm1, dtype=A.dtype)

    if sanity_checks:
        r_nm1_ = G_nm1.dot(M).dot(G_nm1.conj().T)
        if not sp.allclose(r_nm1_, r_nm1.A, atol=1E-13, rtol=1E-13):
            log.warning("Sanity Fail in restore_RCF_r!: r != g old_r gH! %s %s",
                        la.norm(r_nm1_ - r_nm1), sc_data)
        
        r_nm1_ = eps_r_noop(r, A, A)
        if not sp.allclose(r_nm1_, r_nm1.A, atol=1E-13, rtol=1E-13):
            log.warning("Sanity Fail in restore_RCF_r!: r is bad! %s %s",
                        la.norm(r_nm1_ - r_nm1), sc_data)

    return r_nm1, G_nm1, G_nm1_i

def restore_RCF_l(A, lm1, Gm1, sanity_checks=False):
    """Transforms a single A[n] to obtain diagonal l[n].

    Applied after restore_RCF_r(), this completes the full canonical form
    of sub-section 3.1, theorem 1 of arXiv:quant-ph/0608197v2.

    This function must be called for each n in turn, starting at 1,
    passing the gauge transformation matrix from the previous step
    as an argument.

    Finds a G[n] such that orthonormalization is fulfilled for n.

    The diagonal entries of l[n] are sorted in
    ascending order (for example l[n] = diag([0, 0, 0.1, 0.2, ...])).

    Parameters
    ----------
    A : ndarray
        The parameter tensor for the nth site A[n].
    lm1 : ndarray or object with array interface
        The matrix l[n - 1].
    Gm1 : ndarray
        The gauge transform matrix for site n obtained in the previous step (for n - 1).
    sanity_checks : bool (False)
        Whether to perform additional sanity checks.
        
    Returns
    -------
    l : ndarray or simple_diag_matrix
        The new, diagonal matrix l[n]. 
    G : ndarray
        The gauge transformation matrix for site n.
    G_i : ndarray
        Inverse of G.
    """
    if Gm1 is None:
        x = lm1
    else:
        x = Gm1.conj().T.dot(lm1.dot(Gm1))

    M = eps_l_noop(x, A, A)
    ev, EV = la.eigh(M) #wraps lapack routines, which return eigenvalues in ascending order
    
    if sanity_checks:
        assert np.all(ev == np.sort(ev)), "unexpected eigenvalue ordering"
    
    l = mm.simple_diag_matrix(ev, dtype=A.dtype)
    G_i = EV

    if Gm1 is None:
        Gm1 = EV.conj().T #for left uniform case
        lm1 = l #for sanity check

    for s in xrange(A.shape[0]):
        A[s] = Gm1.dot(A[s].dot(G_i))

    if sanity_checks:
        l_ = eps_l_noop(lm1, A, A)
        if not sp.allclose(l_, l, atol=1E-12, rtol=1E-12):
            log.warning("Sanity Fail in restore_RCF_l!: l is bad!")
            log.warning(la.norm(l_ - l))

    G = EV.conj().T

    return l, G, G_i

def restore_LCF_l_seq(A, l, G0=None, sanity_checks=False, sc_data=''):
    """Transforms a sequence of A[n]'s to obtain l[n] = eye(D).
    
    Implements the condition for left-orthonormalization.
    
    Uses a reduced QR (RQ) decomposition to avoid inverting anything explicity.
    
    Parameters
    ----------
    A : sequence of ndarray
        The parameter tensors for a sequence of sites [None, A1, A2,..., AN].
        The first entry is ignored so that the indices match up with l.
    l : sequence of ndarray or objects with array interface
        The matrices [l0, l1, l2,..., lN], where l0 will not be changed, but is 
        used for sanity checks.
    G0 : ndarray or scalar
        Initial left gauge transformation matrix for site 0. Only needed when used
        as part of a larger transformation.
    sanity_checks : bool (False)
        Whether to perform additional sanity checks.
    sc_data : string
        A string to be appended to sanity check log messages.
    """
    if G0 is None:
        G = mm.eyemat(A[1].shape[1], dtype=A[1].dtype)
    else:
        G = G0

    for n in xrange(1, len(A)):
        q, Dm1, D = A[n].shape
        GA = sp.array([G.dot(As) for As in A[n]])
        GA = GA.reshape((q * Dm1, D))
        Q, G = la.qr(GA, mode='economic')
        A[n] = Q.reshape((q, Dm1, D))
        
        l[n] = mm.eyemat(D, dtype=A[n].dtype)
        
        if sanity_checks:
            l_ = eps_l_noop(l[n - 1], A[n], A[n])
            if not sp.allclose(l_, l[n].A, atol=1E-13, rtol=1E-13):
                log.warning("Sanity Fail in restore_LCF_l_seq!: l is bad")
                log.warning(la.norm(l_ - l[n].A))
        
    return G
        
def restore_LCF_r_seq(A, r, GiN=None, sanity_checks=False, sc_data=''):
    """
    Transforms a sequence of A to obtain diagonal r.
    
    See restore_LCF_r.
    
    Parameters
    ----------
    A : sequence of ndarray
        The parameter tensors for a sequence of sites [None, A1, A2,..., AN].
        The first entry is ignored so that the indices match up with r.
    r : sequence of ndarray or objects with array interface
        The matrices [r0, r1, r2,..., rN], where rN will not be changed, but is 
        used for sanity checks.
    GiN : ndarray or scalar
        Initial right gauge transformation matrix for site N. Only needed when used
        as part of a larger transformation.
    sanity_checks : bool (False)
        Whether to perform additional sanity checks.
    sc_data : string
        A string to be appended to sanity check log messages.
    """
    if GiN is None:
        Gi = sp.eye(A[-1].shape[2], dtype=A[-1].dtype)
    else:
        Gi = GiN
    for n in xrange(len(A) - 1, 0, -1):
        r[n - 1], G, Gi = restore_LCF_r(A[n], r[n], Gi, sanity_checks)

    return Gi

def restore_LCF_l(A, lm1, Gm1, sanity_checks=False, zero_tol=1E-15):
    if Gm1 is None:
        GhGm1 = lm1
    else:
        GhGm1 = Gm1.conj().T.dot(lm1.dot(Gm1))

    M = eps_l_noop(GhGm1, A, A)
    
    G, Gi, new_D = herm_fac_with_inv(M, zero_tol=zero_tol, return_rank=True, 
                                     sanity_checks=sanity_checks)

    if Gm1 is None:
        Gm1 = G

    if sanity_checks:
        if new_D == A.shape[2]:
            eye = sp.eye(A.shape[2])
        else:
            eye = mm.simple_diag_matrix(np.append(np.zeros(A.shape[2] - new_D),
                                                   np.ones(new_D)), dtype=A.dtype)
        if not sp.allclose(G.dot(Gi), eye, atol=1E-13, rtol=1E-13):
            log.warning("Sanity Fail in restore_LCF_l!: Bad GT!")

    for s in xrange(A.shape[0]):
        A[s] = Gm1.dot(A[s]).dot(Gi)

    if new_D == A.shape[2]:
        l = mm.eyemat(A.shape[2], A.dtype)
    else:
        l = mm.simple_diag_matrix(np.append(np.zeros(A.shape[2] - new_D),
                                                np.ones(new_D)), dtype=A.dtype)

    if sanity_checks:
        lm1_ = mm.eyemat(A.shape[1], A.dtype)

        l_ = eps_l_noop(lm1_, A, A)
        if not sp.allclose(l_, l.A, atol=1E-13, rtol=1E-13):
            log.warning("Sanity Fail in restore_LCF_l!: l is bad")
            log.warning(la.norm(l_ - l))

    return l, G, Gi
    
def restore_LCF_r(A, r, Gi, sanity_checks=False):
    if Gi is None:
        x = r
    else:
        x = Gi.dot(r.dot(Gi.conj().T))

    M = eps_r_noop(x, A, A)
    ev, EV = la.eigh(M) #wraps lapack routines, which return eigenvalues in ascending order
    
    if sanity_checks:
        assert np.all(ev == np.sort(ev)), "unexpected eigenvalue ordering"
    
    rm1 = mm.simple_diag_matrix(ev, dtype=A.dtype)
    Gm1 = EV.conj().T

    if Gi is None:
        Gi = EV #for left uniform case
        r = rm1 #for sanity check

    for s in xrange(A.shape[0]):
        A[s] = Gm1.dot(A[s].dot(Gi))

    if sanity_checks:
        rm1_ = eps_r_noop(r, A, A)
        if not sp.allclose(rm1_, rm1, atol=1E-12, rtol=1E-12):
            log.warning("Sanity Fail in restore_LCF_r!: r is bad!")
            log.warning(la.norm(rm1_ - rm1))

    Gm1_i = EV

    return rm1, Gm1, Gm1_i
