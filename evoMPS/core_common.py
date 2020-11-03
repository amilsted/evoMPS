# -*- coding: utf-8 -*-
"""
Created on Tue Jan 20 17:43:17 2015

@author: ash
"""
from __future__ import absolute_import, division, print_function

import numpy as np
import scipy.linalg as la
from . import matmul as mm

def calc_AA(A, Ap1):
    Dp1 = Ap1.shape[2]
    Dm1 = A.shape[1]
    q = A.shape[0]
    qp1 = Ap1.shape[0]
    
    AA = np.zeros((q, qp1, Dm1, Dp1), dtype=A.dtype)
    for u in range(q):
        for v in range(qp1):
            np.dot(A[u], Ap1[v], out=AA[u, v])
    
    return AA
    
    #This works too: (just for reference)
    #AA = np.array([dot(A[s], A[t]) for s in xrange(self.q) for t in xrange(self.q)])
    #self.AA = AA.reshape(self.q, self.q, self.D, self.D)

    #So does this
    #return np.transpose(np.tensordot(A, Ap1, axes=((2,),(1,))), (0,2,1,3))

def calc_AAA(A, Ap1, Ap2):
    Dp2 = Ap2.shape[2]
    Dp1 = Ap1.shape[2]
    Dm1 = A.shape[1]
    q = A.shape[0]
    qp1 = Ap1.shape[0]
    qp2 = Ap2.shape[0]
    
    AAA = np.zeros((q, qp1, qp2, Dm1, Dp2), dtype=A.dtype)
    tmp = np.zeros((Dm1, Dp1), dtype=A.dtype)
    for u in range(q):
        for v in range(qp1):
            for w in range(qp2):
                np.dot(A[u], Ap1[v], out=tmp)
                np.dot(tmp, Ap2[w], out=AAA[u, v, w])
                #AAA[u, v, w] = A[u].dot(Ap1[v]).dot(Ap2[w])
    
    return AAA
    
def calc_AAA_AA(AA, Ap2):
    Dp2 = Ap2.shape[2]
    Dm1 = AA.shape[2]
    q = AA.shape[0]
    qp1 = AA.shape[1]
    qp2 = Ap2.shape[0]
    
    AAA = np.zeros((q, qp1, qp2, Dm1, Dp2), dtype=AA.dtype)
    for u in range(q):
        for v in range(qp1):
            for w in range(qp2):
                np.dot(AA[u, v], Ap2[w], out=AAA[u, v, w])
    
    return AAA

def calc_AAA_AAr(A, AAp1):
    Dp2 = AAp1.shape[3]
    Dm1 = A.shape[1]
    q = A.shape[0]
    qp1 = AAp1.shape[0]
    qp2 = AAp1.shape[1]
    
    AAA = np.zeros((q, qp1, qp2, Dm1, Dp2), dtype=A.dtype)
    for u in range(q):
        for v in range(qp1):
            for w in range(qp2):
                np.dot(A[u], AAp1[v, w], out=AAA[u, v, w])
    
    return AAA
    
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
    return eps_l_noop_inplace(x, A1, A2, out)
    
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
    
    #Two extra temporaries required because scipy doesn't bother to provide full gemm functionality.
    tmp_A1sh = np.empty((A1[0].shape[1], A1[0].shape[0]), dtype=A1[0].dtype, order='F')
    tmp_xA2s = np.empty((x.shape[0], A2[0].shape[1]), dtype=np.promote_types(x.dtype, A2[0].dtype))
    out_s = np.empty_like(out, order='C')
    
    for s in range(A1.shape[0]):
        tmp_A1sh[:] = A1[s].T
        np.conjugate(tmp_A1sh, out=tmp_A1sh)
        tmp_xA2s = mm.dot_inplace(x, A2[s], tmp_xA2s)
        out_s = np.dot(tmp_A1sh, tmp_xA2s, out=out_s) #dot expects a C-ordered output array
        out += out_s
        
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
    return eps_r_noop_inplace(x, A1, A2, out)
    
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
    
    tmp_A1xs = np.empty((A1[0].shape[0], x.shape[1]), dtype=np.promote_types(A1.dtype, x.dtype))
    tmp_A2sh = np.empty((A2[0].shape[1], A2[0].shape[0]), dtype=A2[0].dtype, order='F')
    out_s = np.empty_like(out, order='C')
    
    for s in range(A1.shape[0]):
        tmp_A1xs = mm.dot_inplace(A1[s], x, tmp_A1xs)
        tmp_A2sh[:] = A2[s].T
        np.conjugate(tmp_A2sh, out=tmp_A2sh)
        out_s = np.dot(tmp_A1xs, tmp_A2sh, out=out_s)
        out += out_s
        
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
    for s in range(A1.shape[0]):
        for t in range(A1.shape[0]):
            o_st = op[t, s]
            if o_st != 0:
                out += o_st * A1[s].conj().T.dot(x.dot(A2[t]))
    return out

#WIP
def eps_l_op_MPO(x, A1, A2, op): #x: [M,Dket,Dbra]
    op = op.conj() #[M1,M2,pbra,pket]
    A1x = np.tensordot(A1.conj(), x, axes=((1), (1))) #[p,D2ket, xM,xD2bra]
    A1xop = np.tensordot(op, A1x, axes=((0,3), (2,0))) #[M2,pbra, D2ket,xD2bra]
    res = np.tensordot(A1xop, A2, axes=((1,3),(0,1))) #[M2,D2ket,D2bra]
    return res

#WIP
def eps_r_op_MPO(x, A1, A2, op): #x: [M,Dket,Dbra], op: [M1,M2,pbra,pket]
    A1x = np.tensordot(A1, x, axes=((2), (1))) #[p,D1ket, xM,xD2bra]
    A1xop = np.tensordot(op, A1x, axes=((1,3), (2,0))) #[M1,pbra, D1ket,xD2bra]
    res = np.tensordot(A1xop, A2.conj(), axes=((1,3),(0,2))) #[M1,D1ket,D1bra]
    return res
    
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
    
    #TODO: Split into groups that can be processed seperately as successive eps_r's?
    
    assert np.all([len(At.shape) >= 3 for At in A1 + A2]), "Invalid input shapes"
    
    #Flatten site indices within each tensor
    A1 = [A1t.reshape((np.prod(A1t.shape[:-2]), A1t.shape[-2], A1t.shape[-1])) for A1t in A1]
    A2 = [A2t.reshape((np.prod(A2t.shape[:-2]), A2t.shape[-2], A2t.shape[-1])) for A2t in A2]
    
    nA1 = len(A1)
    nA2 = len(A2)
    
    A1dims = np.array([1] + [A1t.shape[0] for A1t in reversed(A1)])
    A1dims_prod = np.cumprod(A1dims)
    S = A1dims_prod[-1]
    #print A1dims, A1dims_prod, S
    
    A2dims = np.array([1] + [A2t.shape[0] for A2t in reversed(A2)])
    A2dims_prod = np.cumprod(A2dims)
    #print A2dims, A2dims_prod, S
    
    out = np.zeros((A1[0].shape[1], A2[0].shape[1]), dtype=A1[0].dtype)
    
    for s in range(S):
        A1s_prod = A1[nA1 - 1][s % A1dims[1]]
        for t in range(1, nA1):
            ind = (s // A1dims_prod[t]) % A1dims[t + 1]
            A1s_prod = np.dot(A1[nA1 - t - 1][ind], A1s_prod)
            
#        A1ind = [(s // A1dims_prod[t]) % A1dims[t + 1] for t in xrange(len(A1))]
#        A1s = [A1[t][A1ind[-(t + 1)]] for t in xrange(len(A1))]
#        A1s_prod = reduce(np.dot, A1s)

        A2s_prod = A2[nA2 - 1][s % A2dims[1]]
        for t in range(1, nA2):
            ind = (s // A2dims_prod[t]) % A2dims[t + 1]
            A2s_prod = np.dot(A2[nA2 - t - 1][ind], A2s_prod)
        
#        A2ind = [(s // A2dims_prod[t]) % A2dims[t + 1] for t in xrange(len(A2))]
#        A2s = [A2[t][A2ind[-(t + 1)]] for t in xrange(len(A2))]
#        A2s_prod = reduce(np.dot, A2s)
        
        #print A1s_prod.shape, x.shape, A2s_prod.conj().T.shape
        
        out += A1s_prod.dot(x.dot(A2s_prod.conj().T))
    return out
    
    
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
    out = np.zeros((A1.shape[1], A2.shape[1]), dtype=A1.dtype)
    for s in range(A1.shape[0]):
        xA2s = x.dot(A2[s].conj().T)
        for t in range(A1.shape[0]):
            o_st = op[s, t]
            if o_st != 0:
                out += o_st * A1[t].dot(xA2s)
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
    for u in range(A3.shape[0]):
        for v in range(A4.shape[0]):
            subres = zeros((A1.shape[1], A2.shape[2]), dtype=A1.dtype)
            for s in range(A1.shape[0]):
                for t in range(A2.shape[0]):
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
    for u in range(A3.shape[0]):
        for v in range(A4.shape[0]):
            subres = zeros_like(AA12[0, 0])
            for s in range(AA12.shape[0]):
                for t in range(AA12.shape[1]):
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
    for u in range(AA34.shape[0]):
        for v in range(AA34.shape[1]):
            subres = zeros_like(AA12[0, 0])
            for s in range(AA12.shape[0]):
                for t in range(AA12.shape[1]):
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
    for u in range(A3.shape[0]):
        for v in range(A4.shape[0]):
            res += C12[u, v].dot(x.dot((A3[u].dot(A4[v])).conj().T))
    return res
    
def eps_r_op_2s_C34(x, A1, A2, C34):
    res = np.zeros((A1.shape[1], C34.shape[2]), dtype=A1.dtype)
    for u in range(C34.shape[0]):
        for v in range(C34.shape[1]):
            res += A1[u].dot(A2[v]).dot(x.dot(C34[u, v].conj().T))
    return res
    
def calc_C_func_op(op, A, Ap1):
    q = A.shape[0]
    qp1 = Ap1.shape[0]
    C = np.zeros((A.shape[0], Ap1.shape[0], A.shape[1], Ap1.shape[2]), dtype=A.dtype)
    for u in range(q):
        for v in range(qp1):
            AAuv = A[u].dot(Ap1[v])
            for s in range(q):
                for t in range(qp1):
                    h_nn_stuv = op(s, t, u, v)
                    if h_nn_stuv != 0:
                        C[s, t] += h_nn_stuv * AAuv
    return C
    
def calc_C_func_op_AA(op, AA):
    q = AA.shape[0]
    qp1 = AA.shape[1]
    C = np.zeros_like(AA)
    for u in range(q):
        for v in range(qp1):
            AAuv = AA[u, v]
            for s in range(q):
                for t in range(qp1):
                    h_nn_stuv = op(s, t, u, v)
                    if h_nn_stuv != 0:
                        C[s, t] += h_nn_stuv * AAuv
    return C
