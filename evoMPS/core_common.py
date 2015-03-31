# -*- coding: utf-8 -*-
"""
Created on Tue Jan 20 17:43:17 2015

@author: ash
"""

import numpy as np

def calc_AA(A, Ap1):
    Dp1 = Ap1.shape[2]
    Dm1 = A.shape[1]
    q = A.shape[0]
    qp1 = Ap1.shape[0]
    
    AA = np.zeros((q, qp1, Dm1, Dp1), dtype=A.dtype)
    for u in xrange(q):
        for v in xrange(qp1):
            np.dot(A[u], Ap1[v], out=AA[u, v])
    
    return AA
    
    #This works too: (just for reference)
    #AA = np.array([dot(A[s], A[t]) for s in xrange(self.q) for t in xrange(self.q)])
    #self.AA = AA.reshape(self.q, self.q, self.D, self.D)

def calc_AAA(A, Ap1, Ap2):
    Dp2 = Ap2.shape[2]
    Dp1 = Ap1.shape[2]
    Dm1 = A.shape[1]
    q = A.shape[0]
    qp1 = Ap1.shape[0]
    qp2 = Ap2.shape[0]
    
    AAA = np.zeros((q, qp1, qp2, Dm1, Dp2), dtype=A.dtype)
    tmp = np.zeros((Dm1, Dp1), dtype=A.dtype)
    for u in xrange(q):
        for v in xrange(qp1):
            for w in xrange(qp2):
                np.dot(A[u], Ap1[v], out=tmp)
                np.dot(tmp, Ap2[w], out=AAA[u, v, w])
                #AAA[u, v, w] = A[u].dot(Ap1[v]).dot(Ap2[w])
    
    return AAA
    
def calc_AAA_AA(AAp1, Ap2):
    Dp2 = Ap2.shape[2]
    Dm1 = AAp1.shape[2]
    q = AAp1.shape[0]
    qp1 = AAp1.shape[1]
    qp2 = Ap2.shape[0]
    
    AAA = np.zeros((q, qp1, qp2, Dm1, Dp2), dtype=AAp1.dtype)
    for u in xrange(q):
        for v in xrange(qp1):
            for w in xrange(qp2):
                np.dot(AAp1[u, v], Ap2[w], out=AAA[u, v, w])
    
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
    
    for s in xrange(S):
        A1s_prod = A1[nA1 - 1][s % A1dims[1]]
        for t in xrange(1, nA1):
            ind = (s / A1dims_prod[t]) % A1dims[t + 1]
            A1s_prod = np.dot(A1[nA1 - t - 1][ind], A1s_prod)
            
#        A1ind = [(s / A1dims_prod[t]) % A1dims[t + 1] for t in xrange(len(A1))]
#        A1s = [A1[t][A1ind[-(t + 1)]] for t in xrange(len(A1))]
#        A1s_prod = reduce(np.dot, A1s)

        A2s_prod = A2[nA2 - 1][s % A2dims[1]]
        for t in xrange(1, nA2):
            ind = (s / A2dims_prod[t]) % A2dims[t + 1]
            A2s_prod = np.dot(A2[nA2 - t - 1][ind], A2s_prod)
        
#        A2ind = [(s / A2dims_prod[t]) % A2dims[t + 1] for t in xrange(len(A2))]
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
    out = np.zeros((A1.shape[1], A1.shape[1]), dtype=A1.dtype)
    for s in xrange(A1.shape[0]):
        xA2s = x.dot(A2[s].conj().T)
        for t in xrange(A1.shape[0]):
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
    
def eps_r_op_2s_AA12_C34(x, AA12, C34):
    d = C34.shape[0] * C34.shape[1]
    S1 = (d, AA12.shape[2], AA12.shape[3])
    S2 = (d, C34.shape[2], C34.shape[3])
    return eps_r_noop(x, AA12.reshape(S1), C34.reshape(S2))
    
def eps_l_op_2s_AA12_C34(x, AA12, C34):
    d = AA12.shape[0] * AA12.shape[1]
    S1 = (d, AA12.shape[2], AA12.shape[3])
    S2 = (d, C34.shape[2], C34.shape[3])
    return eps_l_noop(x, AA12.reshape(S1), C34.reshape(S2))
    
def eps_l_op_2s_A1_A2_C34(x, A1, A2, C34):
    res = np.zeros((A2.shape[2], C34.shape[3]), dtype=C34.dtype)
    for u in xrange(C34.shape[0]):
        for v in xrange(C34.shape[1]):
            res += (A1[u].dot(A2[v])).conj().T.dot(x.dot(C34[u, v]))
    return res

def eps_r_op_3s_C123_AAA456(x, C123, AAA456):
    d = C123.shape[0] * C123.shape[1] * C123.shape[2]
    S1 = (d, C123.shape[3], C123.shape[4])
    S2 = (d, AAA456.shape[3], AAA456.shape[4])
    return eps_r_noop(x, C123.reshape(S1), AAA456.reshape(S2))
    
def eps_l_op_3s_AAA123_C456(x, AAA123, C456):
    d = C456.shape[0] * C456.shape[1] * C456.shape[2]
    S1 = (d, AAA123.shape[3], AAA123.shape[4])
    S2 = (d, C456.shape[3], C456.shape[4])
    return eps_l_noop(x, AAA123.reshape(S1), C456.reshape(S2))

def calc_C_mat_op_AA_tensordot(op, AA):
    return np.tensordot(op, AA, ((2, 3), (0, 1)))

def calc_C_mat_op_AA(op, AA):
    AA_ = AA.reshape((AA.shape[0] * AA.shape[1], AA.shape[2] * AA.shape[3]))
    op_ = op.reshape((op.shape[0] * op.shape[1], op.shape[2] * op.shape[3]))
    C_ = op_.dot(AA_)
    return C_.reshape(AA.shape)
    
def calc_C_3s_mat_op_AAA(op, AAA):
    AAA_ = AAA.reshape((AAA.shape[0] * AAA.shape[1] * AAA.shape[2], 
                        AAA.shape[3] * AAA.shape[4]))
    op_ = op.reshape((op.shape[0] * op.shape[1] * op.shape[2], 
                      op.shape[3] * op.shape[4] * op.shape[5]))
    C_ = op_.dot(AAA_)
    return C_.reshape(AAA.shape)
    
def calc_C_3s_mat_op_AAA_tensordot(op, AAA):
    return np.tensordot(op, AAA, ((3, 4, 5), (0, 1, 2)))

def calc_C_conj_mat_op_AA(op, AA):
    AA_ = AA.reshape((AA.shape[0] * AA.shape[1], AA.shape[2] * AA.shape[3]))
    op_ = op.reshape((op.shape[0] * op.shape[1], op.shape[2] * op.shape[3]))
    C_ = op_.conj().T.dot(AA_)
    return C_.reshape(AA.shape)
    
def calc_C_conj_mat_op_AA_tensordot(op, AA):
    return np.tensordot(op.conj(), AA, ((0, 1), (0, 1)))
    
def calc_C_mat_op_tp(op_tp, A, Ap1):
    """op_tp contains the terms of a tensor product decomposition of a
       nearest-neighbour operator.
    """
    C = np.zeros((A.shape[0], Ap1.shape[0], A.shape[1], Ap1.shape[2]), dtype=A.dtype)
    for op_tp_ in op_tp:
        A_op0 = op_tp_[0].dot(A.reshape((A.shape[0], A.shape[1] * A.shape[2]))).reshape(A.shape)
        Ap1_op1 = op_tp_[1].dot(Ap1.reshape((Ap1.shape[0], Ap1.shape[1] * Ap1.shape[2]))).reshape(Ap1.shape)
        
        C += calc_AA(A_op0, Ap1_op1)
        
    return C

def calc_C_func_op(op, A, Ap1):
    q = A.shape[0]
    qp1 = Ap1.shape[0]
    C = np.zeros((A.shape[0], Ap1.shape[0], A.shape[1], Ap1.shape[2]), dtype=A.dtype)
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
    C = np.zeros_like(AA)
    for u in xrange(q):
        for v in xrange(qp1):
            AAuv = AA[u, v]
            for s in xrange(q):
                for t in xrange(qp1):
                    h_nn_stuv = op(s, t, u, v)
                    if h_nn_stuv != 0:
                        C[s, t] += h_nn_stuv * AAuv
    return C