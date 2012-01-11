# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 13:05:59 2011

@author: Ashley Milsted
"""

import numpy as np
import scipy as sp
import scipy.linalg as la
from scipy import *

gemm = None
using_fblas = False

def _matmul_gemm(out, args):
    if len(args) < 3:
        res = args[0]
    else:
        res = gemm(1., args[0], args[1])    
        for x in args[2:-1]:
            res = gemm(1., res, x)
            
    if out is None:
        #out = empty((args[0].shape[0], args[-1].shape[1]), dtype=args[0].dtype)
        return gemm(1., res, args[-1])
    elif la.blas.has_column_major_storage(out) == using_fblas: #'out' must have the right layout for the gemm function...   
        return gemm(1., res, args[-1], c=out, overwrite_c=True)
    else:
        out[...] = gemm(1., res, args[-1])
        return out
    

def _matmul_dot(out, args): #depending on the arguments, dot() may expect a matrix or an array as output... problems?
    if len(args) < 3:
        res = args[0]
    else:
        res = dot(args[0], args[1])    
        
        for x in args[2:-1]:
            res = dot(res, x)
        
    if out is None:
        return dot(res, args[-1])
    elif out.size == 1: #dot() seems to dislike this
        out[...] = dot(res, args[-1])
        return out
    else:
        return dot(res, args[-1], out=out)


def matmul(out, *args):
    """Multiplies a chain of matrices (2-d ndarrays)
    
    The final output matrix may be provided, or may be set to None. Setting out
    to None causes a new ndarray to be created to hold the result.
    
    All matrices must have dimensions compatible with matrix multiplication.
    
    The underlying matrix multiplication algorithm may not support using the output
    matrix as one of the two arguments (the result may just be wrong in this case).
    As such, matmul raises an exception if the specified out matrix (if any) is
    also one of the arguments in the final multiplication operation.
    
    Parameters
    ----------
    out : ndarray
        A matrix to hold the final result (dimensions must be correct). May be None.
    *args : ndarray
        The chain of matrices to multiply together.

    Returns
    -------
    out : ndarray
        The result.
    """
    if not out is None and (args.count == 2 and out in args or args[-1] is out):
        raise
    return _matmul_dot(out, args)
    
def matmul_init(dtype=float64, order='C'):
    global gemm
    global using_fblas
    
    m = empty((1,1), dtype=dtype, order=order)
    gemm, = la.blas.get_blas_funcs(['gemm'], m) #gets correct gemm (i.e. S, C, D, Z)
    using_fblas = gemm.module_name == 'fblas'

def H(m, out=None):
    """Matrix conjugate transpose (adjoint).
    
    This is just a shortcut for performing this operation on normal ndarrays.
    
    Parameters
    ----------
    m : ndarray
        The input matrix.
    out : ndarray
        A matrix to hold the final result (dimensions must be correct). May be None.
        May also be the same object as m.

    Returns
    -------
    out : ndarray
        The result.    
    """
    if out is None:
        return m.T.conjugate()
    else:
        out = sp.conjugate(m.T, out)
        return out
    
def sqrtmh(A, out=None):
    """Return the matrix square root of a hermitian or symmetric matrix

    Uses scipy.linalg.eigh() to diagonalize the input efficiently.

    Parameters
    ----------
    A : ndarray
        A hermitian or symmetric two-dimensional square array (a matrix)

    Returns
    -------
    sqrt_A : ndarray
        An array of the same shape and type as A containing the matrix square root of A.
        
    Notes
    -----
    The result is also Hermitian.

    """
    ev, V = la.eigh(A) #uses LAPACK ***EVR
    
    ev = sp.sqrt(ev) #we don't require positive (semi) definiteness, so we need the scipy sqrt here
    
    #Carry out multiplication with the diagonal matrix of eigenvalue square roots with H(V)
    B = empty_like(A, order='F') #Since we get a column at a time, fortran ordering is more efficient.
    for i in xrange(V.shape[0]):
        B[:,i] = V[i,:] * ev

    conjugate(B, B) #in-place conjugate

    return matmul(out, V, B)
    
def sqrtmpo(A, out=None):
    """Return the matrix square root of a hermitian or symmetric positive definite matrix

    Uses a Cholesky decomposition, followed by a QR decomposition, and then
    Nwewton iteration to obtain a polar form UH, with H Hermitian p.d. and
    the desired square root, as described in algorithm 6.21 in:
        
    Higham, N. J., "Functions of Matrices, Theory and Computation", SCIAM 2008
    
    NOT YET IMPLEMENTED!

    Parameters
    ----------
    A : ndarray
        A hermitian or symmetric two-dimensional square array (a matrix)

    Returns
    -------
    sqrt_A : ndarray
        An array of the same shape and type as A containing the matrix square root of A.

    """
    R = la.cholesky(A)
    R = la.qr(R, overwrite_a=True, mode='r')
    
    #FIXME: NOTFINISHED
    raise
    
    return 0
    
def invtr(A, overwrite=False, lower=False):
    """Return the inverse of a triangular matrix

    Uses the corresponding LAPACK routing.

    Parameters
    ----------
    A : ndarray
        An upper or lower triangular matrix.
        
    overwrite : bool
        Whether to overwrite the input array (may increase performance).
                
    lower : bool
        Whether the input array is lower-triangular, rather than upper-triangular.

    Returns
    -------
    inv_A : ndarray
        The inverse of A, which is also triangular.    

    """    
    trtri, = la.lapack.get_lapack_funcs(('trtri',), (A,))
    
    inv_A, info = trtri(A, lower=lower, overwrite_c=overwrite)
    
    if info > 0:
        raise LinAlgError("%d-th diagonal element of the matrix is zero" % info)
    if info < 0:
        raise ValueError('illegal value in %d-th argument of internal potri'
                                                                    % -info)       
                                                                    
    return inv_A

def invpo(A, out=None, lower=False):
    """Efficient inversion of positive definite matrices using Cholesky decomposition.
    
    NOT YET WORKING
    """
    t = la.cholesky(A, lower=lower)
    
    print allclose(dot(H(t), t), A)
    #a, lower = la.cho_factor(A, lower=lower) #no.. we need a clean answer, it seems
    
    potri, = la.lapack.get_lapack_funcs(('potri',), (A,))
    
    inv_A, info = potri(t, lower=lower, overwrite_c=1, rowmajor=1) #rowmajor (C-order) is the default...
    
    if info > 0:
        raise LinAlgError("%d-th diagonal element of the Cholesky factor is zero" % info)
    if info < 0:
        raise ValueError('illegal value in %d-th argument of internal potri'
                                                                    % -info)    
    return inv_A
