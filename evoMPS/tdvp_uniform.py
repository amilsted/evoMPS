# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 17:29:27 2011

@author: Ashley Milsted

TODO:
    - Clean up CG code: Create nice interface?
    - Split out excitations stuff?

"""
from __future__ import absolute_import, division, print_function

import copy as cp
import numpy as np
import scipy as sp
import scipy.linalg as la
import scipy.sparse.linalg as las
import scipy.optimize as opti
from . import tdvp_common as tm
from . import matmul as m
from .mps_uniform import EvoMPS_MPS_Uniform, EvoMPSNoConvergence, EvoMPSNormError
from .mps_uniform_pinv import pinv_1mE
from .mps_uniform_excite import Excite_H_Op, Excite_H_Op_tp
import logging
from functools import reduce

log = logging.getLogger(__name__)
        
class EvoMPS_TDVP_Uniform(EvoMPS_MPS_Uniform):
        
    def __init__(self, D, q, ham, ham_sites=None, L=1, dtype=None, do_update=True):
        """Implements the TDVP algorithm for uniform MPS.
        
        Parameters
        ----------
            D : int
                The bond-dimension
            q : int
                The single-site Hilbert space dimension
            ham : callable or ndarray
                Local Hamiltonian term (acting on two or three adjacent sites)
            ham_sites : int
                The number of sites acted on non-trivially by ham. Should be specified for callable ham.
            L : int
                The number of sites in a translation invariant block.
            dtype : numpy dtype = None
                Specifies the array type.
        """

        self.ham = ham
        """The local Hamiltonian term. Can be changed, for example, to perform
           a quench. The number of neighbouring sites acted on must be 
           specified in ham_sites."""
          
        self.ham_sites = ham_sites
        
        self.ham_tp = None

        self.K_solver = las.bicgstab
        """The sparse matrix solver to use for finding K (pseudo-inverse)."""
        
        self.PPinv_use_CUDA = False
        """Whether to use CUDA to implement the pseudo-inverse iterations."""
        
        self.h_expect = sp.NaN
        """The energy density expectation value, available only after calling
           update() or calc_K()."""
        
        super(EvoMPS_TDVP_Uniform, self).__init__(D, q, L=L, dtype=dtype, do_update=do_update)
    
    def _init_arrays(self, D, q, L):
        super(EvoMPS_TDVP_Uniform, self)._init_arrays(D, q, L)
            
        self.set_ham(self.ham)
            
        self.AAA = [None] * L
        
        self.Vsh = [None] * L
            
        self.K_left = [None] * L
        
        self.eta_sq = sp.empty((L), dtype=self.A[0].dtype)
        """The site contributions to the norm squared of the TDVP tangent vector 
           (projection of the exact time
           evolution onto the MPS tangent plane. Only available after calling
           take_step()."""
        self.eta_sq.fill(0)
        
        self.eta = sp.NaN
        """The norm of the TDVP tangent vector (square root of the sum of eta_sq)."""
        
        self.etaBB_sq = sp.empty((L), dtype=self.A[0].dtype)
        """The site contributions to the norm squared of the TDVP evolution
           captured by the two-site tangent plane (but not the one-site tangent plane).
           Available after calling take_step() with dynexp=True."""
        self.etaBB_sq.fill(0)
        
        self.etaBB = sp.NaN
        """The norm of the TDVP evolution captured by the two-site tangent plane.
           Available after calling take_step() with dynexp=True."""

    @classmethod
    def from_tensors(cls, As, ham, ham_sites=None, do_update=True):
        """Creates a sandwich state from a collection of MPS tensors.
        `As` is a list of the tensor in the unit cell.
        """
        q, D = As[0].shape[0:2]
        dtype = As[0].dtype
        mps = EvoMPS_MPS_Uniform.from_tensors(As, do_update=do_update)
        return cls.from_mps(mps, ham, ham_sites)

    @classmethod
    def from_mps(cls, mps, ham, ham_sites=None):
        tdvp = cls(mps.D, mps.q, ham, ham_sites, mps.L, mps.typ, False)
        tdvp.A = cp.deepcopy(mps.A)
        tdvp.AA = cp.deepcopy(mps.AA)
        tdvp.l = cp.deepcopy(mps.l)
        tdvp.r = cp.deepcopy(mps.r)
        tdvp.lL_before_CF = tdvp.l[-1]
        tdvp.lL_before_CF = tdvp.r[-1]
        tdvp.odr = mps.odr
        tdvp.itr_rtol = mps.itr_rtol
        tdvp.itr_atol = mps.itr_atol
        tdvp.zero_tol = mps.zero_tol
        tdvp.pow_itr_max = mps.pow_itr_max
        tdvp.ev_brute = mps.ev_brute
        tdvp.ev_use_arpack = mps.ev_use_arpack
        tdvp.ev_arpack_nev = mps.ev_arpack_nev
        tdvp.ev_arpack_ncv = mps.ev_arpack_ncv
        tdvp.ev_arpack_CUDA = mps.ev_arpack_CUDA
        tdvp.CUDA_batch_maxD = mps.CUDA_batch_maxD
        tdvp.symm_gauge = mps.symm_gauge
        tdvp.sanity_checks = mps.sanity_checks
        tdvp.check_fac = mps.check_fac
        tdvp.eps = mps.eps
        tdvp.userdata = mps.userdata
        return tdvp

    def set_ham(self, ham, ham_sites=None):
        if ham_sites is None:
            try:
                self.ham_sites = len(ham.shape) // 2
            except AttributeError: #TODO: Try to count arguments using inspect module
                self.ham_sites = 2
        else:
            self.ham_sites = ham_sites
            
        self.ham = ham
        
        if not (self.ham_sites == 2 or self.ham_sites == 3):
            raise ValueError("Only 2 or 3 site Hamiltonian terms supported!")
        
        ham_shape = []
        for i in range(self.ham_sites):
            ham_shape.append(self.q)
        C_shape = tuple(ham_shape + [self.D, self.D])        
        
        self.C = []
        self.K = []
        for k in range(self.L):
            self.C.append(np.zeros(C_shape, dtype=self.typ, order=self.odr))
            self.K.append(np.ones_like(self.A[k][0]))
            
        self.K0_init = None
        
    def convert_to_TI_blocked(self, do_update=True):
        if self.L == 1:
            return
            
        L = self.L
        q = self.q
        newq = q**L
        
        #Currently, all supported Hams become nearest-neighbour.
        newham = sp.zeros((newq**2, newq**2), dtype=self.typ)
        oldham = self.ham.reshape((q**self.ham_sites, q**self.ham_sites))
        
        for k in range(L):
            newham += sp.kron(sp.kron(sp.eye(q**k), oldham), sp.eye(q**(2 * L - k - self.ham_sites)))
            
        newham = newham.reshape((newq, newq, newq, newq))
        self.ham = newham
        super(EvoMPS_TDVP_Uniform, self).convert_to_TI_blocked(do_update=do_update)
            
    def set_ham_array_from_function(self, ham_func):
        """Generates a Hamiltonian array from a function.
        
        Given a function ham_func(s, t, u, v) this generates an array
        ham[s, t, u, v] (example for self.ham_sites == 2). 
        Using an array instead of a function can significantly
        speed up parts of the algorithm.
        
        Parameters
        ----------
        ham_func : callable
            Local Hamiltonian term with self.ham_sites * 2 required arguments.
        """
        hv = np.vectorize(ham_func, otypes=[np.complex128])
        
        if self.ham_sites == 2:
            self.ham = np.fromfunction(hv, (self.q, self.q, self.q, self.q))
        else:
            self.ham = np.fromfunction(hv, tuple([self.q] * 6))
    
    def calc_C(self):
        """Generates the C tensor used to calculate the K and ultimately B.
        
        This is called automatically by self.update().
        
        C contains a contraction of the Hamiltonian self.ham with the parameter
        tensors over the local basis indices.
        
        This is prerequisite for calculating the tangent vector parameters B,
        which optimally approximate the exact time evolution.

        Makes use only of the nearest-neighbour Hamiltonian, and of the A's.
        
        C depends on A.
        
        """
        if callable(self.ham):
            ham = np.vectorize(self.ham, otypes=[sp.complex128])
            ham = np.fromfunction(ham, tuple(self.C[0].shape[:-2] * 2))
        else:
            ham = self.ham
        
        if self.ham_tp is None:
            self.calc_AA()
        
        for k in range(self.L):
            if self.ham_sites == 2:
                if not self.ham_tp is None:
                    #self.C[k][:] = tm.calc_C_mat_op_tp(self.ham_tp, self.A[k], self.A[(k + 1) % self.L])
                    self.C[k] = tm.calc_C_tp(self.ham_tp, self.A[k], self.A[(k + 1) % self.L])
                else:
                    self.C[k] = tm.calc_C_mat_op_AA(ham, self.AA[k])
            else:
                self.AAA[k] = tm.calc_AAA_AA(self.AA[k], self.A[(k + 2) % self.L])
                self.C[k][:] = tm.calc_C_3s_mat_op_AAA(ham, self.AAA[k])
    
    def calc_PPinv(self, x, p=0, out=None, left=False, A1=None, A2=None, rL=None, 
                   pseudo=True, brute_check=False, sc_data='', solver=None,
                   tol=None):
        """Uses an iterative method to calculate the result of applying 
        the (pseudo) inverse of (1 - exp(1.j * p) * E) to a vector |x>.
        
        Parameters
        ----------
        x : ndarray
            The matrix representation of the vector |x>.
        p : float
            Momentum in units of inverse lattice spacing.
        out : ndarray
            Appropriately-sized output matrix.
        left : bool
            Whether to act left on |x> (instead of right).
        A1 : ndarray
            Ket parameter tensor.
        A2 : ndarray
            Bra parameter tensor.
        r : ndarray
            Right eigenvector of E corresponding to the largest eigenvalue.
        pseudo : bool
            Whether to calculate the pseudo inverse (or just the inverse).
        brute_check : bool
            Whether to check the answer using dense methods (scales as D**6!).
            
        Returns
        -------
        out : ndarray
            The result of applying the inverse operator, in matrix form.
        """
        if A1 is None:
            A1 = self.A
            
        if A2 is None:
            A2 = self.A
            
        if rL is None:
            rL = self.r[-1]
            
        if tol is None:
            tol = self.itr_rtol
        
        out = pinv_1mE(x, A1, A2, self.l[-1], rL, p=p, left=left, pseudo=pseudo, 
                       out=out, tol=tol, solver=solver, brute_check=brute_check,
                       sanity_checks=self.sanity_checks, sc_data=sc_data,
                       use_CUDA=self.PPinv_use_CUDA,
                       CUDA_use_batch=(self.D <= self.CUDA_batch_maxD))

        return out
        
    def calc_K(self):
        """Generates the K matrix used to calculate B.
        
        This also updates the energy-density expectation value self.h_expect.
        
        This is called automatically by self.update().
        
        K contains the (non-trivial) action of the Hamiltonian on the right 
        half of the infinite chain.
        
        It directly depends on A, r, and C.
        """
        L = self.L
        
        if self.ham_sites == 2:
            if not self.ham_tp is None:
                Hr = tm.eps_r_op_2s_C12_tp(self.r[1 % L], self.C[0], self.A[0], self.A[1 % L])
                for k in range(1, L):
                    Hrk = tm.eps_r_op_2s_C12_tp(self.r[(k + 1) % L], self.C[k], self.A[k], self.A[(k + 1) % L])
                    for j in range(k - 1, -1, -1):
                        Hrk = tm.eps_r_noop(Hrk, self.A[j], self.A[j])
                    Hr += Hrk
            else:
                Hr = tm.eps_r_op_2s_C12_AA34(self.r[1 % L], self.C[0], self.AA[0])
                for k in range(1, L):
                    Hrk = tm.eps_r_op_2s_C12_AA34(self.r[(k + 1) % L], self.C[k], self.AA[k])
                    for j in range(k - 1, -1, -1):
                        Hrk = tm.eps_r_noop(Hrk, self.A[j], self.A[j])
                    Hr += Hrk
        else:
            Hr = tm.eps_r_op_3s_C123_AAA456(self.r[2 % L], self.C[0], self.AAA[0])
            for k in range(1, L):
                Hrk = tm.eps_r_op_3s_C123_AAA456(self.r[(k + 2) % L], self.C[k], self.AAA[k])
                for j in range(k - 1, -1, -1):
                    Hrk = tm.eps_r_noop(Hrk, self.A[j], self.A[j])
                Hr += Hrk
        
        self.h_expect = m.adot(self.l[-1], Hr)
        
        QHr = Hr - self.r[-1] * self.h_expect
        
        self.h_expect /= L
        
        if not self.K0_init is None:
            self.K[0] = self.K0_init

        self.calc_PPinv(QHr, out=self.K[0], solver=self.K_solver)
        
        self.K0_init = None
        
        if self.ham_sites == 2:
            if not self.ham_tp is None:
                for k in sp.arange(L - 1, 0, -1) % L:
                    self.K[k], hk = tm.calc_K_tp(self.K[(k + 1) % L], self.l[k - 1], self.r[(k + 1) % L],
                                                 self.A[k], self.A[(k + 1) % L], self.C[k])
                    self.K[k] -= self.r[(k - 1) % L] * hk
            else:
                for k in sp.arange(L - 1, 0, -1) % L:
                    self.K[k], hk = tm.calc_K(self.K[(k + 1) % L], self.C[k], self.l[k - 1], self.r[(k + 1) % L],
                                              self.A[k], self.AA[k])
                    self.K[k] -= self.r[(k - 1) % L] * hk
        else:
            for k in sp.arange(L - 1, 0, -1) % L:
                self.K[k], hk = tm.calc_K_3s(self.K[(k + 1) % L], self.C[k], self.l[k - 1], self.r[(k + 2) % L],
                                             self.A[k], self.AAA[k])
                self.K[k] -= self.r[(k - 1) % L] * hk
        
#        if self.sanity_checks:
#            Ex = tm.eps_r_noop(self.K[k], self.A, self.A)
#            QEQ = Ex - self.r * m.adot(self.l, self.K)
#            res = self.K - QEQ
#            if not np.allclose(res, QHr):
#                log.warning("Sanity check failed: Bad K!")
#                log.warning("Off by: %s", la.norm(res - QHr))
        
    def calc_K_l(self):
        """Generates the left K matrix.
        
        See self.calc_K().
        
        K contains the (non-trivial) action of the Hamiltonian on the left 
        half of the infinite chain.
        
        It directly depends on A, l, and C.
        
        This calculates the "bra-vector" K_l ~ <K_l| (and K_l.conj().T ~ |K_l>)
        so that <K_l|r> = trace(K_l.dot(r))
        
        Returns
        -------
        K_lefts : list of ndarrays
            The left K matrices.
        h : complex
            The energy-density expectation value.
        """
        L = self.L
        if self.ham_sites == 2:
            if not self.ham_tp is None:
                lH = tm.eps_l_op_2s_C34_tp(self.l[(L - 3) % L], self.A[(L - 2) % L], self.A[(L - 1) % L], self.C[(L - 2) % L])
                for k in sp.arange(-1, L - 2) % L:
                    lHk = tm.eps_l_op_2s_C34_tp(self.l[(k - 1) % L], self.A[k], self.A[(k + 1) % L], self.C[k])
                    for j in range((k + 2) % L, L):
                        lHk = tm.eps_l_noop(lHk, self.A[j], self.A[j])
                    lH += lHk
            else:
                lH = tm.eps_l_op_2s_AA12_C34(self.l[(L - 3) % L], self.AA[(L - 2) % L], self.C[(L - 2) % L])
                for k in sp.arange(-1, L - 2) % L:
                    lHk = tm.eps_l_op_2s_AA12_C34(self.l[(k - 1) % L], self.AA[k], self.C[k])
                    for j in range((k + 2) % L, L):
                        lHk = tm.eps_l_noop(lHk, self.A[j], self.A[j])
                    lH += lHk
        else:
            lH = tm.eps_l_op_3s_AAA123_C456(self.l[(L - 4) % L], self.AAA[(L - 3) % L], self.C[(L - 3) % L])
            for k in sp.arange(-2, L - 3) % L:
                lHk = tm.eps_l_op_3s_AAA123_C456(self.l[(k - 1) % L], self.AAA[k], self.C[k])
                for j in range((k + 3) % L, L):
                    lHk = tm.eps_l_noop(lHk, self.A[j], self.A[j])
                lH += lHk
        
        h = m.adot_noconj(lH, self.r[-1]) #=tr(lH r)
        
        lHQ = lH - self.l[-1] * h
        
        h /= L
        
        #Since A1=A2 and p=0, we get the right result without turning lHQ into a ket.
        #This is the same as...
        #self.K_left = (self.calc_PPinv(lHQ.conj().T, left=True, out=self.K_left)).conj().T
        self.K_left[-1] = self.calc_PPinv(lHQ, left=True, out=self.K_left[-1], solver=self.K_solver)
                
        if self.ham_sites == 2:
            if not self.ham_tp is None:
                for k in sp.arange(0, L - 1):
                    self.K_left[k], hk = tm.calc_K_l_tp(self.K[(k - 1) % L],  
                                                 self.l[(k - 2) % L], self.r[k],
                                                 self.A[(k - 1) % L], self.A[k],
                                                 self.C[(k - 1) % L])
                    self.K_left[k] -= self.l[k] * hk
            else:
                for k in sp.arange(0, L - 1):
                    self.K_left[k], hk = tm.calc_K_l(self.K[(k - 1) % L], self.C[(k - 1) % L], 
                                                 self.l[(k - 2) % L], self.r[k],
                                                 self.A[k], self.AA[(k - 1) % L])
                    self.K_left[k] -= self.l[k] * hk
        else:
            for k in sp.arange(0, L - 1):
                self.K_left[k], hk = tm.calc_K_3s_l(self.K[(k - 1) % L], self.C[(k - 2) % L], 
                                                self.l[(k - 3) % L], self.r[k],
                                                self.A[k],
                                                self.AAA[(k - 2) % L])
                self.K_left[k] -= self.l[k] * hk
        
#        if self.sanity_checks:
#            xE = tm.eps_l_noop(self.K_left, self.A, self.A)
#            QEQ = xE - self.l * m.adot(self.r, self.K_left)
#            res = self.K_left - QEQ
#            if not np.allclose(res, lHQ):
#                log.warning("Sanity check failed: Bad K_left!")
#                log.warning("Off by: %s", la.norm(res - lHQ))
        
        return self.K_left, h
        
    def get_B_from_x(self, xk, Vshk, lkm1_sqrt_i, rk_sqrt_i, out=None):
        """Calculates a gauge-fixing B-tensor for a site with block offset k 
        given parameters x.
        
        Parameters
        ----------
        xk : ndarray
            The parameter matrix.
        Vshk : ndarray
            Parametrization tensor for site offset k.
        lkm1_sqrt_i : ndarray
            The matrix self.l[k - 1] to the power of -1/2.
        rk_sqrt_i : ndarray
            The matrix self.r[k] to the power of -1/2.
        out : ndarray
            Output tensor of appropriate shape.
        """
        if out is None:
            out = np.zeros_like(self.A[0])
            
        for s in range(self.q):
            out[s] = lkm1_sqrt_i.dot(xk).dot(rk_sqrt_i.dot(Vshk[s]).conj().T)
            
        return out
        
    def calc_l_r_roots(self):
        """Calculates the (inverse) square roots of self.l and self.r.
        """
        self.l_sqrt = [None] * self.L
        self.l_sqrt_i = [None] * self.L
        self.r_sqrt = [None] * self.L
        self.r_sqrt_i = [None] * self.L
        for k in range(self.L):
            self.l_sqrt[k], self.l_sqrt_i[k], self.r_sqrt[k], self.r_sqrt_i[k] = tm.calc_l_r_roots(self.l[k], self.r[k], zero_tol=self.zero_tol, sanity_checks=self.sanity_checks)
        
    def calc_B(self, set_eta=True, return_x=False):
        """Calculates a gauge-fixing tangent-vector parameter tensor capturing the projected infinitesimal time evolution of the state.
        
        A TDVP time step is defined as: A -= dtau * B
        where dtau is an infinitesimal imaginary time step.        
        
        Parameters
        ----------
        set_eta : bool
            Whether to set self.eta to the norm of the tangent vector.
        """
        self.calc_l_r_roots()
        L = self.L
        
        B = []
        Vsh = []
        x = []
        for k in range(L):
            Vshk = tm.calc_Vsh(self.A[k], self.r_sqrt[k], sanity_checks=self.sanity_checks)
            Vsh.append(Vshk)
            
            if self.ham_sites == 2:
                if not self.ham_tp is None:
                    xk = tm.calc_x_tp(self.K[(k + 1) % L], self.C[k], self.C[(k-1)%L], 
                                  self.r[(k+1)%L], self.l[(k-2)%L], self.A[(k-1)%L], 
                                  self.A[k], self.A[(k+1)%L], 
                                  self.l_sqrt[(k-1)%L], self.l_sqrt_i[(k-1)%L],
                                  self.r_sqrt[k], self.r_sqrt_i[k], Vshk)
                else:
                    xk = tm.calc_x(self.K[(k + 1) % L], self.C[k], self.C[(k-1)%L], 
                                  self.r[(k+1)%L], self.l[(k-2)%L], self.A[(k-1)%L], 
                                  self.A[k], self.A[(k+1)%L], 
                                  self.l_sqrt[(k-1)%L], self.l_sqrt_i[(k-1)%L],
                                  self.r_sqrt[k], self.r_sqrt_i[k], Vshk)
            else:
                xk = tm.calc_x_3s(self.K[(k + 1) % L], self.C[k], self.C[(k-1)%L], 
                                 self.C[(k-2)%L], self.r[(k+1)%L], self.r[(k+2)%L], 
                                 self.l[(k-2)%L], self.l[(k-3)%L], 
                                 self.AA[(k-2)%L], self.A[(k-1)%L], self.A[k], 
                                 self.A[(k+1)%L], self.AA[(k+1)%L], 
                                 self.l_sqrt[(k-1)%L], self.l_sqrt_i[(k-1)%L],
                                 self.r_sqrt[k], self.r_sqrt_i[k], Vshk)
            
            if set_eta:
                self.eta_sq[k] = m.adot(xk, xk)
                
            x.append(xk)
            B.append(self.get_B_from_x(xk, Vshk, self.l_sqrt_i[(k - 1) % L], self.r_sqrt_i[k]))
        
            if self.sanity_checks:
                #Test gauge-fixing:
                tst = tm.eps_r_noop(self.r[k], B[k], self.A[k])
                if not np.allclose(tst, 0):
                    log.warning("Sanity check failed: Gauge-fixing violation! %s", la.norm(tst))
            
        self.Vsh = Vsh
        self.eta = sp.sqrt(self.eta_sq.sum())
            
        if return_x:
            return B, x
        else:
            return B
        
    def calc_BB_Y_2s(self, Vlh):
        L = self.L
        Y = sp.empty((L), dtype=sp.ndarray)
        if self.ham_sites == 2:
            if not self.ham_tp is None:
                for k in range(L):
                    Y[k], self.etaBB_sq[k] = tm.calc_BB_Y_2s_tp(self.C[k], Vlh[k], self.Vsh[(k + 1) % L],
                                                       self.l_sqrt[k - 1], self.r_sqrt[(k + 1) % L])
            else:
                for k in range(L):
                    Y[k], self.etaBB_sq[k] = tm.calc_BB_Y_2s(self.C[k], Vlh[k], self.Vsh[(k + 1) % L],
                                                       self.l_sqrt[k - 1], self.r_sqrt[(k + 1) % L])
        else:
            for k in range(L):
                Y[k], self.etaBB_sq[k] = tm.calc_BB_Y_2s_ham_3s(self.A[k - 1], self.A[(k + 2) % L], 
                                       self.C[k], self.C[k - 1], Vlh[k], self.Vsh[(k + 1) % L],
                                       self.l[(k - 2) % L], self.r[(k + 2) % L],
                                       self.l_sqrt[k - 1], self.l_sqrt_i[k - 1], 
                                       self.r_sqrt[(k + 1) % L], self.r_sqrt_i[(k + 1) % L])
        
        self.etaBB = sp.sqrt(self.etaBB_sq.sum())
        
        return Y, self.etaBB_sq
        
    def calc_B_2s(self, dD_max=16, sv_tol=1E-14):
        Vrh = self.Vsh
        Vlh = []
        L = self.L
        for k in range(L):
            Vlh.append(tm.calc_Vsh_l(self.A[k], self.l_sqrt[k - 1], sanity_checks=self.sanity_checks))
        
        Y, etaBB_sq = self.calc_BB_Y_2s(Vlh)
        
        BB1 = [None] * L
        BB2 = [None] * L
        for k in range(L):
            BB1[k], BB2[(k + 1) % L], dD = tm.calc_BB_2s(Y[k], Vlh[k], Vrh[(k + 1) % L], 
                                              self.l_sqrt_i[k - 1], self.r_sqrt_i[(k + 1) % L],
                                              dD_max=dD_max, sv_tol=0) #FIXME: Make D variable...
        
        return BB1, BB2, etaBB_sq
        
    def update(self, restore_CF=True, auto_truncate=False, restore_CF_after_trunc=True):
        """Updates secondary quantities to reflect the state parameters self.A.
        
        Must be used after taking a step or otherwise changing the 
        parameters self.A before calculating
        physical quantities or taking the next step.
        
        Also (optionally) restores canonical form by calling self.restore_CF().
        
        Parameters
        ----------
        restore_CF : bool (True)
            Whether to restore canonical form.
        auto_truncate : bool (True)
            Whether to automatically truncate the bond-dimension if
            rank-deficiency is detected. Requires restore_CF.
        restore_CF_after_trunc : bool (True)
            Whether to restore_CF after truncation.
        """
        super(EvoMPS_TDVP_Uniform, self).update(restore_CF=restore_CF,
                                                auto_truncate=auto_truncate,
                                                restore_CF_after_trunc=restore_CF_after_trunc)
        self.calc_C()
        self.calc_K()

    def phase_align(self, other):
        """Adjusts the parameter tensor A by a phase-factor to align it with another state.

        This ensures that the largest eigenvalue of the overlap transfer operator
        is real.

        Parameters
        ----------
        other : EvoMPS_MPS_Uniform
            MPS with which to calculate the per-site fidelity.

        Returns
        -------
        phi : complex
            The phase difference (phase of the eigenvalue).
        """
        phi = super(EvoMPS_TDVP_Uniform, self).phase_align(other)
        self.calc_C()  # needed, since AA and C also change phases!
        return phi

    def take_step(self, dtau, B=None, dynexp=False, D_max=128, dD_max=16, 
                  sv_tol=1E-14, BB=None):
        """Performs a complete forward-Euler step of imaginary time dtau.
        
        The operation is A -= dtau * B with B from self.calc_B() by default.
        
        If dtau is itself imaginary, real-time evolution results.
        
        Parameters
        ----------
        dtau : complex
            The (imaginary or real) amount of imaginary time (tau) to step.
        B : ndarray
            A custom parameter-space tangent vector to step along.
        """
        if B is None:
            B = self.calc_B()
        
        if dynexp and self.D < D_max:
            if BB is None:
                BB = self.calc_B_2s(dD_max=min(dD_max, D_max - self.D), sv_tol=sv_tol)
            if not BB is None:
                BB1, BB2, etaBB_sq = BB
                oldD = self.D
                dD = BB1[0].shape[2]
                self.expand_D(self.D + dD, refac=0, imfac=0) #FIXME: Currently expands all D
                #print BB1.shape, la.norm(BB1.ravel()), BB2.shape, la.norm(BB2.ravel())
                for k in range(self.L):
                    self.A[k][:, :oldD, :oldD] += -dtau * B[k]
                    self.A[k][:, :oldD, oldD:] = -1.j * sp.sqrt(dtau) * BB1[k]
                    self.A[k][:, oldD:, :oldD] = -1.j * sp.sqrt(dtau) * BB2[k]
                    self.A[k][:, oldD:, oldD:].fill(0)
                log.info("Dynamically expanded! New D: %d", self.D)
            else:
                for k in range(self.L):
                    self.A[k] += -dtau * B[k]
        else:
            for k in range(self.L):
                self.A[k] += -dtau * B[k]
            
    def take_step_RK4(self, dtau, B_i=None, dynexp=False, D_max=128, dD_max=16, 
                      sv_tol=1E-14, BB=None):
        """Take a step using the fourth-order explicit Runge-Kutta method.
        
        This requires more memory than a simple forward Euler step. 
        It is, however, far more accurate with a per-step error of
        order dtau**5.
        
        Parameters
        ----------
        dtau : complex
            The (imaginary or real) amount of imaginary time (tau) to step.
        B_i : ndarray
            B calculated using self.calc_B() (if known, to avoid calculating it again).
        """
        def update():
            self.calc_lr()
            self.calc_AA()
            self.calc_C()
            self.calc_K()            

        A0 = cp.deepcopy(self.A)
            
        if not B_i is None:
            B = B_i
        else:
            B = self.calc_B() #k1
        B_fin = B
        self.take_step(dtau / 2., B=B, dynexp=dynexp, D_max=D_max, dD_max=dD_max, sv_tol=sv_tol, BB=BB)

        dD = self.A[0].shape[1] - A0[0].shape[1]
        if dD > 0:
            [A0k.resize(self.A[0].shape) for A0k in A0] #pads with zeros
            #B part doesn't work due to different scaling with dtau :(
            assert False
        
        update()
        
        B = self.calc_B(set_eta=False) #k2                
        for k in range(self.L):
            self.A[k] = A0[k] - dtau/2 * B[k]
            B_fin[k] += 2 * B[k]
            
        update()
            
        B = self.calc_B(set_eta=False) #k3                
        for k in range(self.L):
            self.A[k] = A0[k] - dtau * B[k]
            B_fin += 2 * B[k]

        update()
        
        B = self.calc_B(set_eta=False) #k4
        for k in range(self.L):
            B_fin[k] += B[k]
            self.A[k] = A0[k] - dtau/6 * B_fin[k]
             

    def find_ground(use_CG=True, max_steps=None, tol=1E-6):
        pass

    def evolve_state():
        pass
            
    def _prepare_excite_op_top_triv(self, p, pinv_tol=1E-12, pinv_solver=None):
        if callable(self.ham):
            self.set_ham_array_from_function(self.ham)

        self.calc_K_l()
        self.calc_l_r_roots()
        self.Vsh[0] = tm.calc_Vsh(self.A[0], self.r_sqrt[0], sanity_checks=self.sanity_checks)
        
        if not self.ham_tp is None:
            op = Excite_H_Op_tp(self, self, p, pinv_tol=pinv_tol,
                             sanity_checks=self.sanity_checks)
        else:
            op = Excite_H_Op(self, self, p, pinv_tol=pinv_tol,
                             sanity_checks=self.sanity_checks)
        op.pinv_CUDA = self.PPinv_use_CUDA
        op.pinv_solver = pinv_solver

        return op        
    
    def excite_top_triv(self, p, nev=6, tol=0, max_itr=None, v0=None, ncv=None,
                        sigma=None, which='SM', return_eigenvectors=False,
                        max_retries=3, pinv_tol=1E-12, pinv_solver=None):
        """Calculates approximate eigenvectors and eigenvalues of the Hamiltonian
        using tangent vectors of the current state as ansatz states.
        
        This is best used with an approximate ground state to find approximate
        excitation energies.
        
        This uses topologically trivial ansatz states. Given a ground state
        degeneracy, topologically non-trivial low-lying eigenstates 
        (such as kinks or solitons) may also exist. See self.excite_top_nontriv().
        
        Many of the parameters are passed on to scipy.sparse.linalg.eigsh().
        
        Parameters
        ----------
        p : float
            Momentum in units of inverse lattice spacing.
        nev : int
            Number of eigenvalues to calculate.
        tol : float
            Tolerance (defaults to machine precision).
        max_itr : int
            Maximum number of iterations.
        v0 : ndarray
            Starting vector.
        ncv : int
            Number of Arnoldi vectors to store.
        sigma : float
            Eigenvalue shift to use.
        which : string
            Which eigenvalues to find ('SM' means the nev smallest).
        return_eigenvectors : bool
            Whether to return eigenvectors as well as eigenvalues.
        max_retries : int
            Maximum number of retries (with growing ncv), in case of no convergence.
        pinv_tol : float
            Tolerance for pseudo-inverse operations.
            
        Returns
        -------
        ev : ndarray
            List of eigenvalues.
        eV : ndarray
            Matrix of eigenvectors (if return_eigenvectors == True).
        """
        op = self._prepare_excite_op_top_triv(p, pinv_tol=pinv_tol,
                                              pinv_solver=pinv_solver)
                                              
        if ncv is None:
            ncv = max(20, 2 * nev + 1)
        
        for i in range(max_retries):
            try:
                res = las.eigsh(op, which=which, k=nev, v0=v0, ncv=ncv,
                                 return_eigenvectors=return_eigenvectors, 
                                 maxiter=max_itr, tol=tol, sigma=sigma)
            except las.ArpackNoConvergence:
                log.warning("excite_top_triv: Retry %u!", i)
                v0 = None
                ncv = max(20, nev * (3 + i))
                if i == max_retries - 1:
                    log.error("excite_top_triv: Failed to converge!")
                    raise EvoMPSNoConvergence('excite_top_triv')
                          
        return res
    
    def excite_top_triv_brute(self, p, return_eigenvectors=False, pinv_tol=1E-12):
        op = self._prepare_excite_op_top_triv(p, pinv_tol=pinv_tol)
        
        x = np.empty(((self.q - 1)*self.D**2), dtype=self.typ)
        
        H = np.zeros((x.shape[0], x.shape[0]), dtype=self.typ)
        
        for i in range(x.shape[0]):
            x.fill(0)
            x[i] = 1
            H[:, i] = op.matvec(x)

        if not np.allclose(H, H.conj().T):
            log.warning("Warning! H is not Hermitian! %s", la.norm(H - H.conj().T))
         
        return la.eigh(H, eigvals_only=not return_eigenvectors)

    def _prepare_excite_op_top_nontriv(self, donor, p, pinv_tol=1E-12,
                                       pinv_solver=None, phase_align=True,
                                       force_pseudo=False):
        if callable(self.ham):
            self.set_ham_array_from_function(self.ham)
        if callable(donor.ham):
            donor.set_ham_array_from_function(donor.ham)

        if phase_align:
            self.phase_align(donor)
            self.calc_C()  # needed, since phases on AA and C also change!

        self.calc_K_l()
        self.calc_l_r_roots()
        donor.calc_l_r_roots()
        donor.Vsh[0] = tm.calc_Vsh(donor.A[0], donor.r_sqrt[0], sanity_checks=self.sanity_checks)
        
        if not self.ham_tp is None:
            op = Excite_H_Op_tp(self, donor, p, pinv_tol=pinv_tol,
                             sanity_checks=self.sanity_checks,
                             force_pseudo=force_pseudo)
        else:
            op = Excite_H_Op(self, donor, p, pinv_tol=pinv_tol,
                             sanity_checks=self.sanity_checks,
                             force_pseudo=force_pseudo)
        op.pinv_CUDA = self.PPinv_use_CUDA
        op.pinv_solver = pinv_solver

        return op 

    def excite_top_nontriv(self, donor, p, nev=6, tol=0, max_itr=None, v0=None,
                           which='SM', return_eigenvectors=False, sigma=None,
                           ncv=None, max_retries=3, pinv_tol=1E-12,
                           pinv_solver=None, phase_align=True,
                           force_pseudo=False):
        """Calculates approximate eigenvectors and eigenvalues of the Hamiltonian
        using tangent vectors of the current state as ansatz states.
        
        This is best used with an approximate ground state to find approximate
        excitation energies.
        
        This uses topologically nontrivial ansatz states and requires two
        ground of the possible states. One will be used on one infinite half
        of the chain, the other on the other half. The local disturbance
        represents the interface (a kink, for example) between the two.
        
        The second ground state can be generated by applying a symmetry
        operation to a copy of an existing ground state.
        
        Many of the parameters are passed on to scipy.sparse.linalg.eigsh().
        
        Parameters
        ----------
        donor : EvoMPS_TDVP_Uniform
            The second ground state.
        p : float
            Momentum in units of inverse lattice spacing.
        nev : int
            Number of eigenvalues to calculate.
        tol : float
            Tolerance (defaults to machine precision).
        max_itr : int
            Maximum number of iterations.
        v0 : ndarray
            Starting vector.
        ncv : int
            Number of Arnoldi vectors to store.
        sigma : float
            Eigenvalue shift to use.
        which : string
            Which eigenvalues to find ('SM' means the nev smallest).
        return_eigenvectors : bool
            Whether to return eigenvectors as well as eigenvalues.
        max_retries : int
            Maximum number of retries (with growing ncv), in case of no convergence.
        pinv_tol : float
            Tolerance for pseudo-inverse operations.
        phase_align : bool
            If False, do not adjust the global phase of the "donor" state. This
            may lead to an effective non-physical momentum offset.
            
        Returns
        -------
        ev : ndarray
            List of eigenvalues.
        eV : ndarray
            Matrix of eigenvectors (if return_eigenvectors == True).
        """
        op = self._prepare_excite_op_top_nontriv(donor, p, pinv_tol=pinv_tol,
                                                 pinv_solver=pinv_solver,
                                                 phase_align=phase_align,
                                                 force_pseudo=force_pseudo)
        
        if ncv is None:
            ncv = max(20, 2 * nev + 1)
        
        for i in range(max_retries):
            try:
                res = las.eigsh(op, sigma=sigma, which=which, k=nev, v0=v0,
                                    return_eigenvectors=return_eigenvectors, 
                                    maxiter=max_itr, tol=tol, ncv=ncv)
            except las.ArpackNoConvergence:
                log.warning("excite_top_nontriv: Retry %u!", i)
                v0 = None
                ncv = max(20, nev * (3 + i))
                if i == max_retries - 1:
                    log.error("excite_top_nontriv: Failed to converge!")
                    raise EvoMPSNoConvergence('excite_top_nontriv')
                          
        return res
        
    def excite_top_nontriv_brute(self, donor, p, return_eigenvectors=False,
                                 pinv_tol=1E-12, phase_align=True,
                                 force_pseudo=False):
        op = self._prepare_excite_op_top_nontriv(donor, p,
                                                 pinv_tol=pinv_tol,
                                                 phase_align=phase_align,
                                                 force_pseudo=force_pseudo)
        
        x = np.empty(((self.q - 1)*self.D**2), dtype=self.typ)
        
        H = np.zeros((x.shape[0], x.shape[0]), dtype=self.typ)
        
        for i in range(x.shape[0]):
            x.fill(0)
            x[i] = 1
            H[:, i] = op.matvec(x)

        if not np.allclose(H, H.conj().T):
            log.warning("Warning! H is not Hermitian! %s", la.norm(H - H.conj().T))
         
        return la.eigh(H, eigvals_only=not return_eigenvectors)

    def _B_overlap_calc_BR(self, B, p_, conj=False, con_tol=None, GF_tol=1E-15,
                           A2=None, l=None, r=None,):
        L = self.L
        BRr = [None] * L
        BR = [None] * L
        
        if A2 is None:
            A2 = self.A
        if l is None:
            l = self.l
        if r is None:
            r = self.r
        
        M = len(B[0].shape) - 2
        Ae = list(self.A) * int(sp.ceil((L - 1. + M) / L))
        A2e = list(A2) * int(sp.ceil((L - 1. + M) / L))
        
        #print "M=", M
        #print len(A)
        
        As = [None] * L
        A2s = [None] * L
        for k in range(L):
            As[k] = Ae[k:k + M]
            A2s[k] = A2e[k:k + M]
        Bs = [None] * L
        for k in range(L):
            Bs[k] = [B[k]]
        
        if conj:
            X1 = As
            X2 = Bs
        else:
            X1 = Bs
            X2 = A2s

        BRblock = 0
        for k in range(L - 1, -1, -1):
            BRr[k] = tm.eps_r_noop_multi(r[(k + M - 1) % L], X1[k], X2[k])
            #BRr[k] = tm.eps_r_noop(self.r[k], X1[k][0], X2[k][0])
            if k < L - 1:
                BRblock = sp.exp(1.j * p_) * tm.eps_r_noop(BRblock, Ae[k], A2e[k])
            BRblock += BRr[k]
                
        #print "BRblock", la.norm(BRblock.ravel())
                
        if sp.allclose(BRblock.ravel(), 0, rtol=GF_tol): #Skip the far connected contributions if we are approximately gauge-fixing
            BRblock_fid = 0
        else:
            BRblock_fid = m.adot(l[-1], BRblock)
            QBRblock = BRblock - r[-1] * BRblock_fid
            BR[0] = QBRblock.copy()
            BR[0] = self.calc_PPinv(QBRblock, p=(L * p_), solver=las.lgmres, 
                                    tol=con_tol, out=BR[0])
                                                         
            for k in range(L - 1, 0, -1):
                BR[k] = BRr[k] + sp.exp(1.j * p_) * tm.eps_r_noop(BR[(k + 1) % L], Ae[k], A2e[k])
                #BR[k] -= self.r[k - 1] *  m.adot(self.l[k - 1], BR[k]) NO
                
            #BR0_ = BRr[0] + sp.exp(1.j * p_) * tm.eps_r_noop(BR[1 % L], self.A[0], self.A[0])
            #BR0_ -= self.r[-1] *  m.adot(self.l[-1], BR0_)
            #print "BR0_check", la.norm(BR0_ - BR[0])
                    
            #print "BRblock_fid", BRblock_fid
            #print "BR", la.norm(BR[0].ravel())
            
        return BR, BRblock_fid

    def _B_to_B_GF(self, B, p=0, con_tol=None, GF_tol=1E-15):
        L = self.L
        q = self.q
        D = self.D
        M = len(B[0].shape) - 2
        A = list(self.A) * int(sp.ceil((L - 1. + 2. * M - 1.) / L))
        
        BR, BRblock_fid = self._B_overlap_calc_BR(B, p, conj=False, con_tol=con_tol, GF_tol=GF_tol)
        
        x = [None] * L
        newB = [None] * L
        for k in range(L):
            #VR = sp.transpose(self.Vsh[(k + M - 1) % L].conj(), axes=(0, 2, 1))
            r_si = self.r_sqrt_i[(k + M - 1) % L]
            VRh = self.Vsh[(k + M - 1) % L]
            VRr_si = sp.empty((VRh.shape[0], VRh.shape[2], VRh.shape[1]), dtype=VRh.dtype)
            for s in range(len(VRr_si)):
                VRr_si[s] = r_si.dot(VRh[s]).conj().T
            #VRr_si = sp.array([r_si.dot(VRhs).conj().T for VRhs in self.Vsh[(k + M - 1) % L]])
            
            x[k] = [0] * q**(M - 1)
    
            if not BR[(k + M) % L] is None:
                BR_ = tm.eps_r_noop(BR[(k + M) % L], self.A[(k + M - 1) % L], VRr_si)
            else:
                BR_ = 0
                
            #Loops over overlapping terms (right overlap)
            for j in range(0, M):
                Bkj = B[(k + j) % L].reshape((q**(M - 1 - j), q**(j + 1), D, D))
                
                bra = [VRr_si] + A[k + M:k + M + j]
                for s in range(q**(M - 1)):
                    ket = [ Bkj[s % q**(M - 1 - j)] ]
                    
                    if j == 0:
                        As = m.eyemat(D, dtype=self.typ)
                    else:
                        #TODO: Use pre-calculated A products here
                        As = [A[k + l][(s // q**(M - 1 - l - 1)) % q] for l in range(j)]
                        As = reduce(sp.dot, As)
                    
                    Rs = sp.exp(j * p * 1.j) \
                          * tm.eps_r_noop_multi(self.r[(k + M - 1 + j) % L], ket, bra)
                          
                    #convenient place to add connected, but not overlapping contribution
                    if j == M - 1:
                        Rs += sp.exp(M * p * 1.j) * BR_
                        
                    x[k][s] += self.l_sqrt[k - 1].dot(As.dot(Rs))
                                        
            x[k] = sp.array(x[k])

            newB[k] = sp.zeros((q**M, D, D), dtype=self.typ)
            for s in range(q**(M)):
                newB[k][s] = self.l_sqrt_i[k - 1].dot(x[k][s // q]).dot(VRr_si[s % q])
            newB[k] = newB[k].reshape(tuple([q] * M + [D, D]))
            
        return newB, x
    
    def _B_random(self, M=1):
        sha = tuple([self.q] * M + [self.D, self.D])
        B = [sp.rand(*sha) + 1.j * sp.rand(*sha) for k in range(self.L)]
        return B
    
    def _B_to_B_GF_test(self, ps=[0], M=1):
        self.calc_B()
        
        B1GF, B1GFx, B1n = self._B_random_GF(M=M)
        B2GF, B2GFx, B2n = self._B_random_GF(M=M)
        
        ol = sp.sum([m.adot(x1k, x2k) for x1k, x2k in zip(B2GFx, B1GFx)])
        ol2 = self._B_overlap(B1GF, B2GF, B1_is_GF=True, B2_is_GF=True)
        print("GF overlap:", ol, ol2)
        
        y1 = self._B_random_GT_par(M=M)
        y2 = self._B_random_GT_par(M=M)
        
        for p in ps:
            B1 = self._B_rand_GT(B1GF, p=p, x=y1)
            B2 = self._B_rand_GT(B2GF, p=p, x=y2)
            
            print("p=", p)
            print("--------------")
            ol_ = self._B_overlap(B1, B2, p=p)
            print("B1,B2", ol_, abs(ol - ol_))
            B1GF_, B1GF_x = self._B_to_B_GF(B1, p=p)
            B2GF_, B2GF_x = self._B_to_B_GF(B2, p=p)
            ol_ = self._B_overlap(B1GF_, B2GF_, B1_is_GF=True, B2_is_GF=True)
            print("B1GF_,B2GF_", ol_, abs(ol - ol_))
            ol_ = self._B_overlap(B1GF_, B2GF, p=p, B1_is_GF=True, B2_is_GF=True)
            print("B1GF_,B2GF", ol_, abs(ol - ol_))
            ol_ = self._B_overlap(B1GF, B2GF_, p=p, B1_is_GF=True, B2_is_GF=True)
            print("B1GF,B2GF_", ol_, abs(ol - ol_))
            
            print() 
            
    def _B_overlap_onsite(self, B1, B2, p=0, B1_is_GF=False, B2_is_GF=False, 
                          A2=None, l=None, r=None):
        L = self.L
        M = len(B1[0].shape) - 2
        #print "M=", M
        Ae = list(self.A) * int(sp.ceil((L - 1. + 2. * M - 1.) / L))
        if A2 is None:
            A2e = A
        else:
            A2e = list(A2) * int(sp.ceil((L - 1. + 2. * M - 1.) / L))
        #print len(A)
        
        if l is None:
            l = self.l
        if r is None:
            r = self.r
        
        #ter = range(len(A))
        
        res = 0
        res12 = 0
        res21 = 0
        for k in range(L):
            res += m.adot(l[k - 1], tm.eps_r_noop_multi(r[(k + M - 1) % L], [B1[k]], [B2[k]]))
            
            #B1 right of B2
            if not B1_is_GF:
                for j in range(1, M):
                    #print k, j, ter[k:k + j], ter[k + M:k + M + j]
                    ket = Ae[k:k + j] + [B1[(k + j) % L]]
                    bra = [B2[k]] + A2e[k + M:k + M + j]
                    R = tm.eps_r_noop_multi(r[(k + M + j - 1) % L], ket, bra)
                    #print la.norm(R), m.adot(self.l[k - 1], R)
                    res21 += sp.exp((j + 1) * 1.j * p) * m.adot(l[k - 1], R)
            
            #B2 right of B1
            if not B2_is_GF:
                for j in range(1, M):
                    ket = [B1[k]] + Ae[k + M:k + M + j]
                    bra = A2e[k:k + j] + [B2[(k + j) % L]]
                    R = tm.eps_r_noop_multi(r[(k + M + j - 1) % L], ket, bra)
                    #print la.norm(R), m.adot(self.l[k - 1], R)
                    res12 += sp.exp(-(j + 1) * 1.j * p) * m.adot(l[k - 1], R)
                    
            if M == 1:
                res21 += sp.exp(1.j * p) * m.adot(l[k - 1], tm.eps_r_noop(tm.eps_r_noop(r[(k + 1) % L], B1[(k + 1) % L], A2e[(k + 1) % L]), Ae[k], B2[k]))
                res12 += sp.exp(-1.j * p) * m.adot(l[k - 1], tm.eps_r_noop(tm.eps_r_noop(r[(k + 1) % L], Ae[(k + 1) % L], B2[(k + 1) % L]), B1[k], A2e[k]))
            
        if M == 1:
            #print "res12, res21", res12, res21
            res12 = res21 = 0
        
        return res, res12, res21
        
        
    def _B_overlap(self, B1, B2, p=0, B1_is_GF=False, B2_is_GF=False, 
                   con_tol=None, GF_tol=1E-15, other=None):
        L = self.L
        
        if other is None:
            l = self.l
            r = self.r
            A2 = self.A
        else:
            A2 = other.A
            
            d, ev, lL = self.fidelity_per_site(other, full_output=True, left=True)
            d, ev, rL = self.fidelity_per_site(other, full_output=True, left=False)
            if not abs(1 - d) < 1E-12:
                log.warning("Warning! Attempt to calculate tangent vector overlap for non-equivalent states.", d)
            l = [None] * L
            r = [None] * L
            nrm = abs(m.adot(lL, rL))
            if not abs(nrm - 1) < 1E-12:
                log.warning("Warning! Normalization error in _B_overlap.", nrm)
            l[-1] = lL.reshape(self.l[-1].shape)
            r[-1] = rL.reshape(self.r[-1].shape)

            for k in range(len(self.A) - 1, 0, -1):
                r[k - 1] = tm.eps_r_noop(r[k], self.A[k], A2[k])
                
            for k in range(0, len(self.A) - 1):
                l[k] = tm.eps_l_noop(l[k - 1], self.A[k], A2[k])
        
        res, res12, res21 = self._B_overlap_onsite(B1, B2, p=p, B1_is_GF=B1_is_GF, B2_is_GF=B2_is_GF, 
                                                   A2=A2, l=l, r=r)
        #print "onsite = ", res, res12, res21
        
        res += res12 + res21
            
        def calc_con(BnoGF, BGF, p_, noGF_conj):
            if noGF_conj:
                p_ = -p_
            BR, BRbl_fid = self._B_overlap_calc_BR(BnoGF, p_, conj=noGF_conj, 
                                                   con_tol=con_tol, GF_tol=GF_tol,
                                                   A2=A2, l=l, r=r)

            rescon = 0
            if not BR[0] is None:
                M = len(BGF[0].shape) - 2
                Ae = list(self.A) * int(sp.ceil((L - 1. + M) / L))
                A2e = list(A2) * int(sp.ceil((L - 1. + M) / L))

                As = [None] * L
                A2s = [None] * L
                BGFs = [None] * L
                for k in range(L):
                    As[k] = Ae[k:k + M]
                    A2s[k] = A2e[k:k + M]
                    BGFs[k] = [BGF[k]]
                
                if noGF_conj:
                    Y1 = BGFs
                    Y2 = A2s
                else:
                    Y1 = As
                    Y2 = BGFs
                #rescon_1 = 0
                for k in range(L):
                    #print "sum BR", sp.sum(BR[k].ravel()), len(Y1[k]), len(Y2[k])
                    if True or M == 1:
                        R = tm.eps_r_noop_multi(BR[(k + M) % L], Y1[k], Y2[k])
                        #R2 = sp.exp(1.j * p_) * tm.eps_r_noop(tm.eps_r_noop(BR[(k + 2) % L], self.A[(k+1)%L], self.A[(k+1)%L]), Y1[k][0], Y2[k][0])
                        #rescon_1 += m.adot(self.l[k - 1], R2)
                    else:
                        if noGF_conj:
                            R = tm.eps_r_op_2s_C12(BR[(k + M) % L], BGF[k], A2e[k], A2e[(k + 1) % L])
                        else:
                            R = tm.eps_r_op_2s_C34(BR[(k + M) % L], Ae[k], Ae[(k + 1) % L], BGF[k])
                    rescon += m.adot(l[k - 1], R)
                rescon *= sp.exp(M * 1.j * p_)
                #rescon_1 *= sp.exp(M * 1.j * p_)
                #print "rescon_1", rescon_1
                #print "rescon - rescon_1", rescon - rescon_1
            return rescon, BRbl_fid

        if not B2_is_GF:
            con_B1_right, B2_overlap = calc_con(B2, B1, p, True)
            res += con_B1_right
            #print "con_B1_right = ", con_B1_right
            #print "con_B1_right + os_21", con_B1_right + res21
            #print "B2_overlap = ", B2_overlap
        else:
            B2_overlap = 0

        if not B1_is_GF:
            con_B2_right, B1_overlap = calc_con(B1, B2, p, False)
            res += con_B2_right
            #print "con_B2_right = ", con_B2_right
            #print "con_B2_right + os_12", con_B2_right + res12
            #print "B1_overlap = ", B1_overlap
        else:
            B1_overlap = 0
            
        #print "overlap part", B1_overlap * B2_overlap
        res -= B1_overlap * B2_overlap
        
        #print "final", res
        
        return res
        
    def _B_random_GF(self, M=1):
        sha = (self.q**(M - 1), self.D, (self.q - 1) * self.D)
        x1 = [sp.rand(*sha) + 1.j * sp.rand(*sha) for k in range(self.L)]
        
        B1 = [None] * self.L
        for k in range(self.L):
            VR = sp.transpose(self.Vsh[(k + M - 1) % self.L].conj(), axes=(0, 2, 1))
            VRr_si = sp.array([VRs.dot(self.r_sqrt_i[(k + M - 1) % self.L].A) for VRs in VR])
            B1[k] = sp.zeros((self.q**M, self.D, self.D), dtype=self.typ)
            for s in range(self.q**(M)):
                B1[k][s] = self.l_sqrt_i[k - 1].dot(x1[k][s // self.q]).dot(VRr_si[s % self.q])
            B1[k] = B1[k].reshape(tuple([self.q] * M + [self.D, self.D]))
            
        etasq1 = sp.sum([m.adot(x1k, x1k) for x1k in x1])
        
        return B1, x1, etasq1

    def _B_random_GT_par(self, M=1, fac=1.):
        sha = (self.q**(M - 1), self.D, self.D * (self.q - 1))
        x = [fac * (sp.rand(*sha) + 1.j * sp.rand(*sha)) for k in range(self.L)]
        
        return x
        
    def _B_rand_GT(self, B, p=0, fac=1., x=None):
        L = self.L
        q = self.q
        M = len(B[0].shape) - 2
        if x is None:
            x = self._B_random_GT_par(M=M, fac=fac)
            
        B2 = [Bk.copy() for Bk in B]
        for k in range(L):
            sha = (q**M, self.D, self.D)
            B2k = B2[k].reshape(*sha)
            for s in range(q**M):
                B2k[s] += (self.A[k][s // q**(M - 1)].dot(x[(k + 1) % L][s % q**(M - 1)]) 
                           - sp.exp(-1.j * p) 
                             * x[k][s // q].dot(self.A[(k + M - 1) % L][s % q]))
                                             
        return B2
    
    def _B_overlap_test(self, M=1, np=5):
        self.calc_B()
        
        B1, x1, etasq1 = self._B_random_GF(M=M)
        
        B1B1 = self._B_overlap(B1, B1, p=0, B1_is_GF=True, B2_is_GF=True)
        B1B1nGF = self._B_overlap(B1, B1, p=0)
        B1_, x1_ = self._B_to_B_GF(B1, p=0)
        B1B1_ = self._B_overlap(B1, B1_, p=0, B1_is_GF=True, B2_is_GF=True)
        print("Initial check:", etasq1, B1B1, B1B1nGF, B1B1_)
        
        y1 = self._B_random_GT_par(M=M, fac=etasq1)
        y2 = self._B_random_GT_par(M=M, fac=etasq1)
        
        for p in sp.linspace(0, sp.pi, num=np):
            B2 = self._B_rand_GT(B1, p=p, x=y1)
            B3 = self._B_rand_GT(B1, p=p, x=y2)
            
            print("p=", p)
            print("------------")
            overlap = self._B_overlap(B3, B3, p)
            print("B3,B3", overlap, abs(overlap - etasq1))
            overlap = self._B_overlap(B2, B3, p)
            print("B2,B3", overlap, abs(overlap - etasq1))
            overlap = self._B_overlap(B1, B2, p, B1_is_GF=True)
            print("B1,B2", overlap, abs(overlap - etasq1))
            overlap = self._B_overlap(B2, B1, p, B2_is_GF=True)
            print("B2,B1", overlap, abs(overlap - etasq1))
            
            B2_, x2_ = self._B_to_B_GF(B2, p=p)
            overlap = self._B_overlap(B1, B2_, p, B1_is_GF=True, B2_is_GF=True)
            print("B1,B2_proj", overlap, abs(overlap - etasq1))
            print("------------")
            print() 
        
    def _B_overlap_GF(self, B1, B2):
        """Note: Both Bs must satisfy the GFC!
        """
        return self._B_overlap(B1, B2, B1_is_GF=True, B2_is_GF=True)
        
    def calc_B_CG(self, BCG0, BG0, BG0dotBG0, tau0, tau_init=0.05,
                  reset=False, use_PR=True, B_overlap_tol=1E-6):
        """Calculates a tangent vector using the non-linear conjugate gradient method.
        
        Parameters:
            BCG0 : ndarray
                Tangent vector used to make the previous step. Ignored on reset.
            BG0 : ndarray
                Gradient tangent vector from previous step. Ignored on reset.
            BdotB_0 : float
                Norm of the previous tangent vector. Ignored on reset.
            tau0 : float
                Step length from the previous iteration. ignored on reset.
            tau_init : float = 0.05
                A base step size for initialising line searches. Should usually decrease the energy.
            reset : bool = False
                Whether to perform a manual reset, using the gradient as the next search direction.
            use_PR : bool = True
                Whether to use the Polak-Ribière formula for beta (rather than Fletcher-Reeves).
        """
        BG = self.calc_B()
        BGdotBG = self.eta.real**2
    
        if reset:
            beta = 0.
            log.debug("CG RESET")
            
            BCG = BG
        else:
            if use_PR:                
                BGdotBG0 = self._B_overlap(BG, BG0, B1_is_GF=True)
                beta = (BGdotBG - BGdotBG0) / BG0dotBG0 #Note: Overall factors/signs on the gradients do not affect beta
            else:
                beta = BGdotBG / BG0dotBG0 #FR
        
            log.debug("Beta = %s", beta)
            
            beta = max(0, beta.real)
            
            if beta > 100:
                log.warning("CG: RESET due to large beta!")
                BCG = BG
            else:
                BCG0, BCG0x = self._B_to_B_GF(BCG0)
                BCG = [None] * self.L 
                for k in range(self.L):
                    BCG[k] = BG[k] + beta * BCG0[k]
        
        ls = EvoMPS_line_search(self, BCG, BG, B_overlap_tol=B_overlap_tol)
        g0 = ls.gs[0].real
        if g0 > 0:
            log.info("CG: RESET due to bad search direction.")
            self.update() #Restores CF
            BG = self.calc_B()
            BCG = BG
            ls = EvoMPS_line_search(self, BCG, BG, B_overlap_tol=B_overlap_tol)
            g0 = ls.gs[0].real
        
        try:
            tau, h_min, ind = ls.brentq(max(tau0, tau_init * 0.5))
        except (EvoMPSNoConvergence, EvoMPSNormError):
            tau = 0

        if tau == 0:
            log.warning("CG RESET with dtau_init due to failed line search!")
            retry = True
        elif h_min - ls.hs[0] > max(self.itr_rtol * 10, 1E-14):
            log.warning("CG RESET with dtau_init due to energy increase!")
            retry = True
        else:
            retry = False

        if retry:
            self.update() #Restores CF
            BG = self.calc_B()
            BCG = BG
            ls = EvoMPS_line_search(self, BCG, BG, B_overlap_tol=B_overlap_tol)
            try:
                tau, h_min, ind = ls.brentq(tau_init)
            except (EvoMPSNoConvergence, EvoMPSNormError):
                tau = 0
            
        if tau > 0:
            self.lL_before_CF = ls.lLs[ind]
            self.rL_before_CF = ls.rLs[ind]
            self.K0_init = ls.K0s[ind]
        else:
            log.warning("CG retries failed. Taking a small GD step.")
            tau = tau_init #If all else fails, take a blind imtime step
        
        return BCG, BG, BGdotBG, tau
        
            
    def export_state(self, userdata=None):
        if userdata is None:
            userdata = self.userdata

        lL = np.asarray(self.l[-1])
        rL = np.asarray(self.r[-1])
            
        tosave = np.empty((5), dtype=np.ndarray)
        if self.L == 1:
            tosave[0] = self.A[0]
        else:
            tosave[0] = sp.array(self.A)
        tosave[1] = lL
        tosave[2] = rL
        tosave[3] = self.K[0]
        tosave[4] = np.asarray(userdata)
        
        return tosave
            
    def save_state(self, file, userdata=None):
        np.save(file, self.export_state(userdata))
        
    def import_state(self, state, expand=False, truncate=False, 
                     expand_q=False, shrink_q=False, refac=0.1, imfac=0.1,
                     do_update=True):
        newA = state[0]
        newlL = state[1]
        newrL = state[2]
        newK0 = state[3]
        if state.shape[0] > 4:
            self.userdata = state[4]
            
        if len(newA.shape) == 3:
            newA = [newA]
        elif len(newA.shape) == 4:
            newA = list(newA)
        
        if (len(newA) == self.L and newA[0].shape == self.A[0].shape):
            self.A = newA
            self.K[0] = newK0
            self.l[-1] = np.asarray(newlL)
            self.r[-1] = np.asarray(newrL)
            self.lL_before_CF = self.l[-1]
            self.rL_before_CF = self.r[-1]
        elif expand and len(newA) == self.L and (
        len(newA[0].shape) == 3) and (newA[0].shape[0] == 
        self.A[0].shape[0]) and (newA[0].shape[1] == newA[0].shape[2]) and (
        newA[0].shape[1] <= self.A[0].shape[1]):
            newD = self.D
            savedD = newA[0].shape[1]
            self._init_arrays(savedD, self.q, self.L)
            self.A = newA
            self.K[0] = newK0
            self.l[-1] = np.asarray(newlL)
            self.r[-1] = np.asarray(newrL)
            self.expand_D(newD, refac, imfac)
            self.lL_before_CF = self.l[-1]
            self.rL_before_CF = self.r[-1]
            log.warning("EXPANDED!")
        elif truncate and len(newA) == self.L and (len(newA[0].shape) == 3) \
                and (newA[0].shape[0] == self.A[0].shape[0]) \
                and (newA[0].shape[1] == newA[0].shape[2]) \
                and (newA[0].shape[1] >= self.A[0].shape[1]):
            newD = self.D
            savedD = newA[0].shape[1]
            self._init_arrays(savedD, self.q, self.L)
            self.A = newA
            self.K[0] = newK0
            self.l[-1] = np.asarray(newlL)
            self.r[-1] = np.asarray(newrL)
            self.update()  # to make absolutely sure we're in CF
            self.truncate(newD, update=True)
            log.warning("TRUNCATED!")
        elif expand_q and len(newA) == self.L and (len(newA[0].shape) == 3) and (
        newA[0].shape[0] <= self.A[0].shape[0]) and (newA[0].shape[1] == 
        newA[0].shape[2]) and (newA[0].shape[1] == self.A[0].shape[1]):
            newQ = self.q
            savedQ = newA[0].shape[0]
            self._init_arrays(self.D, savedQ, self.L)
            self.A = newA
            self.K[0] = newK0
            self.l[-1] = np.asarray(newlL)
            self.r[-1] = np.asarray(newrL)
            self.expand_q(newQ)
            self.lL_before_CF = self.l[-1]
            self.rL_before_CF = self.r[-1]
            log.warning("EXPANDED in q!")
        elif shrink_q and len(newA) == self.L and (len(newA[0].shape) == 3) and (
        newA[0].shape[0] >= self.A[0].shape[0]) and (newA[0].shape[1] == 
        newA[0].shape[2]) and (newA[0].shape[1] == self.A[0].shape[1]):
            newQ = self.q
            savedQ = newA[0].shape[0]
            self._init_arrays(self.D, savedQ)
            self.A = newA
            self.K[0] = newK0
            self.l[-1] = np.asarray(newlL)
            self.r[-1] = np.asarray(newrL)
            self.shrink_q(newQ)
            self.lL_before_CF = self.l[-1]
            self.rL_before_CF = self.r[-1]
            log.warning("SHRUNK in q!")
        else:
            log.error("Failed to import state!")
            return False
            
        if do_update:
            self.update()
            
        return True
            
    def load_state(self, file, expand=False, truncate=False, expand_q=False,
                   shrink_q=False, refac=0.1, imfac=0.1):
        state = np.load(file, allow_pickle=True)
        return self.import_state(state, expand=expand, truncate=truncate,
                                 expand_q=expand_q, shrink_q=shrink_q,
                                 refac=refac, imfac=imfac)

    @classmethod
    def from_file(cls, file, ham, ham_sites=None):
        state = np.load(file, allow_pickle=True)
        newA = state[0]
        if len(newA.shape) == 3:
            newA = [newA]
        elif len(newA.shape) == 4:
            newA = list(newA)
        dtype = newA[0].dtype
        L = len(newA)
        q, D, _ = newA[0].shape
        tdvp = cls(D, q, ham, ham_sites, L, dtype, False)
        tdvp.import_state(state)
        return tdvp

    def set_q(self, newq, offset=0):
        oldK = self.K        
        super(EvoMPS_TDVP_Uniform, self).set_q(newq, offset=offset)        
        self.K = oldK
        print("New Hamiltonian required! Use set_ham(...).")
                    
    def expand_D(self, newD, refac=100, imfac=0):
        oldK0 = self.K[0]
        oldD = self.D

        super(EvoMPS_TDVP_Uniform, self).expand_D(newD, refac=refac, imfac=imfac)
        #self._init_arrays(newD, self.q)
        
        self.K[0][:oldD, :oldD] = oldK0
        #self.K[oldD:, :oldD].fill(0 * 1E-3 * la.norm(oldK) / oldD**2)
        #self.K[:oldD, oldD:].fill(0 * 1E-3 * la.norm(oldK) / oldD**2)
        #self.K[oldD:, oldD:].fill(0 * 1E-3 * la.norm(oldK) / oldD**2)
        val = abs(oldK0.mean())
        m.randomize_cmplx(self.K[0].ravel()[oldD**2:], a=0, b=val, aj=0, bj=0)
        
    def expect_2s(self, op, k=0):
        if op is self.ham and self.ham_sites == 2:
            res = tm.eps_r_op_2s_C12_AA34(self.r[(k + 1) % self.L], self.C[k], self.AA[k])
            return m.adot(self.l[(k - 1) % self.L], res)
        else:
            return super(EvoMPS_TDVP_Uniform, self).expect_2s(op, k=k)
            
    def expect_3s(self, op, k=0):
        if op is self.ham and self.ham_sites == 3:
            res = tm.eps_r_op_3s_C123_AAA456(self.r[(k + 2) % self.L], self.C[k], self.AAA[k])
            return m.adot(self.l[(k - 1) % self.L], res)
        else:
            return super(EvoMPS_TDVP_Uniform, self).expect_3s(op, k=k)
            
    def _get_tangent_lr(self, B, d, nb=None, op1=None, s2=None):
        if nb is None:
            nb = d + 1
        if s2 is None:
            A2 = self.A[0]
            r = self.r
        else:
            A2 = s2.A[0]
            r = s2.r
        As = [self.A[0]] * nb + [B] + [A2] * (2*d + 1 - nb)
        
        ls = [self.l[0]] * (2*d + 2)
        ls[0] = self.l[0]
        if op1 is None:
            ls[0] = self.l[0]
            ls[1] = tm.eps_l_noop(ls[0], As[1], As[1])
        else:
            ls[1] = tm.eps_l_op_1s(ls[0], As[1], As[1], op1)
            
        for n in range(2, 2 * d + 1):
            ls[n] = tm.eps_l_noop(ls[n - 1], As[n], As[n])
            
        rs = [None] * (2*d + 2)
        rs[-1] = r[0]
        for n in range(2*d + 1, 0, -1):
            rs[n - 1] = tm.eps_r_noop(rs[n], As[n], As[n])
            
        return ls, rs
            
    def expect_tangent_1s(self, B, op, d, nb=None, lrs=None, s2=None):
        if nb is None:
            nb = d + 1
        if lrs is None:
            ls, rs = self._get_tangent_lr(B, d, nb=nb, s2=s2)
        else:
            ls, rs = lrs
            
        if s2 is None:
            A2 = self.A[0]
        else:
            A2 = s2.A[0]
        As = [self.A[0]] * nb + [B] + [A2] * (2*d + 1 - nb)
            
        exs = [m.adot(ls[n - 1], tm.eps_r_op_1s(rs[n], As[n], As[n], op)) 
               for n in range(1, 2*d + 2)]
                   
        return sp.array(exs)
        
    def expect_tangent_2s(self, B, op, d):
        AA = tm.calc_AA(self.A[0], self.A[0])
        AB = tm.calc_AA(self.A[0], B)
        BA = tm.calc_AA(B, self.A[0])
        Cop = tm.calc_C_mat_op_AA(op, AA)
        CopAB = tm.calc_C_mat_op_AA(op, AB)
        CopBA = tm.calc_C_mat_op_AA(op, BA)
        
        nb = d + 1
        ls, rs = self._get_tangent_lr(B, d)
            
        AAs = [AA] * (2*d + 2)
        AAs[nb - 1] = AB 
        AAs[nb] = BA
        
        Cs = [Cop] * (2*d + 2)
        Cs[nb - 1] = CopAB
        Cs[nb] = CopBA
            
        exs = [m.adot(ls[n - 1], tm.eps_r_op_2s_AA12_C34(rs[n + 1], Cs[n], AAs[n])) 
               for n in range(1, 2*d + 1)]
                   
        return sp.array(exs)
        
    def correlation_tangent_1s_1s(self, B, op1, op2, d, nb=None, s2=None):
        ls, rs = self._get_tangent_lr(B, d, nb=nb, s2=s2)
        exs1 = self.expect_tangent_1s(B, op1, d, nb=nb, lrs=(ls, rs), s2=s2)
        exs2 = self.expect_tangent_1s(B, op2, d, nb=nb, lrs=(ls, rs), s2=s2)
        
        if nb is None:
            nb = d + 1

        if s2 is None:
            A2 = self.A[0]
        else:
            A2 = s2.A[0]
        As = [self.A[0]] * nb + [B] + [A2] * (2*d + 1 - nb)
        
        ls, rs = self._get_tangent_lr(B, d, op1=op1, nb=nb, s2=s2)
        g = [m.adot(ls[0], tm.eps_r_op_1s(rs[1], As[1], As[1], op1.dot(op2)))]
        g += [m.adot(ls[n - 1], tm.eps_r_op_1s(rs[n], As[n], As[n], op2)) 
              for n in range(2, 2*d + 2)]
        
        g = sp.array(g)
        
        c = g - exs1 * exs2
                   
        return c, g

class EvoMPS_line_search_wolfe_sat(Exception):
    def __init__(self, message, tau, *args):
        self.message = message
        self.tau = tau
        super(EvoMPS_line_search_wolfe_sat, self).__init__(message, tau, *args)

class EvoMPS_line_search():
    def __init__(self, tdvp, B, Bg, B_overlap_tol=1E-6):
        self.B = B
        self.Bg0 = Bg
        self.penalise_neg = True
        self.in_search = False
        self.con_tol = B_overlap_tol
        
        self.tdvp0 = tdvp
        self.tdvp = cp.deepcopy(tdvp)
        self.tdvp_tau = 0
        
        #Bring all l and r into dense form
        for k in range(tdvp.L):
            try:
                tdvp.l[k] = tdvp.l[k].A
            except AttributeError:
                pass
            
            try:
                tdvp.r[k] = tdvp.r[k].A
            except AttributeError:
                pass        
        
        self.taus = [0]            
        self.hs = [tdvp.h_expect.real]
        self.gs = [-2 * tdvp._B_overlap(Bg, B, B1_is_GF=True, con_tol=self.con_tol)]
        self.lLs = [tdvp.lL_before_CF.copy()]
        self.rLs = [tdvp.rL_before_CF.copy()]
        self.K0s = [tdvp.K[0]]
        self.Ws = [False]
            
    def clear_hist(self):
        self.taus = self.taus[:1]           
        self.hs = self.hs[:1]
        self.gs = self.gs[:1]
        self.lLs = self.lLs[:1]
        self.rLs = self.rLs[:1]
        self.K0s = self.K0s[:1]
        self.Ws = self.Ws[:1]
        
    def delete(self, i):
        self.taus.pop(i)
        self.hs.pop(i)
        self.gs.pop(i)
        self.lLs.pop(i)
        self.rLs.pop(i)
        self.K0s.pop(i)
        self.Ws.pop(i)
        
    def add_res(self, tau, h, g, lL, rL, K0, w):
        self.taus.append(tau)
        self.hs.append(h)
        self.gs.append(g)
        self.lLs.append(lL)
        self.rLs.append(rL)
        self.K0s.append(K0)
        self.Ws.append(w)
        
    def f(self, tau, *args):
        if self.penalise_neg and tau < 0:
            g = tau**2 + self.gs[0].real
            log.debug((tau, g, "punishing negative tau!"))
            self.add_res(tau, self.hs[0], self.gs[0], self.lLs[0], 
                         self.rLs[0], self.K0s[0], False)
            return g
        try:
            i = self.taus.index(tau)
            log.debug((tau, self.gs[i], "from stored"))
            return self.gs[i].real
        except ValueError:
            for k in range(self.tdvp.L):
                self.tdvp.A[k] = self.tdvp0.A[k] - tau * self.B[k]
            
            #Use iterative results from nearest tau as starting points
            nearest_tau_ind = abs(sp.array(self.taus) - tau).argmin()
            self.lL_before_CF = self.lLs[nearest_tau_ind] #needn't copy these
            self.rL_before_CF = self.rLs[nearest_tau_ind]
            self.tdvp.K[0] = self.K0s[nearest_tau_ind].copy() #but this gets updated in place, so copy
            
            self.tdvp.update(restore_CF=False)
            Bg = self.tdvp.calc_B()
            self.tdvp_tau = tau
            
            g = -2 * self.tdvp._B_overlap(Bg, self.B, B1_is_GF=True, con_tol=self.con_tol)
            h_exp = self.tdvp.h_expect.real
            wol = self.wolfe(g, h_exp, tau)
            K0 = self.tdvp.K[0].copy()
            
            log.debug((tau, g, h_exp, h_exp - self.hs[0], wol,
                       self.tdvp.itr_l, self.tdvp.itr_r))
            
            self.add_res(tau, h_exp, g, self.tdvp.l[-1].copy(), 
                         self.tdvp.r[-1].copy(), K0, wol)

            if wol and self.in_search:
                raise EvoMPS_line_search_wolfe_sat("Wolfe conditions satisfied.", tau)
            
            return g.real
            
    def wolfe(self, g, h, tau, strong=False, c1=1E-4, c2=0.1):
        #I wonder how badly this breaks if the energy is innaccurate
        w1 = h <= self.hs[0] + c1 * tau * self.gs[0].real
        if strong:
            w2 = abs(g.real) <= c2 * abs(self.gs[0].real)
        else:
            w2 = g.real >= c2 * self.gs[0].real
        
        return w1 and w2
            
    def sane_first_step(self, dtau_init, max_itr=10):
        tau = dtau_init
        h = 1 + self.hs[0]
        g = -1
        i = 0
        while (g < 0 and h - self.hs[0] > 1E-13) and i < max_itr: #Energy increase with negative gradient!
            self.clear_hist() #delete history, since these steps too far will confuse later searches
        
            self.f(tau)
            ind = self.taus.index(tau)
            h = self.hs[ind]
            g = self.gs[ind]
            
            tau *= 0.5
            i += 1
            
        return ind
        
    def brentq(self, tau_init, tol=1E-3, max_iter=20):
        self.in_search = False
        i = self.sane_first_step(tau_init)
        tau_init = self.taus[i]
        g_init = self.gs[i]
                  
        if self.wolfe(g_init, self.hs[i], tau_init):
            log.debug("CG: Using initial step, since Wolfe satisfied!")
            tau_opt = tau_init
        else:
            self.in_search = True
            
            try:
                taus = None
                if g_init > 0: #We already have g0 < 0 - enough to bracket
                    taus = [0, tau_init]

                if taus is None:
                    for i in range(3):
                        taus = self.bracket_extrap(tau_init / (i + 1.))
                    
                        if not taus is None:
                            break
                        
                        self.clear_hist()
                    
                if taus is None:
                    return 0, self.hs[0], 0 #fail
                
                try:
                    tau_opt = opti.brentq(self.f, taus[0], taus[-1], xtol=tol, maxiter=max_iter)
                except ValueError:
                    log.warning("CG: Failed to find a valid bracket.")
                    return 0, self.hs[0], 0 #fail
                
            except EvoMPS_line_search_wolfe_sat as e:
                log.debug("CG: Aborting early due to Wolfe")
                tau_opt = e.tau
                
        i = self.taus.index(tau_opt)
        h_min = self.hs[i]
        
        self.in_search = False
        return tau_opt, h_min, i
        
    def bracket_extrap(self, dtau_init, fac_red=0.9, fac_inc=1.1, max_itr=20, max_deg=3, max_points=5):
        log.debug("CG: Polynomial extrapolation BEGIN")
        g0 = self.f(dtau_init)

        tau1 = dtau_init
        tau_prev = tau1
        g1 = g0
        
        i = 0
        while g1 * g0 > 0 and i < max_itr:
            tau_prev = tau1
            #Use only tau=0 and points we have visited.
            if len(self.gs) > 1:
                gs = sp.array(self.gs)
                pts = min(max_points, len(gs))
                res = sp.polyfit(self.taus[-pts:], gs[-pts:], max(max_deg, pts - 1), full=True)
                po = res[0]
                por = sp.roots(po)
                if por.max().real > 0:
                    tau1 = por.max().real
                    
                if tau1 > dtau_init * 10:
                    tau1 = min(tau_prev * 2, tau1)
        
            if g1 < 0:
                tau1 *= fac_inc
            else:
                tau1 *= fac_red
        
            g1 = self.f(tau1)
                    
            i += 1
            
        if i == max_itr:
            log.debug("CG: Polynomial extrapolation MAX ITR")
            return None
        else:
            log.debug("CG: Polynomial extrapolation DONE")
            return sorted([tau_prev, tau1])

    def bracket_step(self, dtau_init, fac_red=0.1, fac_inc=2.5, max_itr=10):
        g0 = self.f(dtau_init)
        tau1 = dtau_init
        g1 = g0
        i = 0
        while g1 * g0 > 0 and i < max_itr:
            if g1 < 0:
                tau1 *= fac_inc
            else:
                tau1 *= fac_red
            
            g1 = self.g(tau1)
            
            i += 1
            
        if i == max_itr:
            return None
        else:
            return sorted([dtau_init, tau1])