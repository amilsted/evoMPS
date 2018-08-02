# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 17:29:27 2011

@author: Ashley Milsted

"""
from __future__ import absolute_import, division, print_function

import scipy as sp
import scipy.linalg as la
from . import matmul as m
from . import tdvp_common as tm
import copy
import logging

log = logging.getLogger(__name__)

def correct_bond_dim_open_chain(D, q):
    N = len(D) - 1
    
    D[0] = 1
    D[N] = 1

    qacc = 1
    for n in range(N - 1, -1, -1):
        if qacc < D.max(): #Avoid overflow!
            qacc *= q[n + 1]

        if D[n] > qacc:
            D[n] = qacc
            
    qacc = 1
    for n in range(1, N + 1):
        if qacc < D.max(): #Avoid overflow!
            qacc *= q[n]

        if D[n] > qacc:
            D[n] = qacc
            
    return D

def mps_from_tensors_gen(A, do_update=True):
    N = len(A)-1
    
    assert A[0] is None, "Site zero does not exist, please set A[0] to None!"

    D = [An.shape[1] for An in A[1:]] + [A[-1].shape[2]]
    q = [0] + [An.shape[0] for An in A[1:]]

    s = EvoMPS_MPS_Generic(N, D, q)
    for n in range(1, N+1):
        s.A[n][:] = A[n]

    if do_update:
        s.update()

    return s

class EvoMPS_MPS_Generic(object):
    
    def __init__(self, N, D, q):
        """Creates a new EvoMPS_MPS_Generic object.
        
        This class implements basic operations on a generic MPS with
        open boundary conditions on a finite chain.
        
        Performs self._correct_bond_dimension().
        
        Sites are numbered 1 to N.
        self.A[n] is the parameter tensor for site n
        with shape == (q[n], D[n - 1], D[n]).
        
        Parameters
        ----------
        N : int
            The number of lattice sites.
        D : ndarray
            A 1d array, length N + 1, of integers indicating the desired 
            bond dimensions.
        q : ndarray
            A 1d array, length N + 1, of integers indicating the 
            dimension of the hilbert space for each site. 
            Entry 0 is ignored (there is no site 0).
         
        """
        
        self.odr = 'C'
        self.typ = sp.complex128
        
        self.sanity_checks = False
        """Whether to perform additional (potentially costly) sanity checks."""
        
        self.canonical_form = 'right'
        """Canonical form to use when performing restore_CF(). Possible
           settings are 'right' and 'left'."""
        
        self.eps = sp.finfo(self.typ).eps
        
        self.zero_tol = sp.finfo(self.typ).resolution
        """Tolerance for detecting zeros. This is used when (pseudo-) inverting 
           l and r."""
        
        self.N = N
        """The number of sites. Do not change after initializing."""
        
        self.D = sp.array(D)
        """Vector containing the bond-dimensions. A[n] is a 
           q[n] x D[n - 1] x D[n] tensor."""
        
        self.q = sp.array(q)
        """Vector containing the site Hilbert space dimensions. A[n] is a 
           q[n] x D[n - 1] x D[n] tensor."""

        if (self.D.ndim != 1) or (self.q.ndim != 1):
            raise ValueError('D and q must be 1-dimensional!')
            
        if (self.D.shape[0] != N + 1) or (self.q.shape[0] != N + 1):
            raise ValueError('D and q must have length N + 1')

        self._correct_bond_dimension()
        
        self._init_arrays()
        
        self.initialize_state()
    
    def _init_arrays(self):
        self.A = sp.empty((self.N + 1), dtype=sp.ndarray) #Elements 1..N
        
        self.r = sp.empty((self.N + 1), dtype=sp.ndarray) #Elements 0..N
        self.l = sp.empty((self.N + 1), dtype=sp.ndarray)        
        
        self.r[0] = sp.zeros((self.D[0], self.D[0]), dtype=self.typ, order=self.odr)  
        self.l[0] = m.eyemat(self.D[0], dtype=self.typ)
    
        for n in range(1, self.N + 1):
            self.r[n] = sp.zeros((self.D[n], self.D[n]), dtype=self.typ, order=self.odr)
            self.l[n] = sp.zeros((self.D[n], self.D[n]), dtype=self.typ, order=self.odr)
            self.A[n] = sp.zeros((self.q[n], self.D[n - 1], self.D[n]), dtype=self.typ, order=self.odr)
            
        sp.fill_diagonal(self.r[self.N], 1.)        
    
    def initialize_state(self):
        """Initializes the state to a hard-coded full rank state with norm 1.
        """
        for n in range(1, self.N + 1):
            self.A[n].fill(0)
            
            f = sp.sqrt(1. / self.q[n])
            
            if self.D[n-1] == self.D[n]:
                for s in range(self.q[n]):
                    sp.fill_diagonal(self.A[n][s], f)
            else:
                x = 0
                y = 0
                s = 0
                
                if self.D[n] > self.D[n - 1]:
                    f = 1.
                
                for i in range(max((self.D[n], self.D[n - 1]))):
                    self.A[n][s, x, y] = f
                    x += 1
                    y += 1
                    if x >= self.A[n][s].shape[0]:
                        x = 0
                        s += 1
                    elif y >= self.A[n][s].shape[1]:
                        y = 0
                        s += 1
    

    def set_state_product(self, psi, do_update=True):
        """Initializes the parameter tensors self.A from a pure product state.
        The state psi is given as a list of self.N self.q-dimensional vectors.
        The bond dimensions will be the same as specified in self.D, hence the
        result may have low rank. 
        
        It is recommended to use this method together with a bond dimension 
        of 1 if you plan to use the TDVP (with bond-dimension expansion) to 
        evolve the state.

        Parameters
        ----------
        psi : list of vectors
            The product state to be transformed into an MPS.

        """

        for n in range(self.N):
            self.A[n+1].fill(0)
            for q in range(self.q[n]):
                self.A[n+1][q,0,0] = psi[n][q]

        if do_update:
            self.update()

    def set_state_from_tensors(self, A, do_update=True):
        """Updates the state from a list of tensors A. Like self.A, A[0] is ignored and
        must be set to None for this method to work. The number of sites cannot change.

        Parameters
        ----------
        A : list of tensors, each with 3 dimensions. A[n] has dimensions (q[n],D[n-1],D[n]). A[0] is None.
        do_update : bool (True)
            Whether to perform self.update() after changing the tensors.
        """
        N = len(A)-1
        
        assert N == self.N, "Number of sites must match!"
        assert A[0] is None, "Site zero does not exist, please set A[0] to None!"

        self.D = sp.array([An.shape[1] for An in A[1:]] + [A[-1].shape[2]])
        self.q = sp.array([0] + [An.shape[0] for An in A[1:]])
        self._init_arrays()

        for n in range(1, N+1):
            self.A[n][:] = A[n]

        if do_update:
            self.update()

    def randomize(self, do_update=True):
        """Randomizes the parameter tensors self.A.
        
        Parameters
        ----------
        do_update : bool (True)
            Whether to perform self.update() after randomizing.
        """
        for n in range(1, self.N + 1):
            self.A[n] = ((sp.rand(*self.A[n].shape) - 0.5) 
                         + 1.j * (sp.rand(*self.A[n].shape) - 0.5))
            self.A[n] /= la.norm(self.A[n])
        
        if do_update:
            self.update()

        
    def add_noise(self, fac=1.0, do_update=True):
        """Adds some random (white) noise of a given magnitude to the parameter 
        tensors A.
        
        Parameters
        ----------
        fac : number
            A factor determining the amplitude of the random noise.
        do_update : bool (True)
            Whether to perform self.update() after randomizing.
        """
        for n in range(1, self.N + 1):
            self.A[n].real += (sp.rand(*self.A[n].shape) - 0.5) * 2 * fac
            self.A[n].imag += (sp.rand(*self.A[n].shape) - 0.5) * 2 * fac
            
        if do_update:
            self.update()
        
    def _correct_bond_dimension(self):
        """Reduces bond dimensions to the maximum physically useful values.
        
        Bond dimensions will be adjusted where they are too high to be useful
        (when they would be higher than the corresponding maximum
        Schmidt ranks). The maximum value for D[n] is the minimum of the 
        dimensions of the two partial Hilbert spaces corresponding to a cut 
        between sites n and n + 1.
        """                
        
        self.D = correct_bond_dim_open_chain(self.D, self.q)
                
    
    def update(self, restore_CF=True, normalize=True, auto_truncate=False, restore_CF_after_trunc=True):
        """Updates secondary quantities to reflect the state parameters self.A.
        
        Must be used after changing the parameters self.A before calculating
        physical quantities, such as expectation values.
        
        Also (optionally) restores the right canonical form.
        
        Parameters
        ----------
        restore_RCF : bool (True)
            Whether to restore right canonical form.
        normalize : bool
            Whether to normalize the state in case restore_CF is False.
        auto_truncate : bool (True)
            Whether to automatically truncate the bond-dimension if
            rank-deficiency is detected. Requires restore_RCF.
        restore_RCF_after_trunc : bool (True)
            Whether to restore_RCF after truncation.
            
        """
        assert restore_CF or not auto_truncate, "auto_truncate requires restore_RCF"
        
        if restore_CF:
            self.restore_CF()
            if auto_truncate:
                data = self.auto_truncate(update=False, 
                                          return_update_data=not restore_CF_after_trunc)
                if data:
                    log.info("Auto-truncated! New D: %s", self.D)
                    if restore_CF_after_trunc:
                        self.restore_CF()
                    else:
                        self._update_after_truncate(*data)
        else:
            self.calc_l()
            if normalize:
                self.simple_renorm(update_r=False)
            self.calc_r()
    
    def calc_l(self, n_low=-1, n_high=-1):
        """Updates the l matrices to reflect the current state parameters self.A.
        
        Implements step 5 of the TDVP algorithm or, equivalently, eqn. (41).
        (arXiv:1103.0936v2 [cond-mat.str-el])
        """
        if n_low < 0:
            n_low = 1
        if n_high < 0:
            n_high = self.N
        for n in range(n_low, n_high + 1):
            self.l[n] = tm.eps_l_noop(self.l[n - 1], self.A[n], self.A[n])
    
    def calc_r(self, n_low=-1, n_high=-1):
        """Updates the r matrices using the current state parameters self.A.
        
        Implements step 5 of the TDVP algorithm or, equivalently, eqn. (41).
        (arXiv:1103.0936v2 [cond-mat.str-el])
        """
        if n_low < 0:
            n_low = 0
        if n_high < 0:
            n_high = self.N - 1
        for n in range(n_high, n_low - 1, -1):
            self.r[n] = tm.eps_r_noop(self.r[n + 1], self.A[n + 1], self.A[n + 1])
    
    def inner(self, other_mps):
        """Computes the inner product between the present MPS and another.
        Computes <other|this>.
        """
        if self.N != other_mps.N:
            raise ValueError('Number of sites must match!')
        r = self.r[self.N]
        for n in range(self.N, 0, -1):
            r = tm.eps_r_noop(r, self.A[n], other_mps.A[n])
        return r.ravel()[0]

    def simple_renorm(self, update_r=True):
        """Renormalize the state by altering A[N] by a factor.
        
        We change A[N] only, which is a column vector because D[N] = 1, using a factor
        equivalent to an almost-gauge transformation where all G's are the identity, except
        G[N], which represents the factor. "Almost" means G[0] =/= G[N] (the norm is allowed to change).
        
        Requires that l is up to date. 
        
        Note that this generally breaks canonical form
        because we change r[N - 1] by the same factor.
        
        By default, this also updates the r matrices to reflect the change in A[N].
        
        Parameters
        ----------
        update_r : bool (True)
            Whether to call update all the r matrices to reflect the change.
        """
        norm = self.l[self.N][0, 0].real
        G_N = 1 / sp.sqrt(norm)
        
        self.A[self.N] *= G_N
        
        self.l[self.N][:] *= 1 / norm
        
        if update_r:
            for n in range(self.N):
                self.r[n] *= 1 / norm    
                
    def restore_CF(self, use_QR=True):
        if self.canonical_form == 'right':
            self.restore_RCF(use_QR=use_QR)
        else:
            self.restore_LCF(use_QR=use_QR)

    def _are_bond_dims_synced(self):
        Dcorrect = self.D[0] == self.A[1].shape[1]
        if Dcorrect:
            for n in range(1,self.N+1):
                if self.D[n] != self.A[n].shape[2]:
                    Dcorrect = False
                    break
        return Dcorrect

    def restore_RCF(self, use_QR=True, update_l=True, diag_l=True):
        """Use a gauge-transformation to restore right canonical form.
        
        Implements the conditions for right canonical form from sub-section
        3.1, theorem 1 of arXiv:quant-ph/0608197v2.
        
        It is possible to skip the diagonalization of the l's, such that
        only the right orthonormalization condition (r_n = eye) is met.
        
        By default, the l's are updated even if diag_l=False.
        
        Parameters
        ----------
        update_l : bool
            Whether to call calc_l() after completion (defaults to True)
        diag_l : bool
            Whether to put l in diagonal form (defaults to True)
        """   
        if use_QR:
            tm.restore_RCF_r_seq(self.A, self.r, sanity_checks=self.sanity_checks,
                                     sc_data="restore_RCF_r")
            if not self._are_bond_dims_synced():
                log.info("Bond dimension changed during restore_RCF.")
                A = copy.copy(self.A)
                r = copy.copy(self.r)
                self.set_state_from_tensors(A, do_update=False)
                self.r = r
        else:
            G_n_i = sp.eye(self.D[self.N], dtype=self.typ) #This is actually just the number 1
            for n in range(self.N, 0, -1):
                self.r[n - 1], G_n, G_n_i = tm.restore_RCF_r(self.A[n], self.r[n], 
                                                             G_n_i, sc_data=('site', n),
                                                             zero_tol=self.zero_tol,
                                                             sanity_checks=self.sanity_checks)

        if self.sanity_checks:
            if not sp.allclose(self.r[0].A, 1, atol=1E-12, rtol=1E-12):
                log.warning("Sanity Fail in restore_RCF!: r_0 is bad / norm failure")
        
        if diag_l:
            tm.restore_RCF_l_seq(self.A, self.l, sanity_checks=self.sanity_checks,
                                                 sc_data="restore_RCF_l")

            if self.sanity_checks:
                if not sp.allclose(self.l[self.N].A, 1, atol=1E-12, rtol=1E-12):
                    log.warning("Sanity Fail in restore_RCF!: l_N is bad / norm failure")
                    log.warning("l_N = %s", self.l[self.N].squeeze().real)
                
                for n in range(1, self.N + 1):
                    r_nm1 = tm.eps_r_noop(self.r[n], self.A[n], self.A[n])
                    #r_nm1 = tm.eps_r_noop(m.eyemat(self.D[n], self.typ), self.A[n], self.A[n])
                    if not sp.allclose(r_nm1, self.r[n - 1], atol=1E-11, rtol=1E-11):
                        log.warning("Sanity Fail in restore_RCF!: r_%u is bad (off by %g)", n, la.norm(r_nm1 - self.r[n - 1]))
        elif update_l:
            self.calc_l()
            
    def restore_LCF(self, use_QR=True, update_r=True, diag_r=True):
        """Use a gauge-transformation to restore left canonical form.
        
        See restore_RCF.
        """
        if use_QR:
            tm.restore_LCF_l_seq(self.A, self.l, sanity_checks=self.sanity_checks,
                                     sc_data="restore_LCF_l")
            if not self._are_bond_dims_synced():
                log.info("Bond dimension changed during restore_LCF.")
                A = copy.copy(self.A)
                l = copy.copy(self.l)
                self.set_state_from_tensors(A, do_update=False)
                self.l = l
        else:
            G = sp.eye(self.D[0], dtype=self.typ) #This is actually just the number 1
            for n in range(1, self.N + 1):
                self.l[n], G, Gi = tm.restore_LCF_l(self.A[n], self.l[n - 1], G,
                                                    zero_tol=self.zero_tol,
                                                    sanity_checks=self.sanity_checks)
        
        if self.sanity_checks:
            lN = tm.eps_l_noop(self.l[self.N - 1], self.A[self.N], self.A[self.N])
            if not sp.allclose(lN, 1, atol=1E-12, rtol=1E-12):
                log.warning("Sanity Fail in restore_LCF!: l_N is bad / norm failure")
        
        if diag_r:
            tm.restore_LCF_r_seq(self.A, self.r, sanity_checks=self.sanity_checks,
                                                 sc_data="restore_LCF_r")
    
            if self.sanity_checks:
                if not sp.allclose(self.r[0].A, 1, atol=1E-12, rtol=1E-12):
                    log.warning("Sanity Fail in restore_LCF!: r_0 is bad / norm failure")
                    log.warning("r_0 = %s", self.r[0].squeeze().real)
                
                for n in range(1, self.N + 1):
                    l = tm.eps_l_noop(self.l[n - 1], self.A[n], self.A[n])
                    if not sp.allclose(l, self.l[n], atol=1E-11, rtol=1E-11):
                        log.warning("Sanity Fail in restore_LCF!: l_%u is bad (off by %g)", n, la.norm(l - self.l[n]))
        elif update_r:
            self.calc_r()
                    
    
    def auto_truncate(self, update=True, zero_tol=None, return_update_data=False):
        """Automatically reduces the bond-dimension in case of rank-deficiency.
        
        Canonical form is required. Always perform self.restore_CF() first!
        
        Parameters
        ----------
            update : bool (True)
                Whether to call self.update() after truncation.
            zero_tol : float
                Tolerance for interpretation of values as zero.
            return_update_data : bool
                Whether to return additional data needed to perform a minimal update.
        Returns
        -------
            truncated : bool
                Whether truncation was performed (if return_update_data == False).
            data : stuff
                Additional data needed by self._update_after_truncate() (if return_update_data == True).
        """
        if zero_tol is None:
            zero_tol = self.zero_tol
        
        new_D = self.D.copy()
        
        if self.canonical_form == 'right':
            for n in range(1, self.N + 1):
                try:
                    ldiag = self.l[n].diag
                except AttributeError:
                    ldiag = self.l[n].diagonal()
                
                new_D[n] = sp.count_nonzero(abs(ldiag) > zero_tol)
        else:
            for n in range(1, self.N + 1):
                try:
                    rdiag = self.r[n].diag
                except AttributeError:
                    rdiag = self.r[n].diagonal()
                
                new_D[n] = sp.count_nonzero(abs(rdiag) > zero_tol)
        
        if not sp.all(new_D == self.D):
            data = self.truncate(new_D, update=update, return_update_data=return_update_data)
        
            if update:
                self.update()
            
            if return_update_data:
                return data
            else:
                return True
        else:
            return False
            
        
    def truncate(self, new_D, update=True, return_update_data=False):
        """Reduces the bond-dimensions by truncating the least-significant Schmidt vectors.
        
        The parameters must be in canonical form to ensure that
        the discarded parameters correspond to the smallest Schmidt 
        coefficients. Always perform self.restore_RCF() first!
        
        Each bond-dimension can either be reduced or left unchanged.
        
        The resulting parameters self.A will not generally have canonical 
        form after truncation.
        
        Parameters
        ----------
        new_D : list or ndarray of int
            The new bond-dimensions in a vector of length N + 1.
        update : bool (True)
            Whether to call self.update() after truncation (turn off if you plan to do it yourself).
        return_update_data : bool
                Whether to return additional data needed to perform a minimal update.
        Returns
        -------
            data : stuff
                Additional data needed by self._update_after_truncate() (if return_update_data == True).
        """
        new_D = sp.array(new_D)
        assert new_D.shape == self.D.shape, "new_D must have same shape as self.D"
        assert sp.all(new_D <= self.D), "new bond-dimensions must be less than or equal to current dimensions"
    
        last_trunc = sp.argwhere(self.D - new_D).max()
        first_trunc = sp.argwhere(self.D - new_D).min()
        
        tmp_A = self.A
        old_l = self.l
        old_r = self.r
        
        self.D = new_D
        self._init_arrays()

        for n in range(1, self.N + 1):
            self.A[n][:] = tmp_A[n][:, -self.D[n - 1]:, -self.D[n]:]
        
        if update:
            self.update()
        
        if return_update_data:    
            return last_trunc, old_l, first_trunc, old_r
            
    def _update_after_truncate(self, n_last_trunc, old_l, n_first_trunc, old_r):
        if self.canonical_form == 'right':
            self.r[0][0, 0] = 1
            
            for n in range(1, self.N):
                self.l[n] = m.simple_diag_matrix(old_l[n].diag[-self.D[n]:], dtype=self.typ)
                
            self.l[self.N][0, 0] = 1
            
            for n in range(self.N - 1, n_last_trunc - 1, - 1):
                self.r[n] = m.eyemat(self.D[n], dtype=self.typ)
                
            self.calc_r(n_high=n_last_trunc - 1)
        else:
            self.l[0][0, 0] = 1
            
            for n in range(1, self.N):
                self.r[n] = m.simple_diag_matrix(old_r[n].diag[-self.D[n]:], dtype=self.typ)
                
            self.r[0][0, 0] = 1
            
            for n in range(1, n_first_trunc):
                self.l[n] = m.eyemat(self.D[n], dtype=self.typ)
                
            self.calc_l(n_low=n_first_trunc)
            
        self.simple_renorm()
        
    
    def check_RCF(self):
        """Tests for right canonical form.
        
        Uses the criteria listed in sub-section 3.1, theorem 1 of arXiv:quant-ph/0608197v2.
        
        This is a consistency check mainly intended for debugging purposes.
        
        FIXME: The tolerances appear to be too tight!
        
        Returns
        -------
        (rnsOK, ls_trOK, ls_pos, ls_diag, normOK) : tuple of bool
            rnsOK: Right orthonormalization is fullfilled (self.r[n] = eye)
            ls_trOK: all self.l[n] have trace 1
            ls_pos: all self.l[n] are positive-definite
            ls_diag: all self.l[n] are diagonal
            normOK: the state it normalized
        """
        rnsOK = True
        ls_trOK = True
        ls_herm = True
        ls_pos = True
        ls_diag = True
        
        for n in range(1, self.N + 1):
            rnsOK = rnsOK and sp.allclose(self.r[n], sp.eye(self.r[n].shape[0]), atol=self.eps*2, rtol=0)
            ls_herm = ls_herm and sp.allclose(self.l[n] - m.H(self.l[n]), 0, atol=self.eps*2)
            ls_trOK = ls_trOK and sp.allclose(sp.trace(self.l[n]), 1, atol=self.eps*1000, rtol=0)
            ls_pos = ls_pos and all(la.eigvalsh(self.l[n]) > 0)
            ls_diag = ls_diag and sp.allclose(self.l[n], sp.diag(self.l[n].diagonal()))
        
        normOK = sp.allclose(self.l[self.N], 1., atol=self.eps*1000, rtol=0)
        
        return (rnsOK, ls_trOK, ls_pos, ls_diag, normOK)
        
        
    def schmidt_sq(self, n):
        """Returns the squared Schmidt coefficients for a left-right parition.
        
        The chain can be split into two parts between any two sites.
        This returns the sqaured coefficients of the corresponding Schmidt
        decomposition, which are equal to the (non-zero) 
        eigenvalues of the corresponding reduced density matrix.
        
        Parameters
        ----------
        n : int
            Site for split.
            
        Returns
        -------
        lam : sequence of float (if ret_schmidt_sq==True)
            The squared Schmidt coefficients.
        """
        lr = self.l[n].dot(self.r[n])
        try: 
            lam = lr.diag
        except AttributeError: #Assume we are not in canonical form.
            lam = la.eigvals(lr)
        return lam
                        
    def entropy(self, n, ret_schmidt_sq=False):
        """Returns the von Neumann entropy of part of the system under a left-right split.
        
        The chain can be split into two parts between any two sites.
        This function returns the corresponding von Neumann entropy, which
        is a measure of the entanglement between the two parts.
        
        The parameter n specifies that the splitting should be done between 
        sites n and n + 1.
        
        Parameters
        ----------
        k : int
            Site offset for split.
        ret_schmidt_sq : bool
            Whether to also return the squared Schmidt coefficients.
            
        Returns
        -------
        S : float
            The half-chain entropy.
        lam : sequence of float (if ret_schmidt_sq==True)
            The squared Schmidt coefficients.
        """
        lam = self.schmidt_sq(n)
        flt = lam.nonzero()
        S = -sp.sum(lam[flt] * sp.log2(lam[flt])).real
            
        if ret_schmidt_sq:
            return S, lam
        else:
            return S
    
    def expect_1s(self, op, n):
        """Computes the expectation value of a single-site operator.
        
        The operator should be a q[n] x q[n] matrix or generating function 
        such that op[s, t] or op(s, t) equals <s|op|t>.
        
        The state must be up-to-date -- see self.update()!
        
        Parameters
        ----------
        op : ndarray or callable
            The operator.
        n : int
            The site number (1 <= n <= N).
            
        Returns
        -------
        expval : floating point number
            The expectation value (data type may be complex)
        """        
        if callable(op):
            op = sp.vectorize(op, otypes=[sp.complex128])
            op = sp.fromfunction(op, (self.q[n], self.q[n]))
            
        res = tm.eps_r_op_1s(self.r[n], self.A[n], self.A[n], op)
        return  m.adot(self.l[n - 1], res)
        
    def expect_2s(self, op, n, AA=None):
        """Computes the expectation value of a nearest-neighbour two-site operator.
        
        The operator should be a q[n] x q[n + 1] x q[n] x q[n + 1] array 
        such that op[s, t, u, v] = <st|op|uv> or a function of the form 
        op(s, t, u, v) = <st|op|uv>.
        
        The state must be up-to-date -- see self.update()!
        
        Parameters
        ----------
        op : ndarray or callable
            The operator array or function.
        n : int
            The leftmost site number (operator acts on n, n + 1).
            
        Returns
        -------
        expval : floating point number
            The expectation value (data type may be complex)
        """
        A = self.A[n]
        Ap1 = self.A[n + 1]
        if AA is None:
            AA = tm.calc_AA(A, Ap1)
        
        if callable(op):
            op = sp.vectorize(op, otypes=[sp.complex128])
            op = sp.fromfunction(op, (A.shape[0], Ap1.shape[0], A.shape[0], Ap1.shape[0]))
            
        C = tm.calc_C_mat_op_AA(op, AA)
        res = tm.eps_r_op_2s_C12_AA34(self.r[n + 1], C, AA)
        return m.adot(self.l[n - 1], res)

    def expect_3s(self, op, n, AAA=None):
        """Computes the expectation value of a nearest-neighbour three-site operator.

        The operator should be a q[n] x q[n + 1] x q[n + 2] x q[n] x
        q[n + 1] x q[n + 2] array such that op[s, t, u, v, w, x] =
        <stu|op|vwx> or a function of the form op(s, t, u, v, w, x) =
        <stu|op|vwx>.

        The state must be up-to-date -- see self.update()!

        Parameters
        ----------
        op : ndarray or callable
            The operator array or function.
        n : int
            The leftmost site number (operator acts on n, n + 1, n + 2).

        Returns
        -------
        expval : floating point number
            The expectation value (data type may be complex)
        """
        A = self.A[n]
        Ap1 = self.A[n + 1]
        Ap2 = self.A[n + 2]
        if AAA is None:
            AAA = tm.calc_AAA(A, Ap1, Ap2)

        if callable(op):
            op = sp.vectorize(op, otypes=[sp.complex128])
            op = sp.fromfunction(op, (A.shape[0], Ap1.shape[0], Ap2.shape[0],
                                      A.shape[0], Ap1.shape[0], Ap2.shape[0]))

        C = tm.calc_C_3s_mat_op_AAA(op, AAA)
        res = tm.eps_r_op_3s_C123_AAA456(self.r[n + 2], C, AAA)
        return m.adot(self.l[n - 1], res)

    def expect_1s_1s(self, op1, op2, n1, n2, return_intermediates=False):
        """Computes the expectation value of two single site operators acting 
        on two different sites.
        
        The result is < op1 op2 >.
        
        See expect_1s().
        
        Requires n1 < n2.
        
        The state must be up-to-date -- see self.update()!
        
        Parameters
        ----------
        op1 : ndarray or callable
            The first operator, acting on the first site.
        op2 : ndarray or callable
            The second operator, acting on the second site.
        n1 : int
            The site number of the first site.
        n2 : int
            The site number of the second site (must be > n1).
        return_intermediates : bool
            Whether to return results for intermediate sites.
            
        Returns
        -------
        expval : complex128 or sequence of complex128
            The expectation value (data type may be complex), or values if
            return_intermediates == True.
        """        
        if callable(op1):
            op1 = sp.vectorize(op1, otypes=[sp.complex128])
            op1 = sp.fromfunction(op1, (self.q[n1], self.q[n1]))
        
        if callable(op2):
            op2 = sp.vectorize(op2, otypes=[sp.complex128])
            op2 = sp.fromfunction(op2, (self.q[n2], self.q[n2])) 
        
        d = n2 - n1
        
        res = sp.zeros((d + 1), dtype=sp.complex128)
        lj = tm.eps_l_op_1s(self.l[n1 - 1], self.A[n1], self.A[n1], op1)
        
        if return_intermediates:
            res[0] = self.expect_1s(op1.dot(op1), n1)

        for j in range(1, d + 1):
            if return_intermediates or j == d:
                lj_op = tm.eps_l_op_1s(lj, self.A[n1 + j], self.A[n1 + j], op2)
                res[j] = m.adot(lj_op, self.r[n1 + j])
                
            if j < d:
                lj = tm.eps_l_noop(lj, self.A[n1 + j], self.A[n1 + j])
                
        if return_intermediates:
            return res
        else:
            return res[-1]
            
    def correlation_1s_1s(self, op1, op2, n1, d, return_exvals=False):
        """Computes a correlation function of two 1 site operators.
        
        The result is < op1_k op2_k+j > - <op1_k> * <op2_k+j> 
        with the operators acting on sites k and k + j, with j running from
        0 to d.
        
        Optionally returns the corresponding expectation values <op1_k+j> and 
        <op2_k+j>.
        
        Parameters
        ----------
        op1 : ndarray or callable
            The first operator, acting on the first site.
        op2 : ndarray or callable
            The second operator, acting on the second site.
        n1  : int
            Site to begin from.
        d : int
            The distance (number of sites) between the two sites acted on non-trivially.
        return_exvals : bool
            Whether to return expectation values for op1 and op2 for all sites.
            
        Returns
        -------
        ccf : sequence of complex128
            The correlation function across d + 1 sites (including site k).
        ex1 : sequence of complex128
            Expectation values of op1 for each site. Only if return_exvals == True.
        ex2 : sequence of complex128
            See ex1.
        """
        ex1 = sp.zeros((d + 1), dtype=sp.complex128)
        for j in range(d + 1):
            ex1[j] = self.expect_1s(op1, n1 + j)
            
        if op1 is op2:
            ex2 = ex1
        else:
            ex2 = sp.zeros((d + 1), dtype=sp.complex128)
            for j in range(d + 1):
                ex2[j] = self.expect_1s(op2, n1 + j)
            
        cf = self.expect_1s_1s(op1, op2, n1, n1 + d, return_intermediates=True)
            
        ccf = sp.zeros((d + 1), dtype=sp.complex128)
        for j in range(d + 1):
            ccf[j] = cf[j] - ex1[0] * ex2[j]
            
        if return_exvals:
            ex1_ = sp.tile(ex1, [(d + 1) // d + 1])[:d + 1]
            ex2_ = sp.tile(ex1, [(d + 1) // d + 1])[:d + 1]
            return ccf, ex1_, ex2_
        else:
            return ccf
        
    def expect_string_1s(self, op, n, d):
        """Calculates the expectation values of finite strings
        with lengths 1 to d, starting at position n.
        """
        if callable(op):
            op = sp.vectorize(op, otypes=[sp.complex128])
            op = sp.fromfunction(op, (self.q, self.q))
        
        res = sp.zeros((d), dtype=self.A[1].dtype)
        x = self.l[n - 1]
        for j in range(n, n + d + 1):
            Aop = sp.tensordot(op, self.A[j], axes=([1],[0]))
            x = tm.eps_l_noop(x, self.A[j], Aop)
            res[j - n - 1] = m.adot(x, self.r[j])
        
        return res

    def expect_MPO(self, MPO, n1):
        """Calculates the expectation value of a Matrix Product Operator.

        The MPO must be supplied as a list of ndarrays, one entry for each 
        physical site on which the MPO acts.
        
        Each ndarray must have shape 
        [mL,mR,pbra,pket], where mL, mR are the left and right MPO indices
        and pbra and pket are the physical indices such that, for example, 
        MPO[n][0,0,:,:]|psi> is a physical operator acting on a state.

        NOTE: All entries of the MPO list are used, so for n1 = 1, MPO[0]
              is applied to site 1 of the MPS.

        Parameters
        ----------
        MPO : list of ndarrays
            The MPO.
        n1 : integer
            The first site on which the MPO is to act.
        Returns
        -------
        exval : scalar
            The expectation value.
        """
        nsites = len(MPO)
        n_end = n1 + nsites - 1

        #reshape to MPO vector
        x = sp.reshape(self.r[n_end], (1,self.D[n_end],self.D[n_end]))
        for j in range(n_end,n1-1,-1):
            x = tm.eps_r_op_MPO(x, self.A[j], self.A[j], MPO[j-n1])

        x = sp.reshape(x, (self.D[n1-1], self.D[n1-1]))

        exval = m.adot(self.l[n1 - 1], x)
        return exval

    def density_1s(self, n):
        """Returns a reduced density matrix for a single site.
        
        The site number basis is used: rho[s, t] 
        with 0 <= s, t < q[n].
        
        The state must be up-to-date -- see self.update()!
        
        Parameters
        ----------
        n1 : int
            The site number.
            
        Returns
        -------
        rho : ndarray
            Reduced density matrix in the number basis.
        """
        rho = sp.empty((self.q[n], self.q[n]), dtype=sp.complex128)
                    
        r_n = self.r[n]
        r_nm1 = sp.empty_like(self.r[n - 1])
        for s in range(self.q[n]):
            for t in range(self.q[n]):
                r_nm1 = m.mmul(self.A[n][t], r_n, m.H(self.A[n][s]))                
                rho[s, t] = m.adot(self.l[n - 1], r_nm1)
        return rho
        
    def density_2s(self, n1, n2):
        """Returns a reduced density matrix for a pair of (seperated) sites.
        
        The site number basis is used: rho[s * q[n1] + u, t * q[n1] + v]
        with 0 <= s, t < q[n1] and 0 <= u, v < q[n2].
        
        The state must be up-to-date -- see self.update()!
        
        Parameters
        ----------
        n1 : int
            The site number of the first site.
        n2 : int
            The site number of the second site (must be > n1).
            
        Returns
        -------
        rho : ndarray
            Reduced density matrix in the number basis.
        """
        rho = sp.empty((self.q[n1] * self.q[n2], self.q[n1] * self.q[n2]), dtype=sp.complex128)
        
        for s2 in range(self.q[n2]):
            for t2 in range(self.q[n2]):
                r_n2 = m.mmul(self.A[n2][t2], self.r[n2], m.H(self.A[n2][s2]))
                
                r_n = r_n2
                for n in reversed(range(n1 + 1, n2)):
                    r_n = tm.eps_r_noop(r_n, self.A[n], self.A[n])        
                    
                for s1 in range(self.q[n1]):
                    for t1 in range(self.q[n1]):
                        r_n1 = m.mmul(self.A[n1][t1], r_n, m.H(self.A[n1][s1]))
                        tmp = m.adot(self.l[n1 - 1], r_n1)
                        rho[s1 * self.q[n1] + s2, t1 * self.q[n1] + t2] = tmp
        return rho
    
    def apply_op_1s(self, op, n, do_update=True):
        """Applies a single-site operator to a single site.
        
        By default, this performs self.update(), which also restores
        state normalization.        
        
        Parameters
        ----------
        op : ndarray or callable
            The single-site operator. See self.expect_1s().
        n: int
            The site to apply the operator to.
        do_update : bool
            Whether to update after applying the operator.
        """
        if callable(op):
            op = sp.vectorize(op, otypes=[sp.complex128])
            op = sp.fromfunction(op, (self.q[n], self.q[n]))
            
        newAn = sp.zeros_like(self.A[n])
        
        for s in range(self.q[n]):
            for t in range(self.q[n]):
                newAn[s] += self.A[n][t] * op[s, t]
                
        self.A[n] = newAn
        
        if do_update:
            self.update()

    def apply_op_MPO(self, MPO, n1, do_update=True):
        """Applies a Matrix Product Operator to the state.
        
        Note that this will increase the bond dimension where the MPO
        is applied.

        By default, this performs self.update(), which also restores
        state normalization.        
        
        Parameters
        ----------
        op : list of ndarray
            The MPO. See self.expect_MPO().
        n1: int
            The first site to apply the operator to.
        do_update : bool
            Whether to update after applying the operator.
        """
        A = copy.copy(self.A)
        n = n1
        for j in range(len(MPO)):
            A[n] = tm.apply_MPO_local(MPO[j], self.A[n])
            n += 1
        
        self.set_state_from_tensors(A, do_update=do_update)
    
    def save_state(self, file):
        """Saves the parameter tensors self.A to a file. 
        
        Uses numpy binary format.
        
        Parameters
        ----------
        file ; path or file
            The file to save the state into.
        """
        sp.save(file, self.A)
        
    def load_state(self, file, do_update=True):
        """Loads the parameter tensors self.A from a file.

        The saved state must contain the right number of tensors with
        the correct shape corresponding to self.N and self.q.
        self.D will be recovered from the saved state.

        Parameters
        ----------
        file ; path or file
            The file to load the state from.
        """
        tmp_A = sp.load(file)

        self.D[0] = 1
        for n in range(self.N):
            self.D[n + 1] = tmp_A[n + 1].shape[2]
        self._init_arrays()
        self.A = tmp_A
        
        if do_update:
            self.update()
