# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 17:29:27 2011

@author: Ashley Milsted

"""
import numpy as np
import scipy as sp
import scipy.linalg as la
import scipy.sparse.linalg as las
from . import tdvp_common as tm
from . import matmul as m
import math as ma
import logging

log = logging.getLogger(__name__)

class EOp:
    def __init__(self, A1, A2, left):
        """Creates a new LinearOperator interface to the superoperator E.
        
        This is a wrapper to be used with SciPy's sparse linear algebra routines.
        
        Parameters
        ----------
        A1 : ndarray
            Ket parameter tensor. 
        A2 : ndarray
            Bra parameter tensor.
        left : bool
            Whether to multiply with a vector to the left (or to the right).
        """
        self.A1 = A1
        self.A2 = A2
        
        self.D1 = A1[0].shape[1]
        self.D2 = A2[0].shape[1]
        
        self.shape = (self.D1 * self.D2, self.D1 * self.D2)
        
        self.dtype = np.dtype(A1[0].dtype)
        
        self.calls = 0
        
        self.left = left
        
        if left:
            self.eps = tm.eps_l_noop_inplace
        else:
            self.eps = tm.eps_r_noop_inplace
            
        self.out1 = sp.empty(((self.D1, self.D2)), dtype=self.dtype)
        self.out2 = sp.empty(((self.D1, self.D2)), dtype=self.dtype)
    
    def matvec(self, v):
        """Matrix-vector multiplication. 
        Result = Ev or vE (if self.left == True).
        """
        x = v.reshape((self.D1, self.D2))
        
        res = self.out1
        out = self.out2
        res[:] = x #we should *not* modify x, since it belongs to ARPACK's workspace
        
        if self.left:           
            for n in range(len(self.A1)):
                out = self.eps(res, self.A1[n], self.A2[n], out)
                tmp = res
                res = out
                out = tmp
        else:
            for n in range(len(self.A1) - 1, -1, -1):
                out = self.eps(res, self.A1[n], self.A2[n], out)
                tmp = res
                res = out
                out = tmp

        self.calls += 1
        
        #if res is F-ordered, this will make a copy
        return res.ravel() #the return value gets copied into ARPACK's workspace.
        
class EvoMPSNoConvergence(Exception):
    pass
        
class EvoMPSNormError(Exception):
    pass

class EvoMPS_MPS_Uniform(object):   
        
    def __init__(self, D, q, L=1, dtype=None, do_update=True):
        """Creates a new EvoMPS_MPS_Uniform object.
        
        This class implements basic operations on a uniform 
        (translation-invariant) MPS in the thermodynamic limit.
        
        self.A is the parameter tensor and has shape (q, D, D).
        
        Parameters
        ----------
        D : int
            The bond-dimension.
        q : int
            The single-site Hilbert-space dimension.
        L : int
            Block length.
        dtype : numpy-compatible dtype
            The data-type to be used. The default is double-precision complex.
        """
        self.odr = 'C' 
        
        if dtype is None:
            self.typ = np.complex128
        
        self.itr_rtol = 1E-13
        self.itr_atol = 1E-14
        
        self.zero_tol = sp.finfo(self.typ).resolution
        """Tolerance for detecting zeros. This is used when (pseudo-) inverting 
           l and r."""
        
        self.pow_itr_max = 2000
        """Maximum number of iterations to use in the power-iteration algorithm 
           for finding l and r."""

        self.ev_brute = False
        """Use dense methods to find l and r. For debugging. Scales as D**6.
           Overrides ev_use_arpack."""
           
        self.ev_use_arpack = True
        """Whether to use ARPACK (implicitly restarted Arnoldi iteration) to 
           find l and r. If False, use simple power-iteration instead. In most cases,
           ARPACK should converge on the solution faster."""
           
        self.ev_arpack_nev = 1
        """The number of eigenvalues to find when calculating l and r using ARPACK. 
           If the spectrum is approximately degenerate, this may need to be increased."""
           
        self.ev_arpack_ncv = max(20, 2 * self.ev_arpack_nev + 1)
        """The number of intermediate vectors stored during Arnoldi iteration.
           If this parameter is too small, ARPACK sometimes converges to
           the wrong solution.
           See the documentation for scipy.sparse.linalg.eig()."""
           
        self.ev_arpack_CUDA = False
        """Whether to use CUDA to implement the transfer matrix when calculating
           l and r. Requires pycuda and scikits.cuda and a working CUDA setup
           supporting double-precision arithmetic."""
           
        self.CUDA_batch_maxD = 128
        """Maximum bond dimension for use of batched CUBLAS GEMM variants.
        Above this bond dimension, CUDA streams will be used.
        """
                
        self.symm_gauge = True
        """Whether to use symmetric canonical form or (if False) right canonical form."""
        
        self.sanity_checks = False
        """Whether to perform additional (potentially costly) sanity checks."""

        self.check_fac = 50
        self.eps = np.finfo(self.typ).eps
        
        self.userdata = None        
        """This variable is saved (pickled) with the state. 
           For storing arbitrary user data."""
        
        #Initialize some more instance attributes.
        self.itr_l = 0
        """Contains the number of eigenvalue solver iterations needed to find l."""
        self.itr_r = 0
        """Contains the number of eigenvalue solver iterations needed to find r."""
                
        self._init_arrays(D, q, L)
                    
        self.randomize(do_update=do_update)

    def randomize(self, do_update=True):
        """Randomizes the parameter tensors self.A.
        
        Parameters
        ----------
        do_update : bool (True)
            Whether to perform self.update() after randomizing.
        """
        for Ak in self.A:
            m.randomize_cmplx(Ak)
            Ak /= la.norm(Ak)
        
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
        for Ak in self.A:
            norm = la.norm(Ak)
            f = fac * (norm / (self.q * self.D**2))
            R = np.empty_like(Ak)
            m.randomize_cmplx(R, -f / 2.0, f / 2.0)
            
            Ak += R
        
        if do_update:
            self.update()
            
    def convert_to_TI_blocked(self, do_update=True):
        if self.L == 1:
            return
            
        L = self.L
        q = self.q
        newq = q**L
        if newq > 1024:
            print("Warning: Local dimension will be", newq)
        
        newA = sp.zeros((newq, self.D, self.D), dtype=self.typ)
        for s in range(newq):
            newA[s] = sp.eye(self.D)
            for k in range(L):
                t = (s // q**k) % q
                newA[s] = self.A[L - k - 1][t].dot(newA[s])
                
        newlL = sp.asarray(self.l[-1])
        newrL = sp.asarray(self.r[-1])
                
        self._init_arrays(self.D, newq, 1)
        
        self.A[0] = newA
        self.lL_before_CF = newlL
        self.rL_before_CF = newrL
        
        if do_update:
            self.update()
    
    def _init_arrays(self, D, q, L):
        self.D = D
        self.q = q
        self.L = L
        
        self.A = []
        self.AA = []
        self.l = []
        self.r = []
        for m in range(L):
            self.A.append(np.zeros((q, D, D), dtype=self.typ, order=self.odr))
            self.AA.append(np.zeros((q, q, D, D), dtype=self.typ, order=self.odr))
            self.l.append(np.ones_like(self.A[0][0]))
            self.r.append(np.ones_like(self.A[0][0]))
            
        self.lL_before_CF = self.l[-1]
        self.rL_before_CF = self.r[-1]
            
        self.conv_l = True
        self.conv_r = True
        
        self.tmp = np.zeros_like(self.A[0][0])
        
    def _get_dense_transfer_op_block(self):
        E = m.eyemat(self.D**2)
        for k in range(self.L):
            A = self.A[k]
            Ek = sp.zeros((A.shape[1]**2, A.shape[2]**2), dtype=A.dtype)
            for s in range(A.shape[0]):
                Ek += sp.kron(A[s], A[s].conj())
            E = E.dot(Ek)
        return E
        
    def _calc_lr_brute(self):
        Eblock = self._get_dense_transfer_op_block()
            
        ev, eVL, eVR = la.eig(Eblock, left=True, right=True)
        
        i = np.argmax(abs(ev))
        
        self.A[0] *= 1 / sp.sqrt(ev[i])        
        
        lL = eVL[:,i].reshape((self.D, self.D))
        rL = eVR[:,i].reshape((self.D, self.D))
        
        norm = m.adot(lL, rL)
        lL *= 1 / sp.sqrt(norm)
        rL *= 1 / sp.sqrt(norm)

        return lL, rL        
        
    def _get_EOP(self, A1, A2, left):
        if self.ev_arpack_CUDA:
            from . import cuda_alternatives as tcu
            opE = tcu.EOp_CUDA(A1, A2, left, use_batch=(self.D <= self.CUDA_batch_maxD))
        else:
            opE = EOp(A1, A2, left)
            
        return opE
    
    def _calc_lr_ARPACK(self, x, tmp, calc_l=False, A1=None, A2=None, rescale=True,
                        tol=1E-14, ncv=None, nev=1, max_retries=4, which='LM'):
        if A1 is None:
            A1 = self.A
        if A2 is None:
            A2 = self.A
            
        if ncv is None:
            ncv = max(20, 2 * nev + 1)
            
        symmetric = sp.all([A1[j] is A2[j] for j in range(len(A1))])
                        
        n = x.size #we will scale x so that stuff doesn't get too small
        
        opE = self._get_EOP(A1, A2, calc_l)
        x *= n / la.norm(x.ravel())
        v0 = x.ravel()
        for i in range(max_retries):
            if i > 0:
                log.warning("_calc_lr_ARPACK: Retry #%u (%s)", i, "l" if calc_l else "r")
            try:
                ev, eV = las.eigs(opE, which=which, k=nev, v0=v0, tol=tol, ncv=ncv)
                conv = True
                ind = abs(ev).argmax()
                ev = np.real_if_close(ev[ind])
                ev = np.asscalar(ev)
                eV = eV[:, ind]
                if abs(ev) < 1E-12:
                    raise ValueError("Largest eigenvalue too small!")
                if symmetric and np.imag(ev) != 0:
                    raise ValueError("Largest eigenvalue is not real (%g)! (ncv too small?)" % np.imag(ev))
                break
            except (las.ArpackNoConvergence, ValueError) as e:
                log.warning("_calc_lr_ARPACK(nev=%u,ncv=%u): %s Try %u! (%s)", nev, ncv, e, i, "l" if calc_l else "r")
                v0 = None
                nev += 1
                ncv += 5
                
        if i == max_retries - 1:
            log.error("_calc_lr_ARPACK(nev=%u,ncv=%u): Failed to converge! (%s)", nev, ncv, "l" if calc_l else "r")
            raise EvoMPSNoConvergence("_calc_lr_ARPACK: Failed to converge!")
        
        #remove any additional phase factor on the eigenvector
        eVmean = eV.mean()
        eV *= sp.sqrt(sp.conj(eVmean) / eVmean)
        
        if eV.mean() < 0:
            eV *= -1

        eV = eV.reshape(self.D, self.D)
        
        x[:] = eV
                    
        if rescale: 
            fac = (1 / sp.sqrt(ev))**(1. / len(A1))
            for A in A1:
                A *= fac
                
            if self.sanity_checks:
                if not symmetric:
                    log.warning("Sanity check failed: Re-scaling with A1 <> A2!")
                tmp = opE.matvec(x.ravel())
                ev = tmp.mean() / x.mean()
                if not abs(ev - 1) < tol:
                    log.warning("Sanity check failed: Largest ev after re-scale = %s", ev)
        
        return x, conv, opE.calls, nev, ncv
        
    def _calc_E_largest_eigenvalues(self, k=0, tol=1E-6, nev=2, ncv=None, 
                                    left=False, max_retries=3, 
                                    return_eigenvectors=False):
        if self.D == 1:
            return sp.array([1. + 0.j])
        elif self.D <= 3:
            E = self._get_dense_transfer_op_block()
            return la.eigvals(E)
        
        A = list(self.A)
        A = A[k:] + A[:k]
        
        opE = self._get_EOP(A, A, left)
        
        if left:
            v0 = np.asarray(self.l[(k - 1) % self.L])
        else:
            v0 = np.asarray(self.r[(k - 1) % self.L])
        
        for i in range(max_retries):
            try:
                res = las.eigs(opE, which='LM', k=nev, v0=v0.ravel(), tol=tol, 
                              ncv=ncv, return_eigenvectors=return_eigenvectors)
                break
            except las.ArpackNoConvergence:
                log.warning("_calc_E_largest_eigenvalues: Retry %u!", i)
                nev += 1
                ncv = nev * 3
        
        if i == max_retries - 1:
            log.error("_calc_E_largest_eigenvalues: Failed to converge!")
            raise EvoMPSNoConvergence("_calc_E_largest_eigenvalues failed!")
                          
        return res
        
    def calc_E_gap(self, tol=1E-6, nev=2, ncv=None):
        """
        Calculates the spectral gap of E by calculating the second-largest eigenvalue.
        
        The result is the difference between the largest eigenvalue and the 
        magnitude of the second-largest divided by the largest 
        (which should be equal to 1).
        
        This is related to the correlation length. See self.correlation_length().
        
        Parameters
        ----------
        tol : float
            Tolerance for second-largest eigenvalue.
        nev : int
            Number of eigenvalues to calculate.
        ncv : int
            Number of Arnoldii basis vectors to store.
        """
        ev = self._calc_E_largest_eigenvalues(tol=tol, nev=nev, ncv=ncv)
        
        if len(ev) == 1:
            return sp.NaN
        
        ev = abs(ev)
        ev.sort()
        ev1_mag = ev[-1]
        ev2_mag = ev[-2]
        
        return ((ev1_mag - ev2_mag) / ev1_mag)
        
    def correlation_length(self, tol=1E-12, nev=3, ncv=None):
        """
        Calculates the correlation length in units of the lattice spacing.
        
        The correlation length is equal to the inverse of the natural logarithm
        of the maginitude of the second-largest eigenvalue of the transfer 
        (or super) operator E.
        
        Parameters
        ----------
        tol : float
            Tolerance for second-largest eigenvalue.
        nev : int
            Number of eigenvalues to calculate.
        ncv : int
            Number of Arnoldi basis vectors to store.
        """
        if self.D == 1:
            return 0.
        
        if ncv is None:
            ncv = max(20, 2 * nev + 1)
        
        ev = self._calc_E_largest_eigenvalues(tol=tol, nev=nev, ncv=ncv)
        log.debug("Eigenvalues of the transfer operator: %s", ev)
        
        #We only require the absolute values, and sort() does not handle
        #complex values nicely (it sorts by real part).
        ev = abs(ev)
        
        ev.sort()
        log.debug("Eigenvalue magnitudes of the transfer operator: %s", ev)
                          
        ev1 = ev[-1]
        ev = ev[:-1]
        
        if abs(ev1 - 1) > tol:
            log.warning("Warning: Largest eigenvalue != 1")

        while True:
            if ev.shape[0] > 1 and (ev1 - ev[-1]) < tol:
                ev = ev[:-1]
                log.warning("Warning: Degenerate largest eigenvalue in E spectrum!")
            else:
                break

        if ev.shape[0] == 0:
            log.warning("Warning: No eigenvalues detected with magnitude significantly different to largest.")
            return sp.NaN
        
        return -self.L / sp.log(ev[-1])
        
    def _calc_lr(self, x, tmp, calc_l=False, A1=None, A2=None, rescale=True,
                 max_itr=1000, rtol=1E-14, atol=1E-14):
        """Power iteration to obtain eigenvector corresponding to largest
           eigenvalue.
        """        
        if A1 is None:
            A1 = self.A
        if A2 is None:
            A2 = self.A
            
        symmetric = sp.all([A1[j] is A2[j] for j in range(len(A1))])
                        
        n = x.size #we will scale x so that stuff doesn't get too small
        
        opE = self._get_EOP(A1, A2, calc_l)
        
        x = x.ravel()
        tmp = tmp.ravel()

        x *= n / la.norm(x)
        tmp[:] = x
        for i in range(max_itr):
            x[:] = tmp
            tmp[:] = opE.matvec(x)
                    
            ev_mag = la.norm(tmp) / n
            ev = (tmp.mean() / x.mean()).real
            tmp *= (1 / ev_mag)
            if la.norm(tmp - x) < atol + rtol * n:
                x[:] = tmp
                break
        
        x = x.reshape((self.D, self.D))
                    
        if rescale: 
            fac = (1 / sp.sqrt(ev))**(1. / len(A1))
            for A in A1:
                A *= fac
                
            if self.sanity_checks:
                if not symmetric:
                    log.warning("Sanity check failed: Re-scaling with A1 <> A2!")
                tmp = opE.matvec(x.ravel())
                ev = tmp.mean() / x.mean()
                if not abs(ev - 1) < atol:
                    log.warning("Sanity check failed: Largest ev after re-scale = %s", ev)
        
        return x, i < max_itr - 1, i
    
    def calc_lr(self):
        """Determines the dominant left and right eigenvectors of the transfer 
        operator E.
        
        Uses an iterative method (e.g. Arnoldi iteration) to determine the
        largest eigenvalue and the correspoinding left and right eigenvectors,
        which are stored as self.l and self.r respectively.
        
        The parameter tensor self.A is rescaled so that the largest eigenvalue
        is equal to 1 (thus normalizing the state).
        
        The largest eigenvalue is assumed to be non-degenerate.
        
        """
        tmp = np.empty_like(self.tmp)
        
        #Make sure...
        self.lL_before_CF = np.asarray(self.lL_before_CF)
        self.rL_before_CF = np.asarray(self.rL_before_CF)
        
        for retries in range(2):
            if self.ev_brute or self.D <= 3:
                self.l[-1], self.r[-1] = self._calc_lr_brute()
            else:
                if self.ev_use_arpack:
                    self.l[-1], self.conv_l, self.itr_l, nev, ncv = self._calc_lr_ARPACK(self.lL_before_CF, tmp,
                                                                   calc_l=True, rescale=True,
                                                                   tol=self.itr_rtol,
                                                                   nev=self.ev_arpack_nev,
                                                                   ncv=self.ev_arpack_ncv)
                    self.r[-1], self.conv_r, self.itr_r, nev, ncv = self._calc_lr_ARPACK(self.rL_before_CF, tmp, 
                                                                   calc_l=False, which='LR', rescale=False,
                                                                   tol=self.itr_rtol,
                                                                   nev=nev,
                                                                   ncv=ncv)
                else:
                    self.l[-1], self.conv_l, self.itr_l = self._calc_lr(self.lL_before_CF, 
                                                            tmp, 
                                                            calc_l=True,
                                                            max_itr=self.pow_itr_max,
                                                            rtol=self.itr_rtol, 
                                                            atol=self.itr_atol)
                    self.r[-1], self.conv_r, self.itr_r = self._calc_lr(self.rL_before_CF, 
                                                            tmp, 
                                                            calc_l=False,
                                                            max_itr=self.pow_itr_max,
                                                            rtol=self.itr_rtol, 
                                                            atol=self.itr_atol)
                
            norm = m.adot(self.l[-1], self.r[-1]).real
            if abs(norm) < 1E-10:
                log.warning("Left and right eigenvectors are orthogonal! (%g) Retrying with initial-vector reset...", norm)
                self.lL_before_CF.fill(1)
                self.rL_before_CF.fill(1)
            else:
                break
            
        self.lL_before_CF = self.l[-1].copy()
        self.rL_before_CF = self.r[-1].copy()
        
        #normalize eigenvectors:
        self.l[-1] *= 1. / ma.sqrt(norm)
        self.r[-1] *= 1. / ma.sqrt(norm)

        if not self.symm_gauge:
            fac = self.D / np.trace(self.r[-1]).real
            self.l[-1] *= 1 / fac
            self.r[-1] *= fac

        #compute eigenvectors for other block boundaries
        for k in range(len(self.A) - 1, 0, -1):
            self.r[k - 1] = tm.eps_r_noop(self.r[k], self.A[k], self.A[k])
            
        for k in range(0, len(self.A) - 1):
            self.l[k] = tm.eps_l_noop(self.l[k - 1], self.A[k], self.A[k])
            
        if self.sanity_checks:
            for k in range(self.L):
                l = self.l[k]
                for j in sp.arange(k + 1, k + self.L + 1) % self.L:
                    l = tm.eps_l_noop(l, self.A[j], self.A[j])
                if not np.allclose(l, self.l[k],
                rtol=self.itr_rtol*self.check_fac, 
                atol=self.itr_atol*self.check_fac):
                    log.warning("Sanity check failed: l%u bad! Off by: %s", k,
                                la.norm(l - self.l[k]))

                r = self.r[k]
                for j in sp.arange(k, k - self.L, -1) % self.L:
                    r = tm.eps_r_noop(r, self.A[j], self.A[j])
                if not np.allclose(r, self.r[k],
                rtol=self.itr_rtol*self.check_fac,
                atol=self.itr_atol*self.check_fac):
                    log.warning("Sanity check failed: r%u bad! Off by: %s", k,
                                la.norm(r - self.r[k]))
                
                if not np.allclose(self.l[k], m.H(self.l[k]),
                rtol=self.itr_rtol*self.check_fac, 
                atol=self.itr_atol*self.check_fac):
                    log.warning("Sanity check failed: l%u is not hermitian! Off by: %s",
                                k, la.norm(self.l[-k] - m.H(self.l[k])))
    
                if not np.allclose(self.r[k], m.H(self.r[k]),
                rtol=self.itr_rtol*self.check_fac, 
                atol=self.itr_atol*self.check_fac):
                    log.warning("Sanity check failed: r%u is not hermitian! Off by: %s",
                                k, la.norm(self.r[k] - m.H(self.r[k])))
                
                minev = la.eigvalsh(self.l[k]).min()
                if minev <= 0:
                    log.warning("Sanity check failed: l%u is not pos. def.! Min. ev: %s", k, minev)
                    
                minev = la.eigvalsh(self.r[k]).min()
                if minev <= 0:
                    log.warning("Sanity check failed: r%u is not pos. def.! Min. ev: %s", k, minev)
                
                norm = m.adot(self.l[k], self.r[k])
                if not np.allclose(norm, 1.0, atol=1E-13, rtol=0):
                    log.warning("Sanity check failed: Bad norm = %s", norm)
    
    def restore_SCF(self, zero_tol=None, force_evd=True):
        """Restores symmetric canonical form.
        
        In this canonical form, self.l == self.r and are diagonal matrices
        with the Schmidt coefficients corresponding to the half-chain
        decomposition form the diagonal entries.
        """
        if zero_tol is None:
            zero_tol = self.zero_tol
        
        for k in range(self.L):
            try:
                X = tm.herm_fac_with_inv(self.r[k], lower=True, zero_tol=zero_tol,
                                             force_evd=force_evd, calc_inv=False,
                                             sanity_checks=self.sanity_checks, sc_data='Restore_SCF: r%u' % k)
                
                Y = tm.herm_fac_with_inv(self.l[k], lower=False, zero_tol=zero_tol,
                                             force_evd=force_evd, calc_inv=False,
                                             sanity_checks=self.sanity_checks, sc_data='Restore_SCF: l%u' % k)
            except ValueError:
                log.error("restore_SCF: Decomposition of l and r failed!")
                raise ValueError('restore_SCF: Decomposition of l and r failed!')
            
            U, sv, Vh = la.svd(Y.dot(X))

            #s contains the Schmidt coefficients,
            S = m.simple_diag_matrix(sv, dtype=self.typ)
            Srt = S.sqrt()
            
            #Invert the square roots of the Schmidt coefficients.
            nonzeros = np.count_nonzero(S.diag > zero_tol)
            Srti = sp.zeros_like(S.diag, dtype=S.dtype)
            Srti[-nonzeros:] = 1. / Srt.diag[-nonzeros:]
            Srti = m.simple_diag_matrix(Srti, dtype=S.dtype)  
            
            g = Srti.dot(U.conj().T).dot(Y)
            g_i = Srti.dot_left(X.dot(Vh.conj().T))
            
            j = (k + 1) % self.L
            for s in range(self.q):
                self.A[j][s] = g.dot(self.A[j][s])
                self.A[k][s] = self.A[k][s].dot(g_i)
                    
            if self.sanity_checks:
                assert sp.all(sv[::-1] == sp.sort(sv)), "Singular values returned in unexpected order!"
            
                Sfull = np.asarray(S)
                
                if not np.allclose(g.dot(g_i), np.eye(self.D)):
                    log.warning("Sanity check failed! Restore_SCF, bad GT! Off by %s",
                                la.norm(g.dot(g_i) - np.eye(self.D)))
                
                l = m.mmul(m.H(g_i), self.l[k], g_i)
                r = m.mmul(g, self.r[k], m.H(g))
                
                if not np.allclose(Sfull, l):
                    log.warning("Sanity check failed: Restore_SCF, left failed! Off by %s",
                                la.norm(Sfull - l))
                    
                if not np.allclose(Sfull, r):
                    log.warning("Sanity check failed: Restore_SCF, right failed! Off by %s",
                                la.norm(Sfull - r))
                
                l = Sfull
                for j in (sp.arange(k + 1, k + self.L + 1) % self.L):
                    l = tm.eps_l_noop(l, self.A[j], self.A[j])
                r = Sfull
                for j in (sp.arange(k, k - self.L, -1) % self.L):
                    r = tm.eps_r_noop(r, self.A[j], self.A[j])
                
                if not np.allclose(Sfull, l, rtol=self.itr_rtol*self.check_fac, 
                                   atol=self.itr_atol*self.check_fac):
                    log.warning("Sanity check failed: Restore_SCF, left %u bad! Off by %s",
                                k, la.norm(Sfull - l))
                    
                if not np.allclose(Sfull, r, rtol=self.itr_rtol*self.check_fac, 
                                   atol=self.itr_atol*self.check_fac):
                    log.warning("Sanity check failed: Restore_SCF, right %u bad! Off by %s",
                                k, la.norm(Sfull - r))
    
            self.l[k] = S
            self.r[k] = S
    
    def restore_RCF(self, zero_tol=None, diag_l=True, ks=None, ret_g=False):
        """Restores right canonical form.
        
        In this form, self.r = sp.eye(self.D) and self.l is diagonal, with
        the squared Schmidt coefficients corresponding to the half-chain
        decomposition as eigenvalues.
        
        Parameters
        ----------
        zero_tol : float
            Zero-tolerance. Default is the value in self.zero_tol.
        diag_l : bool
            Whether to diagonalize l. Not doing so does not fully restore
            canonical form, but may be useful in certain circumstances.
        ks : list of ints
            Which l[k] and r[k] to bring into canonical form. Default is all.
        ret_g : bool
            Whether to return the gauge-transformation matrices used.
            
        Returns
        -------
        G, G_i : ndarray
            Gauge transformation matrices g and inverses g_i.
        """
        
        if zero_tol is None:
            zero_tol = self.zero_tol
        
        if ks is None:
            ks = list(range(self.L))
        
        G = [None] * self.L
        G_i = [None] * self.L
        
        for k in ks:
            #First get G such that r = eye
            G[k], G_i[k], rank = tm.herm_fac_with_inv(self.r[k], lower=True, zero_tol=zero_tol,
                                                return_rank=True,
                                                sanity_checks=self.sanity_checks,
                                                sc_data='Restore_RCF: r')
    
            self.l[k] = G[k].conj().T.dot(self.l[k].dot(G[k]))
            
            if diag_l:
                #Now bring l into diagonal form, trace = 1 (guaranteed by r = eye and normalization)
                ev, EV = la.eigh(self.l[k])
        
                G[k] = G[k].dot(EV)
                G_i[k] = m.H(EV).dot(G_i[k])
                
                #ev contains the squares of the Schmidt coefficients,              
                self.l[k] = m.simple_diag_matrix(ev, dtype=self.typ)
            
            j = (k + 1) % self.L
            for s in range(self.q):
                self.A[j][s] = G_i[k].dot(self.A[j][s])
                self.A[k][s] = self.A[k][s].dot(G[k])
            
            r_old = self.r[k]
            
            if rank == self.D:
                self.r[k] = m.eyemat(self.D, self.typ)
            else:
                self.r[k] = sp.zeros((self.D), dtype=self.typ)
                self.r[k][-rank:] = 1
                self.r[k] = m.simple_diag_matrix(self.r[k], dtype=self.typ)
    
            if self.sanity_checks:            
                r_ = G_i[k].dot(r_old.dot(G_i[k].conj().T)) 
                
                if not np.allclose(self.r[k], r_, 
                                   rtol=self.itr_rtol*self.check_fac,
                                   atol=self.itr_atol*self.check_fac):
                    log.warning("Sanity check failed: Restore_RCF, bad r (bad GT). Off by %s", 
                                la.norm(r_ - self.r[k]))
                
                l = self.l[k]
                for j in (sp.arange(k + 1, k + self.L + 1) % self.L):
                    l = tm.eps_l_noop(l, self.A[j], self.A[j])
                r = self.r[k]
                for j in (sp.arange(k, k - self.L, -1) % self.L):
                    r = tm.eps_r_noop(r, self.A[j], self.A[j])
                
                if not np.allclose(r, self.r[k],
                                   rtol=self.itr_rtol*self.check_fac, 
                                   atol=self.itr_atol*self.check_fac):
                    log.warning("Sanity check failed: Restore_RCF, r not eigenvector! Off by %s", 
                                la.norm(r - self.r[k]))
    
                if not np.allclose(l, self.l[k],
                                   rtol=self.itr_rtol*self.check_fac, 
                                   atol=self.itr_atol*self.check_fac):
                    log.warning("Sanity check failed: Restore_RCF, l not eigenvector! Off by %s", 
                                la.norm(l - self.l[k]))
                                
        if ret_g:
            return G, G_i
        
    def restore_LCF(self, zero_tol=None, diag_r=True, ks=None, ret_g=False):
        """Restores left canonical form.
        
        In this form, self.l = sp.eye(self.D) and self.r is diagonal, with
        the squared Schmidt coefficients corresponding to the half-chain
        decomposition as eigenvalues.
        
        Parameters
        ----------
        zero_tol : float
            Zero-tolerance. Default is the value in self.zero_tol.
        diag_r : bool
            Whether to diagonalize r. Not doing so does not fully restore
            canonical form, but may be useful in certain circumstances.
        ks : list of ints
            Which l[k] and r[k] to bring into canonical form. Default is all.
        ret_g : bool
            Whether to return the gauge-transformation matrices used.
            
        Returns
        -------
        G, G_i : ndarray
            Gauge transformation matrices g and inverses g_i.
        """
        
        if zero_tol is None:
            zero_tol = self.zero_tol
        
        if ks is None:
            ks = list(range(self.L))
        
        G = [None] * self.L
        G_i = [None] * self.L
        
        for k in ks:
            G_i[k], G[k], rank = tm.herm_fac_with_inv(self.l[k], lower=False, zero_tol=zero_tol,
                                                return_rank=True,
                                                sanity_checks=self.sanity_checks,
                                                sc_data='Restore_LCF: l')
    
            self.r[k] = G_i[k].dot(self.r[k].dot(G_i[k].conj().T))
            
            if diag_r:
                ev, EV = la.eigh(self.r[k])
        
                G[k] = EV.dot(G[k])
                G_i[k] = G_i[k].dot(EV.conj().T)
                              
                self.r[k] = m.simple_diag_matrix(ev, dtype=self.typ)
            
            j = (k + 1) % self.L
            for s in range(self.q):
                self.A[j][s] = G_i[k].dot(self.A[j][s])
                self.A[k][s] = self.A[k][s].dot(G[k])
            
            l_old = self.l[k]
            
            if rank == self.D:
                self.l[k] = m.eyemat(self.D, self.typ)
            else:
                self.l[k] = sp.zeros((self.D), dtype=self.typ)
                self.l[k][-rank:] = 1
                self.l[k] = m.simple_diag_matrix(self.l[k], dtype=self.typ)
    
            if self.sanity_checks:            
                l_ = G[k].conj().T.dot(l_old.dot(G[k])) 
                
                if not np.allclose(self.l[k], l_, 
                                   rtol=self.itr_rtol*self.check_fac,
                                   atol=self.itr_atol*self.check_fac):
                    log.warning("Sanity check failed: Restore_LCF, bad l (bad GT). Off by %s", 
                                la.norm(l_ - self.l[k]))
                
                l = self.l[k]
                for j in (sp.arange(k + 1, k + self.L + 1) % self.L):
                    l = tm.eps_l_noop(l, self.A[j], self.A[j])
                r = self.r[k]
                for j in (sp.arange(k, k - self.L, -1) % self.L):
                    r = tm.eps_r_noop(r, self.A[j], self.A[j])
                
                if not np.allclose(r, self.r[k],
                                   rtol=self.itr_rtol*self.check_fac, 
                                   atol=self.itr_atol*self.check_fac):
                    log.warning("Sanity check failed: Restore_LCF, r not eigenvector! Off by %s", 
                                la.norm(r - self.r[k]))
    
                if not np.allclose(l, self.l[k],
                                   rtol=self.itr_rtol*self.check_fac, 
                                   atol=self.itr_atol*self.check_fac):
                    log.warning("Sanity check failed: Restore_LCF, l not eigenvector! Off by %s", 
                                la.norm(l - self.l[k]))
        if ret_g:
            return G, G_i
    
    def restore_CF(self):
        """Restores canonical form.
        
        Performs self.restore_RCF() or self.restore_SCF()
        depending on self.symm_gauge.        
        """
        if self.symm_gauge:
            return self.restore_SCF()
        else:
            return self.restore_RCF()
            
    def auto_truncate(self, update=True, zero_tol=None):
        if zero_tol is None:
            zero_tol = self.zero_tol
        
        new_D_l = 1
        for k in range(self.L):
            new_D_l = max(np.count_nonzero(self.l[k].diag > zero_tol), new_D_l)
        
        if 0 < new_D_l < self.A[0].shape[1]:
            self.truncate(new_D_l, update=False)
            
            if update:
                self.update()
                
            return True
        else:
            return False
    
    def truncate(self, newD, update=True):
        assert newD < self.D, 'new bond-dimension must be smaller!'
        
        tmp_As = self.A
        tmp_ls = [x.diag for x in self.l]
        
        self._init_arrays(newD, self.q, self.L)
        
        if self.symm_gauge:
            for k in range(self.L):
                self.l[k] = m.simple_diag_matrix(tmp_ls[k][:self.D], dtype=self.typ)
                self.r[k] = m.simple_diag_matrix(tmp_ls[k][:self.D], dtype=self.typ)
                self.A[k] = tmp_As[k][:, :self.D, :self.D]
        else:
            for k in range(self.L):
                self.l[k] = m.simple_diag_matrix(tmp_ls[k][-self.D:], dtype=self.typ)
                self.r[k] = m.eyemat(self.D, dtype=self.typ)
                self.A[k] = tmp_As[k][:, -self.D:, -self.D:]
            
        self.lL_before_CF = self.l[-1].A
        self.rL_before_CF = self.r[-1].A

        if update:
            self.update()
    
    def calc_AA(self):
        """Calculates the products A[s] A[t] for s, t in range(self.q).
        The result is stored in self.AA.
        """
        for k in range(len(self.A)):
            self.AA[k] = tm.calc_AA(self.A[k], self.A[(k + 1) % self.L])
        
        
    def update(self, restore_CF=True, auto_truncate=False, restore_CF_after_trunc=True):
        """Updates secondary quantities to reflect the state parameters self.A.
        
        Must be used after changing the parameters self.A before calculating
        physical quantities, such as expectation values.
        
        Also (optionally) restores the right canonical form.
        
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
        assert restore_CF or not auto_truncate, "auto_truncate requires restore_CF"
        
        self.calc_lr()
        if restore_CF:
            self.restore_CF()
            if auto_truncate and self.auto_truncate(update=False):
                log.info("Auto-truncated! New D: %d", self.D)
                self.calc_lr()
                if restore_CF_after_trunc:
                    self.restore_CF()
                
        #self.calc_AA()
            
    def fidelity_per_site(self, other, full_output=False, left=False, 
                          force_dense=False, force_sparse=False,
                          dense_cutoff=64):
        """Returns the per-site fidelity d.
              
        Also returns the largest eigenvalue "w" of the overlap transfer
        operator, as well as the corresponding eigenvector "V" in the
        matrix representation.
        
        If the fidelity per site is 1:
        
          A^s = w g A'^s g^-1      (with g = V r'^-1)
            
        Parameters
        ----------
        other : EvoMPS_MPS_Uniform
            MPS with which to calculate the per-site fidelity.
        full_output : bool
            Whether to return the eigenvector V.
        left : bool
            Whether to find the left eigenvector (instead of the right)
        force_dense : bool
            Forces use of a dense eigenvalue decomposition algorithm
        force_sparse : bool
            Forces use of a sparse, iterative eigenvalue solver (except if D == 1)
        dense_cutoff : int
            The matrix dimension above which a sparse eigenvalue solver is used 
        
        Returns
        -------
        d : float
            The per-site fidelity.
        w : float
            The largest eigenvalue of the overlap transfer operator.
        V : ndarray
            The right (or left if left == True) eigenvector corresponding to w (if full_output == True).
        """
        assert self.q == other.q, "Hilbert spaces must have same dimensions!"
        
        from fractions import gcd
        L_ = self.L * other.L // gcd(self.L, other.L)

        As1 = list(self.A) * (L_ // self.L)
        As2 = list(other.A) * (L_ // other.L)

        ED = self.D * other.D
        
        if ED == 1:
            ev = 1
            for k in range(L_):
                evk = 0
                for s in range(self.q):    
                    evk += As1[k][s] * As2[k][s].conj()
                ev *= evk
            if left:
                ev = ev.conj()
            if full_output:
                return abs(ev)**(1./L_), ev, sp.ones((1), dtype=self.typ)
            else:
                return abs(ev)**(1./L_), ev
        elif (ED <= dense_cutoff or force_dense) and not force_sparse:
            E = m.eyemat(ED, dtype=As1[0].dtype)
            for k in range(L_):
                Ek = sp.zeros((ED, ED), dtype=As1[0].dtype)
                for s in range(As1[k].shape[0]):
                    Ek += sp.kron(As1[k][s], As2[k][s].conj())
                E = E.dot(Ek)
                
            if full_output:
                ev, eV = la.eig(E, left=left, right=not left)
            else:
                ev = la.eigvals(E)
            
            ind = abs(ev).argmax()
            if full_output:
                return abs(ev[ind])**(1./L_), ev[ind], eV[:, ind]
            else:
                return abs(ev[ind])**(1./L_), ev[ind]
        else:
            opE = self._get_EOP(As1, As2, left)
            res = las.eigs(opE, which='LM', k=1, ncv=20, return_eigenvectors=full_output)
            if full_output:
                ev, eV = res
                return abs(ev[0])**(1./L_), ev[0], eV[:, 0]
            else:
                ev = res
                return abs(ev[0])**(1./L_), ev[0]
            
    def phase_align(self, other):
        """Adjusts the parameter tensor A by a phase-factor to align it with another state.
        
        This ensures that the largest eigenvalue of the overlap transfer operator
        is real.
        
        An update() is needed after doing this!
        
        Parameters
        ----------
        other : EvoMPS_MPS_Uniform
            MPS with which to calculate the per-site fidelity.
        
        Returns
        -------
        phi : complex
            The phase difference (phase of the eigenvalue).
        """
        d, phi = self.fidelity_per_site(other, full_output=False, left=False)
        
        self.A[0] *= phi.conj()
        
        return phi

    def gauge_align(self, other, tol=1E-12):
        """Gauge-align the state with another.
        
        Given two states that differ only by a gauge-transformation
        and a phase, this equalizes the parameter tensors by performing 
        the required transformation.
        
        This is only really useful if the corresponding gauge transformation
        matrices are needed for some reason (otherwise you can just check
        the fidelity and set the parameters equal).
        
        Parameters
        ----------
        other : EvoMPS_MPS_Uniform
            MPS with which to calculate the per-site fidelity.
        tol : float
            Tolerance for detecting per-site fidelity != 1.
            
        Returns
        -------
        Nothing if the per-site fidelity is not 1. Otherwise:
            
        g : ndarray
            The gauge-transformation matrix used.
        g_i : ndarray
            The inverse of g.
        phi : complex
            The phase factor.
        """
        d, phi, gRL = self.fidelity_per_site(other, full_output=True)
        
        if abs(d - 1) > tol:
            return
            
        #Phase-align first
        self.A[0] *= phi.conj()
            
        gRL = gRL.reshape(self.D, self.D)
        
        g = [None] * self.L
        gi = [None] * self.L
                
        for k in range(self.L - 1, -1, -1):
            if k == self.L - 1:
                gRk = gRL
            else:
                gRk = tm.eps_r_noop(other.rs[k + 1], self.A[k + 1], other.As[k + 1])
                
            try:
                g[k] = other.rs[k].inv().dotleft(gRk)
            except:
                g[k] = gRk.dot(m.invmh(other.rs[k]))
                
            gi[k] = la.inv(g[k])
            
            j = (k + 1) % self.L
            for s in range(self.q):
                self.A[j][s] = gi[k].dot(self.A[j][s])
                self.A[k][s] = self.A[k][s].dot(g[k])
                
            self.l[k] = m.H(g[k]).dot(self.l[k].dot(g[k]))
            
            self.r[k] = gi[k].dot(self.r[k].dot(m.H(gi[k])))
        
        return g, gi, phi
        
    def schmidt_sq(self, k=0):
        """Returns the squared Schmidt coefficients for a left-right parition.
        
        The chain can be split into two parts between any two sites.
        This returns the squared coefficients of the corresponding Schmidt
        decomposition, which are equal to the eigenvalues of the corresponding
        reduced density matrix.
        
        Parameters
        ----------
        k : int
            Site offset for split.
            
        Returns
        -------
        lam : sequence of float (if ret_schmidt_sq==True)
            The squared Schmidt coefficients.
        """
        lr = self.l[k].dot(self.r[k])
        try: 
            lam = lr.diag
        except AttributeError: #Assume we are not in canonical form.
            lam = la.eigvals(lr)
        return lam
                        
    def entropy(self, k=0, ret_schmidt_sq=False):
        """Returns the von Neumann entropy of one half of the system.
        
        The chain can be split into two halves between any two sites.
        This function returns the corresponding von Neumann entropy, which
        is a measure of the entanglement between the two parts.
        
        For block lengths L > 1, the parameter k specifies that the splitting
        should be done between sites k and k + 1 within a block.
        
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
        lam = self.schmidt_sq(k=k)
        S = -np.sum(lam * sp.log2(lam)).real
            
        if ret_schmidt_sq:
            return S, lam
        else:
            return S
            
    def expect_1s(self, op, k=0):
        """Computes the expectation value of a single-site operator.
        
        The operator should be a self.q x self.q matrix or generating function 
        such that op[s, t] or op(s, t) equals <s|op|t>.
        
        The state must be up-to-date -- see self.update()!
        
        Parameters
        ----------
        op : ndarray or callable
            The operator.
        k : int
            Site offset within block.
            
        Returns
        -------
        expval : floating point number
            The expectation value (data type may be complex)
        """        
        if callable(op):
            op = np.vectorize(op, otypes=[np.complex128])
            op = np.fromfunction(op, (self.q, self.q))
            
        Or = tm.eps_r_op_1s(self.r[k], self.A[k], self.A[k], op)
        
        return m.adot(self.l[k - 1], Or)
        
    def expect_1s_1s(self, op1, op2, d, k=0, return_intermediates=False):
        """Computes the expectation value of two single site operators acting 
        on two different sites.
        
        The result is < op1_n op2_n+d > with the operators acting on sites
        n and n + d.
        
        See expect_1s().
        
        Requires d > 0. Optionally returns results for all distances 0 to d.
        
        The state must be up-to-date -- see self.update()!
        
        Parameters
        ----------
        op1 : ndarray or callable
            The first operator, acting on the first site.
        op2 : ndarray or callable
            The second operator, acting on the second site.
        d : int
            The distance (number of sites) between the two sites acted on non-trivially.
        k : int
            Site offset of first site within block.
        return_intermediates : bool
            Whether to return results for intermediate d.
            
        Returns
        -------
        expval : complex128 or sequence of complex128
            The expectation value (data type may be complex), or values if
            return_intermediates == True.
        """        
        
        assert d > 0, 'd must be greater than 1'
        
        if callable(op1):
            op1 = sp.vectorize(op1, otypes=[sp.complex128])
            op1 = sp.fromfunction(op1, (self.q, self.q))
        
        if callable(op2):
            op2 = sp.vectorize(op2, otypes=[sp.complex128])
            op2 = sp.fromfunction(op2, (self.q, self.q))
            
        L = self.L
        
        res = sp.zeros((d + 1), dtype=sp.complex128)
        lj = tm.eps_l_op_1s(self.l[(k - 1) % L], self.A[k], self.A[k], op1)
        
        if return_intermediates:
            res[0] = self.expect_1s(op1.dot(op2), k=k)

        for j in range(1, d + 1):
            if return_intermediates or j == d:
                lj_op = tm.eps_l_op_1s(lj, self.A[(k + j) % L], self.A[(k + j) % L], op2)
                res[j] = m.adot(lj_op, self.r[(k + j) % L])
                
            if j < d:
                lj = tm.eps_l_noop(lj, self.A[(k + j) % L], self.A[(k + j) % L])
                
        if return_intermediates:
            return res
        else:
            return res[-1]
            
    def correlation_1s_1s(self, op1, op2, d, k=0, return_exvals=False):
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
        d : int
            The distance (number of sites) between the two sites acted on non-trivially.
        k : int
            Site offset of first site within block.
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
        L = self.L
        ex1 = sp.zeros((L), dtype=sp.complex128)
        for j in range(L):
            ex1[j] = self.expect_1s(op1, (k + j) % L)
            
        if op1 is op2:
            ex2 = ex1
        else:
            ex2 = sp.zeros((L), dtype=sp.complex128)
            for j in range(L):
                ex2[j] = self.expect_1s(op2, (k + j) % L)
            
        cf = self.expect_1s_1s(op1, op2, d, k=k, return_intermediates=True)
            
        ccf = sp.zeros((d + 1), dtype=sp.complex128)
        for n in range(d + 1):
            ccf[n] = cf[n] - ex1[0] * ex2[n % L]
            
        if return_exvals:
            ex1_ = sp.tile(ex1, [(d + 1) // L + 1])[:d + 1]
            ex2_ = sp.tile(ex1, [(d + 1) // L + 1])[:d + 1]
            return ccf, ex1_, ex2_
        else:
            return ccf
            
    def expect_2s(self, op, k=0):
        """Computes the expectation value of a nearest-neighbour two-site operator.
        
        The operator should be a q x q x q x q array 
        such that op[s, t, u, v] = <st|op|uv> or a function of the form 
        op(s, t, u, v) = <st|op|uv>.
        
        The state must be up-to-date -- see self.update()!
        
        Parameters
        ----------
        op : ndarray or callable
            The operator array or function.
        k : int
            Site offset within block.
            
        Returns
        -------
        expval : floating point number
            The expectation value (data type may be complex)
        """
        if callable(op):
            op = np.vectorize(op, otypes=[np.complex128])
            op = np.fromfunction(op, (self.q, self.q, self.q, self.q))        
        
        Ck = tm.calc_C_mat_op_AA(op, self.AA[k])
        res = tm.eps_r_op_2s_C12_AA34(self.r[k], Ck, self.AA[k])
        
        return m.adot(self.l[k - 1], res)
        
    def expect_2s_tp(self, op_tp, k=0):
        """Computes the expectation value of a nearest-neighbour two-site operator.
        
        The operator should be a q x q x q x q array 
        such that op[s, t, u, v] = <st|op|uv> or a function of the form 
        op(s, t, u, v) = <st|op|uv>.
        
        The state must be up-to-date -- see self.update()!
        
        Parameters
        ----------
        op : list of list of ndarrays
            The operator in tensor product decomposition
        k : int
            Site offset within block.
            
        Returns
        -------
        expval : floating point number
            The expectation value (data type may be complex)
        """
        L = self.L
        Ck = tm.calc_C_tp(op_tp, self.A[k], self.A[(k + 1) % L])
        res = tm.eps_r_op_2s_C12_tp(self.r[k], Ck, self.A[k], self.A[(k + 1) % L])
        
        return m.adot(self.l[k - 1], res)
        
    def expect_3s(self, op, k=0):
        """Computes the expectation value of a nearest-neighbour three-site operator.

        The operator should be a q x q x q x q x q x q 
        array such that op[s, t, u, v, w, x] = <stu|op|vwx> 
        or a function of the form op(s, t, u, v, w, x) = <stu|op|vwx>.

        The state must be up-to-date -- see self.update()!

        Parameters
        ----------
        op : ndarray or callable
            The operator array or function.
        k : int
            Site offset within block.
            
        Returns
        -------
        expval : floating point number
            The expectation value (data type may be complex)
        """
        A = self.A
        AAA = tm.calc_AAA(A[k], A[(k + 1) % self.L], A[(k + 2) % self.L])

        if callable(op):
            op = sp.vectorize(op, otypes=[sp.complex128])
            op = sp.fromfunction(op, (A.shape[0], A.shape[0], A.shape[0],
                                      A.shape[0], A.shape[0], A.shape[0]))

        C = tm.calc_C_3s_mat_op_AAA(op, AAA)
        res = tm.eps_r_op_3s_C123_AAA456(self.r[(k + 2) % self.L], C, AAA)
        return m.adot(self.l[k - 1], res)
        
    def density_1s(self, k=0):
        """Returns a reduced density matrix for a single site.
        
        The site number basis is used: rho[s, t] 
        with 0 <= s, t < q.
        
        The state must be up-to-date -- see self.update()!
            
        Parameters
        ----------
        k : int
            Site offset within block.

        Returns
        -------
        rho : ndarray
            Reduced density matrix in the number basis.
        """
        rho = np.empty((self.q, self.q), dtype=self.typ)
        for s in range(self.q):
            for t in range(self.q):                
                rho[s, t] = m.adot(self.l[k - 1], m.mmul(self.A[k][t], self.r[k], m.H(self.A[k][s])))
        return rho
                
    def apply_op_1s(self, op, k=0, do_update=True):
        """Applies a single-site operator to all sites.
        
        This applies the product (or string) of a single-site operator o_n over 
        all sites so that the new state |Psi'> = ...o_(n-1) o_n o_(n+1)... |Psi>.
        
        By default, this performs self.update(), which also restores
        state normalization.
        
        Parameters
        ----------
        op : ndarray or callable
            The single-site operator. See self.expect_1s().
        k : int
            Site offset within block.
        do_update : bool
            Whether to update after applying the operator.
        """
        if callable(op):
            op = np.vectorize(op, otypes=[np.complex128])
            op = np.fromfunction(op, (self.q, self.q))
            
        newAk = sp.zeros_like(self.A[k])
        
        for s in range(self.q):
            for t in range(self.q):
                newAk[s] += self.A[k][t] * op[s, t]
                
        self.A[k] = newAk
        
        if do_update:
            self.update()

    def expect_string_1s_density_hc(self, op, k=0, ncv=20, return_g=False):
        """Returns the expectation values of a string operator acting on the
           Schmidt vectors for the half-chain decomposition.
           
        The string operator must be a string of single site operators.
        
        For block lengths other than 1, the cut is *before* the kth site in a 
        block (counting from zero).
        
        The operator should be a self.q x self.q matrix or generating function 
        such that op[s, t] or op(s, t) equals <s|op|t>.
        
        The state must be up-to-date and in canonical form -- see self.update()!
        
        The results only make sense if the string is a symmetry of the state,
        such that it consitutes and MPS gauge transformation. In this case,
        the fidelity per site will be equal to 1.
        
        Parameters
        ----------
        op : ndarray or callable
            The operator.
        k : int
            Site offset within block.
        ncv : int
            Parameter for ARNOLDI iteration. See scipy.sparse.linalg.eigs().
        return_g : bool
            Return the representation of "op" on the virtual indices.
            
        Returns
        -------
        expval : ndarray
            The expectation values for each Schmidt vector (data type may be complex).
        fid_per_site : float
            Fidelity per site of state with transformed state.
        """
        if callable(op):
            op = np.vectorize(op, otypes=[np.complex128])
            op = np.fromfunction(op, (self.q, self.q))
            
        Ashift = self.A[k:] + self.A[:k]
            
        Aop = [np.tensordot(op, A, axes=([1],[0])) for A in Ashift]
        
        if self.D == 1:
            ev = 1
            for j in range(self.L):
                evk = 0
                for s in range(self.q):
                    evk += Aop[j][s] * Ashift[j][s].conj()
                ev *= evk
            eV = sp.ones((1), dtype=sp.complex128)
        else:            
            opE = self._get_EOP(Aop, Ashift, False)
            ev, eV = las.eigs(opE, v0=np.asarray(self.r[(k - 1) % self.L]), which='LM', k=1, ncv=ncv)
            ev = ev[0]
            #Note: eigs normalizes the eigenvector so that norm(eV) = 1.
            
        r = eV.reshape((self.D, self.D))
        
        if self.symm_gauge:
            g = self.r[(k - 1) % self.L].inv().dot(r) #r = self.r[k] * g
        else:
            g = r * sp.sqrt(self.D) #Must restore normalization.
            
        Or = r.diagonal().copy()
        
        if return_g:
            return Or, abs(ev)**(1./self.L), g
        else:
            return Or, abs(ev)**(1./self.L)

    def expect_string_per_site_1s(self, op, ncv=20):
        """Calculates the per-site factor of a string expectation value.
        
        The string operator is the product over all sites of a single-site
        operator 'op'. 
        
        The expectation value is related to the infinite power of the per-site 
        fidelity of the original state and the state obtained by applying the 
        string. Since the operator need not preserve the norm, this can be 
        greater than 1, in which case the expectation value diverges.
        
        The expectation value of a string is only well-defined if the string
        is a symmetry of the state such that the per-site factor == 1, although
        an absolute value of 1 ensures that it does not diverge or go to zero.
        
        Parameters
        ----------
        op : ndarray or callable
            The single-site operator. See self.expect_1s().
            
        Returns
        -------
        ev : complex
            The per-site factor of the expectation value.
        """
        if callable(op):
            op = np.vectorize(op, otypes=[np.complex128])
            op = np.fromfunction(op, (self.q, self.q))
        
        Aop = [np.tensordot(op, A, axes=([1],[0])) for A in self.A]
        
        if self.D == 1:
            ev = 1
            for k in range(self.L):
                evk = 0
                for s in range(self.q):
                    evk += Aop[k][s] * self.A[k][s].conj()
                ev *= evk
            return ev**(1./self.L)
        else:            
            opE = self._get_EOP(Aop, self.A, False)
            ev = las.eigs(opE, v0=np.asarray(self.r[-1]), which='LM', k=1, ncv=ncv)
            return ev[0]**(1./self.L)
            
    def expect_string_1s(self, op, k, d):
        """Calculates the expectation values of finite strings
        with lengths 1 to d, starting at position k.
        """
        if callable(op):
            op = np.vectorize(op, otypes=[np.complex128])
            op = np.fromfunction(op, (self.q, self.q))
        
        Aop = [np.tensordot(op, A, axes=([1],[0])) for A in self.A]
        
        res = sp.zeros((d), dtype=self.A[0].dtype)
        x = self.l[(k - 1) % self.L]
        for n in range(k, k + d + 1):
            nm = n % self.L
            x = tm.eps_l_noop(x, self.A[nm], Aop[nm])
            res[n - k - 1] = m.adot(x, self.r[nm])
        
        return res
        
    def basis_occupancy(self, k=0):
        L = self.L
        A = self.A
        l = self.l
        r = self.r
        res = [m.adot(l[(k - 1) % L], A[k][s].dot(r[k].dot(A[k][s].conj().T))) 
               for s in range(self.q)]
        return sp.array(res).real
                
    def set_q(self, newq, offset=0):
        """Alter the single-site Hilbert-space dimension q.
        
        Any added parameters are set to zero.
        
        Parameters
        ----------
        newq : int
            The new dimension.
        """
        oldq = self.q
        oldA = self.A
        
        oldl = self.l
        oldr = self.r
        
        self._init_arrays(self.D, newq, self.L) 
        
        self.l = oldl
        self.r = oldr
        
        for A in self.A:
            A.fill(0)
        for k in range(self.L):
            if self.q > oldq:
                self.A[k][offset:oldq + offset, :, :] = oldA[k]
            else:
                self.A[k][:] = oldA[k][-offset:self.q - offset, :, :]
        
            
    def expand_D(self, newD, refac=100, imfac=0):
        """Expands the bond dimension in a simple way.
        
        New matrix entries are (mostly) randomized.
        
        Parameters
        ----------
        newD : int
            The new bond-dimension.
        refac : float
            Scaling factor for the real component of the added noise.
        imfac : float
            Scaling factor for the imaginary component of the added noise.
        """
        if newD < self.D:
            return False
        
        oldD = self.D
        oldA = self.A
        
        oldl = self.l
        oldr = self.r
        oldl = list(map(np.asarray, oldl))
        oldr = list(map(np.asarray, oldr))
        
        self._init_arrays(newD, self.q, self.L)
        
        for k in range(self.L):
            realnorm = la.norm(oldA[k].real.ravel())
            imagnorm = la.norm(oldA[k].imag.ravel())
            realfac = (realnorm / oldA[k].size) * refac
            imagfac = (imagnorm / oldA[k].size) * imfac
    #        m.randomize_cmplx(newA[:, self.D:, self.D:], a=-fac, b=fac)
            m.randomize_cmplx(self.A[k][:, :oldD, oldD:], a=0, b=realfac, aj=0, bj=imagfac)
            m.randomize_cmplx(self.A[k][:, oldD:, :oldD], a=0, b=realfac, aj=0, bj=imagfac)
            self.A[k][:, oldD:, oldD:] = 0 #for nearest-neighbour hamiltonian
    
    #        self.A[:, :oldD, oldD:] = oldA[:, :, :(newD - oldD)]
    #        self.A[:, oldD:, :oldD] = oldA[:, :(newD - oldD), :]
            self.A[k][:, :oldD, :oldD] = oldA[k]
    
            self.l[k][:oldD, :oldD] = oldl[k]
            val = abs(oldl[k].mean())
            m.randomize_cmplx(self.l[k].ravel()[oldD**2:], a=0, b=val, aj=0, bj=0)
            #self.l[:oldD, oldD:].fill(0 * 1E-3 * la.norm(oldl) / oldD**2)
            #self.l[oldD:, :oldD].fill(0 * 1E-3 * la.norm(oldl) / oldD**2)
            #self.l[oldD:, oldD:].fill(0 * 1E-3 * la.norm(oldl) / oldD**2)
            
            self.r[k][:oldD, :oldD] = oldr[k]
            val = abs(oldr[k].mean())
            m.randomize_cmplx(self.r[k].ravel()[oldD**2:], a=0, b=val, aj=0, bj=0)
            #self.r[oldD:, :oldD].fill(0 * 1E-3 * la.norm(oldr) / oldD**2)
            #self.r[:oldD, oldD:].fill(0 * 1E-3 * la.norm(oldr) / oldD**2)
            #self.r[oldD:, oldD:].fill(0 * 1E-3 * la.norm(oldr) / oldD**2)
