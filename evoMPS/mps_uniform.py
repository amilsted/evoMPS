# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 17:29:27 2011

@author: Ashley Milsted

"""
import numpy as np
import scipy as sp
import scipy.linalg as la
import scipy.sparse.linalg as las
import tdvp_common as tm
import matmul as m
import math as ma

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
        
        self.D = A1.shape[1]
        
        self.shape = (self.D**2, self.D**2)
        
        self.dtype = np.dtype(A1.dtype)
        
        self.out = np.empty((self.D, self.D), dtype=self.dtype)
        
        self.calls = 0
        
        if left:
            self.eps = tm.eps_l_noop_inplace
        else:
            self.eps = tm.eps_r_noop_inplace
    
    def matvec(self, v):
        """Matrix-vector multiplication. 
        Result = Ev or vE (if self.left == True).
        """
        x = v.reshape((self.D, self.D))

        Ex = self.eps(x, self.A1, self.A2, self.out)
        
        self.calls += 1
        
        return Ex.ravel()

class EvoMPS_MPS_Uniform(object):   
        
    def __init__(self, D, q, dtype=None):
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
        dtype : numpy-compatible dtype
            The data-type to be used. The default is double-precision complex.
        """
        self.odr = 'C' 
        
        if dtype is None:
            self.typ = np.complex128
        
        self.itr_rtol = 1E-13
        self.itr_atol = 1E-14
        
        self.itr_l = 0
        self.itr_r = 0
        
        self.pow_itr_max = 2000
        self.ev_use_arpack = True
        self.ev_arpack_nev = 1
        self.ev_arpack_ncv = None
                
        self.symm_gauge = False
        
        self.sanity_checks = False
        self.check_fac = 50
        
        self.userdata = None        
        
        self.eps = np.finfo(self.typ).eps
                
        self._init_arrays(D, q)        
                    
        self.randomize()

    def randomize(self, do_update=True):
        """Randomizes the parameter tensors self.A.
        
        Parameters
        ----------
        do_update : bool (True)
            Whether to perform self.update() after randomizing.
        """
        m.randomize_cmplx(self.A)
        self.A /= la.norm(self.A)
        
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
        norm = la.norm(self.A)
        fac = fac * (norm / (self.q * self.D**2))        
        
        R = np.empty_like(self.A)
        m.randomize_cmplx(R, -fac / 2.0, fac / 2.0)
        
        self.A += R
        
        if do_update:
            self.update()
    
    def _init_arrays(self, D, q):
        self.D = D
        self.q = q
        
        self.A = np.zeros((q, D, D), dtype=self.typ, order=self.odr)
        self.AA = np.zeros((q, q, D, D), dtype=self.typ, order=self.odr)
        
        self.l = np.ones_like(self.A[0])
        self.r = np.ones_like(self.A[0])
        self.l_before_CF = self.l
        self.r_before_CF = self.r
        self.conv_l = True
        self.conv_r = True
        
        self.tmp = np.zeros_like(self.A[0])
        
    def _calc_lr_brute(self):
        E = np.zeros((self.D**2, self.D**2), dtype=self.typ, order='C')
        
        for s in xrange(self.q):
            E += sp.kron(self.A[s], self.A[s].conj())
            
        ev, eVL, eVR = la.eig(E, left=True, right=True)
        
        i = np.argmax(ev)
        
        self.A *= 1 / sp.sqrt(ev[i])        
        
        self.l = eVL[:,i].reshape((self.D, self.D))
        self.r = eVR[:,i].reshape((self.D, self.D))
        
        norm = m.adot(self.l, self.r)
        self.l *= 1 / sp.sqrt(norm)
        self.r *= 1 / sp.sqrt(norm)        
        
        print "Sledgehammer:"
        print "Left ok?: " + str(np.allclose(
                                tm.eps_l_noop(self.l, self.A, self.A), self.l))
        print "Right ok?: " + str(np.allclose(
                                tm.eps_r_noop(self.r, self.A, self.A), self.r))
    
    def _calc_lr_ARPACK(self, x, tmp, calc_l=False, A1=None, A2=None, rescale=True,
                        tol=1E-14, ncv=None, k=1):
        if A1 is None:
            A1 = self.A
        if A2 is None:
            A2 = self.A
            
        if self.D == 1:
            x.fill(1)
            if calc_l:
                ev = tm.eps_l_noop(x, A1, A2)[0, 0]
            else:
                ev = tm.eps_r_noop(x, A1, A2)[0, 0]
            
            if rescale and not abs(ev - 1) < tol:
                A1 *= 1 / sp.sqrt(ev)
            
            return x, True, 1
                        
        try:
            norm = la.get_blas_funcs("nrm2", [x])
        except (ValueError, AttributeError):
            norm = np.linalg.norm
    
        n = x.size #we will scale x so that stuff doesn't get too small
        
        #start = time.clock()
        opE = EOp(A1, A2, calc_l)
        x *= n / norm(x.ravel())
        try:
            ev, eV = las.eigs(opE, which='LM', k=k, v0=x.ravel(), tol=tol, ncv=ncv)
            conv = True
        except las.ArpackNoConvergence:
            print "Reset! (l? %s)" % str(calc_l)
            ev, eV = las.eigs(opE, which='LM', k=k, tol=tol, ncv=ncv)
            conv = True
            
        #print ev2
        #print ev2 * ev2.conj()
        ind = ev.argmax()
        ev = np.real_if_close(ev[ind])
        ev = np.asscalar(ev)
        eV = eV[:, ind]
        
        #remove any additional phase factor
        eVmean = eV.mean()
        eV *= np.sqrt(np.conj(eVmean) / eVmean)
        
        if eV.mean() < 0:
            eV *= -1

        eV = eV.reshape(self.D, self.D)
        
        eV *= n / norm(eV.ravel())
        
        x[:] = eV
        
        #print "splinalg: %g" % (time.clock() - start)   
        
        #print "Herm? %g" % norm((eV - m.H(eV)).ravel())
        #print "Norm of diff: %g" % norm((eV - x).ravel())
        #print "Norms: (%g, %g)" % (norm(eV.ravel()), norm(x.ravel()))
                    
        if rescale and not abs(ev - 1) < tol:
            A1 *= 1 / sp.sqrt(ev)
            if self.sanity_checks:
                if not A1 is A2:
                    print "Sanity check failed: Re-scaling with A1 <> A2!"
                if calc_l:
                    tm.eps_l_noop_inplace(x, A1, A2, tmp)
                else:
                    tm.eps_r_noop_inplace(x, A1, A2, tmp)
                ev = tmp.mean() / x.mean()
                if not abs(ev - 1) < tol:
                    print "Sanity check failed: Largest ev after re-scale = " + str(ev)
        
        return x, conv, opE.calls
        
    def _calc_E_largest_two_eigenvalues(self, tol=1E-6, ncv=10):
        opE = EOp(self.A, self.A, False)
        
        r = np.asarray(self.r)
        
        ev = las.eigs(opE, which='LM', k=2, v0=r.ravel(), tol=tol, ncv=ncv,
                          return_eigenvectors=False)
                          
        return ev
        
    def calc_E_gap(self, tol=1E-6, ncv=10):
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
        ncv : int
            Number of Arnoldii basis vectors to store.
        """
        ev = self._calc_E_largest_two_eigenvalues(tol=tol, ncv=ncv)
                          
        ev1_mag = abs(ev).max()
        ev2_mag = abs(ev).min()
        
        return ((ev1_mag - ev2_mag) / ev1_mag)
        
    def correlation_length(self, tol=1E-6, ncv=10):
        """
        Calculates the correlation length in units of the lattice spacing.
        
        The correlation length is equal to the inverse of the natural logarithm
        of the maginitude of the second-largest eigenvalue of the transfer 
        (or super) operator E.
        
        Parameters
        ----------
        tol : float
            Tolerance for second-largest eigenvalue.
        ncv : int
            Number of Arnoldii basis vectors to store.
        """
        ev = self._calc_E_largest_two_eigenvalues(tol=tol, ncv=ncv)
                          
        ev1_mag = abs(ev).max()
        ev2_mag = abs(ev).min()
        
        if not abs(1 - ev1_mag) < self.itr_rtol:
            print "Warning: Largest eigenvalue != 1"
        
        return -1 / sp.log(ev2_mag)
                
    def _calc_lr(self, x, tmp, calc_l=False, A1=None, A2=None, rescale=True,
                 max_itr=1000, rtol=1E-14, atol=1E-14):
        """Power iteration to obtain eigenvector corresponding to largest
           eigenvalue.
           
           x is modified in place.
        """        
        if A1 is None:
            A1 = self.A
        if A2 is None:
            A2 = self.A
                        
        try:
            norm = la.get_blas_funcs("nrm2", [x])
        except (ValueError, AttributeError):
            norm = np.linalg.norm
            
#        try:
#            allclose = ac.allclose_mat
#        except:
#            allclose = np.allclose
#            print "Falling back to numpy allclose()!"
        
        n = x.size #we will scale x so that stuff doesn't get too small

        x *= n / norm(x.ravel())
        tmp[:] = x
        for i in xrange(max_itr):
            x[:] = tmp
            if calc_l:
                tm.eps_l_noop_inplace(x, A1, A2, tmp)
            else:
                tm.eps_r_noop_inplace(x, A1, A2, tmp)
            ev_mag = norm(tmp.ravel()) / n
            ev = (tmp.mean() / x.mean()).real
            tmp *= (1 / ev_mag)
            if norm((tmp - x).ravel()) < atol + rtol * n:
#            if allclose(tmp, x, rtol, atol):                
                #print (i, ev, ev_mag, norm((tmp - x).ravel())/n, atol, rtol)
                x[:] = tmp
                break            
#        else:
#            print (i, ev, ev_mag, norm((tmp - x).ravel())/norm(x.ravel()), atol, rtol)
                    
        if rescale and not abs(ev - 1) < atol:
            A1 *= 1 / sp.sqrt(ev)
            if self.sanity_checks:
                if not A1 is A2:
                    print "Sanity check failed: Re-scaling with A1 <> A2!"
                if calc_l:
                    tm.eps_l_noop_inplace(x, A1, A2, tmp)
                else:
                    tm.eps_r_noop_inplace(x, A1, A2, tmp)
                ev = tmp.mean() / x.mean()
                if not abs(ev - 1) < atol:
                    print "Sanity check failed: Largest ev after re-scale = " + str(ev)
        
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
        self.l_before_CF = np.asarray(self.l_before_CF)
        self.r_before_CF = np.asarray(self.r_before_CF)
        
        if self.ev_use_arpack:
            self.l, self.conv_l, self.itr_l = self._calc_lr_ARPACK(self.l_before_CF, tmp,
                                                   calc_l=True,
                                                   tol=self.itr_rtol,
                                                   k=self.ev_arpack_nev,
                                                   ncv=self.ev_arpack_ncv)
        else:
            self.l, self.conv_l, self.itr_l = self._calc_lr(self.l_before_CF, 
                                                    tmp, 
                                                    calc_l=True,
                                                    max_itr=self.pow_itr_max,
                                                    rtol=self.itr_rtol, 
                                                    atol=self.itr_atol)
                                        
        self.l_before_CF = self.l.copy()

        if self.ev_use_arpack:
            self.r, self.conv_r, self.itr_r = self._calc_lr_ARPACK(self.r_before_CF, tmp, 
                                                   calc_l=False,
                                                   tol=self.itr_rtol,
                                                   k=self.ev_arpack_nev,
                                                   ncv=self.ev_arpack_ncv)
        else:
            self.r, self.conv_r, self.itr_r = self._calc_lr(self.r_before_CF, 
                                                    tmp, 
                                                    calc_l=False,
                                                    max_itr=self.pow_itr_max,
                                                    rtol=self.itr_rtol, 
                                                    atol=self.itr_atol)
        self.r_before_CF = self.r.copy()
            
        #normalize eigenvectors:

        if self.symm_gauge:
            norm = m.adot(self.l, self.r).real
            itr = 0 
            while not np.allclose(norm, 1, atol=1E-13, rtol=0) and itr < 10:
                self.l *= 1. / ma.sqrt(norm)
                self.r *= 1. / ma.sqrt(norm)
                
                norm = m.adot(self.l, self.r).real
                
                itr += 1
                
            if itr == 10:
                print "Warning: Max. iterations reached during normalization!"
        else:
            fac = self.D / np.trace(self.r).real
            self.l *= 1 / fac
            self.r *= fac

            norm = m.adot(self.l, self.r).real
            itr = 0 
            while not np.allclose(norm, 1, atol=1E-13, rtol=0) and itr < 10:
                self.l *= 1. / norm
                norm = m.adot(self.l, self.r).real
                itr += 1
                
            if itr == 10:
                print "Warning: Max. iterations reached during normalization!"

        if self.sanity_checks:
            if not np.allclose(tm.eps_l_noop(self.l, self.A, self.A), self.l,
            rtol=self.itr_rtol*self.check_fac, 
            atol=self.itr_atol*self.check_fac):
                print "Sanity check failed: Left eigenvector bad! Off by: " \
                       + str(la.norm(tm.eps_l_noop(self.l, self.A, self.A) - self.l))
                       
            if not np.allclose(tm.eps_r_noop(self.r, self.A, self.A), self.r,
            rtol=self.itr_rtol*self.check_fac,
            atol=self.itr_atol*self.check_fac):
                print "Sanity check failed: Right eigenvector bad! Off by: " \
                       + str(la.norm(tm.eps_r_noop(self.r, self.A, self.A) - self.r))
            
            if not np.allclose(self.l, m.H(self.l),
            rtol=self.itr_rtol*self.check_fac, 
            atol=self.itr_atol*self.check_fac):
                print "Sanity check failed: l is not hermitian!"

            if not np.allclose(self.r, m.H(self.r),
            rtol=self.itr_rtol*self.check_fac, 
            atol=self.itr_atol*self.check_fac):
                print "Sanity check failed: r is not hermitian!"
            
            if not np.all(la.eigvalsh(self.l) > 0):
                print "Sanity check failed: l is not pos. def.!"
                
            if not np.all(la.eigvalsh(self.r) > 0):
                print "Sanity check failed: r is not pos. def.!"
            
            norm = m.adot(self.l, self.r)
            if not np.allclose(norm, 1.0, atol=1E-13, rtol=0):
                print "Sanity check failed: Bad norm = " + str(norm)
    
    def restore_SCF(self, ret_g=False, zero_tol=1E-15):
        """Restores symmetric canonical form.
        
        In this canonical form, self.l == self.r and are diagonal matrices
        with the Schmidt coefficients corresponding to the half-chain
        decomposition form the diagonal entries.
        
        Parameters
        ----------
        ret_g : bool
            Whether to return the gauge-transformation matrices used.
            
        Returns
        -------
        g, g_i : ndarray
            Gauge transformation matrix g and its inverse g_i.
        """
        X, Xi = tm.herm_fac_with_inv(self.r, lower=True, zero_tol=zero_tol)
        
        Y, Yi = tm.herm_fac_with_inv(self.l, lower=False, zero_tol=zero_tol)          
            
        U, sv, Vh = la.svd(Y.dot(X))
        
        #s contains the Schmidt coefficients,
        lam = sv**2
        self.S_hc = - np.sum(lam * sp.log2(lam))
        
        S = m.simple_diag_matrix(sv, dtype=self.typ)
        Srt = S.sqrt()
        
        g = m.mmul(Srt, Vh, Xi)
        
        g_i = m.mmul(Yi, U, Srt)
        
        for s in xrange(self.q):
            self.A[s] = m.mmul(g, self.A[s], g_i)
                
        if self.sanity_checks:
            Sfull = np.asarray(S)
            
            if not np.allclose(g.dot(g_i), np.eye(self.D)):
                print "Sanity check failed! Restore_SCF, bad GT!"
            
            l = m.mmul(m.H(g_i), self.l, g_i)
            r = m.mmul(g, self.r, m.H(g))
            
            if not np.allclose(Sfull, l):
                print "Sanity check failed: Restorce_SCF, left failed!"
                
            if not np.allclose(Sfull, r):
                print "Sanity check failed: Restorce_SCF, right failed!"
                
            l = tm.eps_l_noop(Sfull, self.A, self.A)
            r = tm.eps_r_noop(Sfull, self.A, self.A)
            
            if not np.allclose(Sfull, l, rtol=self.itr_rtol*self.check_fac, 
                               atol=self.itr_atol*self.check_fac):
                print "Sanity check failed: Restorce_SCF, left bad!"
                
            if not np.allclose(Sfull, r, rtol=self.itr_rtol*self.check_fac, 
                               atol=self.itr_atol*self.check_fac):
                print "Sanity check failed: Restorce_SCF, right bad!"

        self.l = S
        self.r = S
        
        if ret_g:
            return g, g_i
        else:
            return
    
    def restore_RCF(self, ret_g=False, zero_tol=1E-15):
        """Restores right canonical form.
        
        In this form, self.r = sp.eye(self.D) and self.l is diagonal, with
        the squared Schmidt coefficients corresponding to the half-chain
        decomposition as eigenvalues.
        
        Parameters
        ----------
        ret_g : bool
            Whether to return the gauge-transformation matrices used.
            
        Returns
        -------
        g, g_i : ndarray
            Gauge transformation matrix g and its inverse g_i.
        """
        #First get G such that r = eye
        G, G_i = tm.herm_fac_with_inv(self.r, lower=True, zero_tol=zero_tol)
        #TODO: Is the new r really the identity?

        self.l = m.mmul(m.H(G), self.l, G)
        
        #Now bring l into diagonal form, trace = 1 (guaranteed by r = eye..?)
        ev, EV = la.eigh(self.l)

        G = G.dot(EV)
        G_i = m.H(EV).dot(G_i)
        
        for s in xrange(self.q):
            self.A[s] = m.mmul(G_i, self.A[s], G)
            
        #ev contains the squares of the Schmidt coefficients,
        self.S_hc = - np.sum(ev * sp.log2(ev))
        
        self.l = m.simple_diag_matrix(ev, dtype=self.typ)

        if self.sanity_checks:
            M = np.zeros_like(self.r)
            for s in xrange(self.q):
                M += m.mmul(self.A[s], m.H(self.A[s]))            
            
            self.r = m.mmul(G_i, self.r, m.H(G_i))
            
            if not np.allclose(M, self.r, 
                               rtol=self.itr_rtol*self.check_fac,
                               atol=self.itr_atol*self.check_fac):
                print "Sanity check failed: RestoreRCF, bad M."
                print "Off by: " + str(la.norm(M - self.r))
                
            if not np.allclose(self.r, np.eye(self.D),
                               rtol=self.itr_rtol*self.check_fac,
                               atol=self.itr_atol*self.check_fac):
                print "Sanity check failed: r not identity."
                print "Off by: " + str(la.norm(np.eye(self.D) - self.r))
            
            l = tm.eps_l_noop(self.l, self.A, self.A)
            r = tm.eps_r_noop(self.r, self.A, self.A)
            
            if not np.allclose(r, self.r,
                               rtol=self.itr_rtol*self.check_fac, 
                               atol=self.itr_atol*self.check_fac):
                print "Sanity check failed: Restore_RCF, bad r!"
                print "Off by: " + str(la.norm(r - self.r))

            if not np.allclose(l, self.l,
                               rtol=self.itr_rtol*self.check_fac, 
                               atol=self.itr_atol*self.check_fac):
                print "Sanity check failed: Restore_RCF, bad l!"
                print "Off by: " + str(la.norm(l - self.l))
    
        self.r = m.eyemat(self.D, dtype=self.typ)
        
        if ret_g:
            return G, G_i
        else:
            return
    
    def restore_CF(self, ret_g=False):
        """Restores canonical form.
        
        Performs self.restore_RCF() or self.restore_SCF()
        depending on self.symm_gauge.        
        """
        if self.symm_gauge:
            return self.restore_SCF(ret_g=ret_g)
        else:
            return self.restore_RCF(ret_g=ret_g)
    
    def auto_truncate(self, update=True, zero_tol=1E-15):
        new_D_l = np.count_nonzero(self.l.diag > zero_tol)
        
        if 0 < new_D_l < self.A.shape[1]:
            self.truncate(new_D_l, update=False)
            
            if update:
                self.update()
                
            return True
        else:
            return False
    
    def truncate(self, newD, update=True):
        assert newD < self.D, 'new bond-dimension must be smaller!'
        
        tmp_A = self.A
        tmp_l = self.l.diag
        
        self._init_arrays(newD, self.q)
        
        if self.symm_gauge:
            self.l = m.simple_diag_matrix(tmp_l[:self.D], dtype=self.typ)
            self.r = m.simple_diag_matrix(tmp_l[:self.D], dtype=self.typ)
            self.A = tmp_A[:, :self.D, :self.D]
        else:
            self.l = m.simple_diag_matrix(tmp_l[-self.D:], dtype=self.typ)
            self.r = m.eyemat(self.D, dtype=self.typ)
            self.A = tmp_A[:, -self.D:, -self.D:]
            
        self.l_before_CF = self.l.A
        self.r_before_CF = self.r.A

        if update:
            self.update()
    
    def calc_AA(self):
        """Calculates the products A[s] A[t] for s, t in range(self.q).
        The result is stored in self.AA.
        """
        self.AA = tm.calc_AA(self.A, self.A)
        
        
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
                print "Auto-truncated! New D: ", self.D
                self.calc_lr()
                if restore_CF_after_trunc:
                    self.restore_CF()
                
        self.calc_AA()
        
        
    def fidelity_per_site(self, other, full_output=False, left=False):
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
        
        Returns
        -------
        d : float
            The per-site fidelity.
        w : float
            The largest eigenvalue of the overlap transfer operator.
        V : ndarray
            The right (or left if left == True) eigenvector corresponding to w (if full_output == True).
        """
        if self.D == 1:
            ev = 0
            for s in xrange(self.q):
                ev += self.A[s] * other.A[s].conj()
            if left:
                ev = ev.conj()
            if full_output:
                return abs(ev), ev, sp.ones((1), dtype=self.typ)
            else:
                return abs(ev), ev
        else:
            opE = EOp(self.A, other.A, left)
            ev, eV = las.eigs(opE, which='LM', k=1, ncv=6, return_eigenvectors=full_output)
            if full_output:
                return abs(ev[0]), ev[0], eV[:, 0]
            else:
                return abs(ev[0]), ev[0]

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
        d, phi = self.fidelity_per_site(other, full_output=False, left=False)
        
        self.A *= phi.conj()
        
        return phi

    def gauge_align(self, other, tol=1E-12):
        """Gauge-align the state with another.
        
        Given two states that differ only by a gauge-transformation
        and a phase, this makes equalizes the parameter tensors by performing 
        the required transformation.
        
        Parameters
        ----------
        other : EvoMPS_MPS_Uniform
            MPS with which to calculate the per-site fidelity.
        tol : float
            Tolerance for detecting per-site fidelity != 1.
            
        Returns
        -------
        Nothing if the per-site fidelity is 1. Otherwise:
            
        g : ndarray
            The gauge-transformation matrix used.
        g_i : ndarray
            The inverse of g.
        phi : complex
            The phase factor.
        """
        d, phi, gR = self.fidelity_per_site(other, full_output=True)
        
        if abs(d - 1) > tol:
            return
            
        gR = gR.reshape(self.D, self.D)
            
        try:
            g = other.r.inv().dotleft(gR)
        except:
            g = gR.dot(m.invmh(other.r))
            
        gi = la.inv(g)
        
        for s in xrange(self.q):
            self.A[s] = phi.conj() * gi.dot(self.A[s]).dot(g)
            
        self.l = m.H(g).dot(self.l.dot(g))
        
        self.r = gi.dot(self.r.dot(m.H(gi)))
            
        return g, gi, phi
        
            
    def expect_1s(self, op):
        """Computes the expectation value of a single-site operator.
        
        The operator should be a self.q x self.q matrix or generating function 
        such that op[s, t] or op(s, t) equals <s|op|t>.
        
        The state must be up-to-date -- see self.update()!
        
        Parameters
        ----------
        op : ndarray or callable
            The operator.
            
        Returns
        -------
        expval : floating point number
            The expectation value (data type may be complex)
        """        
        if callable(op):
            op = np.vectorize(op, otypes=[np.complex128])
            op = np.fromfunction(op, (self.q, self.q))
            
        Or = tm.eps_r_op_1s(self.r, self.A, self.A, op)
        
        return m.adot(self.l, Or)
        
    def expect_1s_1s(self, op1, op2, d):
        """Computes the expectation value of two single site operators acting 
        on two different sites.
        
        The result is < op1_n op2_n+d > with the operators acting on sites
        n and n + d.
        
        See expect_1s().
        
        Requires d > 0.
        
        The state must be up-to-date -- see self.update()!
        
        Parameters
        ----------
        op1 : ndarray or callable
            The first operator, acting on the first site.
        op2 : ndarray or callable
            The second operator, acting on the second site.
        d : int
            The distance (number of sites) between the two sites acted on non-trivially.
            
        Returns
        -------
        expval : floating point number
            The expectation value (data type may be complex)
        """        
        
        assert d > 0, 'd must be greater than 1'
        
        if callable(op1):
            op1 = sp.vectorize(op1, otypes=[sp.complex128])
            op1 = sp.fromfunction(op1, (self.q, self.q))
        
        if callable(op2):
            op2 = sp.vectorize(op2, otypes=[sp.complex128])
            op2 = sp.fromfunction(op2, (self.q, self.q)) 
        
        r_n = tm.eps_r_op_1s(self.r, self.A, self.A, op2)

        for n in xrange(d - 1):
            r_n = tm.eps_r_noop(r_n, self.A, self.A)

        r_n = tm.eps_r_op_1s(r_n, self.A, self.A, op1)
         
        return m.adot(self.l, r_n)
            
    def expect_2s(self, op):
        """Computes the expectation value of a nearest-neighbour two-site operator.
        
        The operator should be a q x q x q x q array 
        such that op[s, t, u, v] = <st|op|uv> or a function of the form 
        op(s, t, u, v) = <st|op|uv>.
        
        The state must be up-to-date -- see self.update()!
        
        Parameters
        ----------
        op : ndarray or callable
            The operator array or function.
            
        Returns
        -------
        expval : floating point number
            The expectation value (data type may be complex)
        """
        if callable(op):
            op = np.vectorize(op, otypes=[np.complex128])
            op = np.fromfunction(op, (self.q, self.q, self.q, self.q))        
        
        C = tm.calc_C_mat_op_AA(op, self.AA)
        res = tm.eps_r_op_2s_C12_AA34(self.r, C, self.AA)
        
        return m.adot(self.l, res)
        
    def expect_3s(self, op, n):
        """Computes the expectation value of a nearest-neighbour three-site operator.

        The operator should be a q x q x q x q x q x q 
        array such that op[s, t, u, v, w, x] = <stu|op|vwx> 
        or a function of the form op(s, t, u, v, w, x) = <stu|op|vwx>.

        The state must be up-to-date -- see self.update()!

        Parameters
        ----------
        op : ndarray or callable
            The operator array or function.
            
        Returns
        -------
        expval : floating point number
            The expectation value (data type may be complex)
        """
        A = self.A
        AAA = tm.calc_AAA(A, A, A)

        if callable(op):
            op = sp.vectorize(op, otypes=[sp.complex128])
            op = sp.fromfunction(op, (A.shape[0], A.shape[0], A.shape[0],
                                      A.shape[0], A.shape[0], A.shape[0]))

        C = tm.calc_C_3s_mat_op_AAA(op, AAA)
        res = tm.eps_r_op_3s_C123_AAA456(self.r, C, AAA)
        return m.adot(self.l, res)
        
    def density_1s(self):
        """Returns a reduced density matrix for a single site.
        
        The site number basis is used: rho[s, t] 
        with 0 <= s, t < q.
        
        The state must be up-to-date -- see self.update()!
            
        Returns
        -------
        rho : ndarray
            Reduced density matrix in the number basis.
        """
        rho = np.empty((self.q, self.q), dtype=self.typ)
        for s in xrange(self.q):
            for t in xrange(self.q):                
                rho[s, t] = m.adot(self.l, m.mmul(self.A[t], self.r, m.H(self.A[s])))
        return rho
        
    def density_2s(self, d):
        """Returns a reduced density matrix for a pair of (seperated) sites.
        
        The site number basis is used: rho[s * q + u, t * q + v]
        with 0 <= s, t < q and 0 <= u, v < q.
        
        The state must be up-to-date -- see self.update()!
        
        Parameters
        ----------
        d : int
            The distance between the first and the second sites considered (d = n2 - n1).
            
        Returns
        -------
        rho : ndarray
            Reduced density matrix in the number basis.
        """
        rho = sp.empty((self.q * self.q, self.q * self.q), dtype=sp.complex128)
        
        for s2 in xrange(self.q):
            for t2 in xrange(self.q):
                r_n2 = m.mmul(self.A[t2], self.r, m.H(self.A[s2]))
                
                r_n = r_n2
                for n in xrange(d - 1):
                    r_n = tm.eps_r_noop(r_n, self.A, self.A)
                    
                for s1 in xrange(self.q):
                    for t1 in xrange(self.q):
                        r_n1 = m.mmul(self.A[t1], r_n, m.H(self.A[s1]))
                        tmp = m.adot(self.l, r_n1)
                        rho[s1 * self.q + s2, t1 * self.q + t2] = tmp
        return rho        
        
    def apply_op_1s(self, op, do_update=True):
        """Applies a single-site operator to all sites.
        
        This applies the product (or string) of a single-site operator o_n over 
        all sites so that the new state |Psi'> = ...o_(n-1) o_n o_(n+1)... |Psi>.
        
        By default, this performs self.update(), which also restores
        state normalization.
        
        Parameters
        ----------
        op : ndarray or callable
            The single-site operator. See self.expect_1s().
        do_update : bool
            Whether to update after applying the operator.
        """
        if callable(op):
            op = np.vectorize(op, otypes=[np.complex128])
            op = np.fromfunction(op, (self.q, self.q))
            
        newA = sp.zeros_like(self.A)
        
        for s in xrange(self.q):
            for t in xrange(self.q):
                newA[s] += self.A[t] * op[s, t]
                
        self.A = newA
        
        if do_update:
            self.update()
    
    def expect_string_per_site_1s(self, op):
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
        
        Aop = np.tensordot(op, self.A, axes=([1],[0]))
        
        if self.D == 1:
            ev = 0
            for s in xrange(self.q):
                ev += Aop[s] * self.A[s].conj()
            return ev
        else:            
            opE = EOp(Aop, self.A, False)
            ev = las.eigs(opE, v0=np.asarray(self.r), which='LM', k=1, ncv=6)
            return ev[0]                
                
    def set_q(self, newq):
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
        
        self._init_arrays(self.D, newq) 
        
        self.l = oldl
        self.r = oldr
        
        self.A.fill(0)
        if self.q > oldq:
            self.A[:oldq, :, :] = oldA
        else:
            self.A[:] = oldA[:self.q, :, :]
        
            
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
        
        oldl = np.asarray(self.l)
        oldr = np.asarray(self.r)
        
        self._init_arrays(newD, self.q)
        
        realnorm = la.norm(oldA.real.ravel())
        imagnorm = la.norm(oldA.imag.ravel())
        realfac = (realnorm / oldA.size) * refac
        imagfac = (imagnorm / oldA.size) * imfac
#        m.randomize_cmplx(newA[:, self.D:, self.D:], a=-fac, b=fac)
        m.randomize_cmplx(self.A[:, :oldD, oldD:], a=0, b=realfac, aj=0, bj=imagfac)
        m.randomize_cmplx(self.A[:, oldD:, :oldD], a=0, b=realfac, aj=0, bj=imagfac)
        self.A[:, oldD:, oldD:] = 0 #for nearest-neighbour hamiltonian

#        self.A[:, :oldD, oldD:] = oldA[:, :, :(newD - oldD)]
#        self.A[:, oldD:, :oldD] = oldA[:, :(newD - oldD), :]
        self.A[:, :oldD, :oldD] = oldA

        self.l[:oldD, :oldD] = oldl
        self.l[:oldD, oldD:].fill(la.norm(oldl) / oldD**2)
        self.l[oldD:, :oldD].fill(la.norm(oldl) / oldD**2)
        self.l[oldD:, oldD:].fill(la.norm(oldl) / oldD**2)
        
        self.r[:oldD, :oldD] = oldr
        self.r[oldD:, :oldD].fill(la.norm(oldr) / oldD**2)
        self.r[:oldD, oldD:].fill(la.norm(oldr) / oldD**2)
        self.r[oldD:, oldD:].fill(la.norm(oldr) / oldD**2)
        