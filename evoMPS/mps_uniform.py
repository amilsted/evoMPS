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
    def __init__(self, tdvp, A1, A2, left):
        self.tdvp = tdvp
        self.A1 = A1
        self.A2 = A2
        
        self.D = tdvp.D
        
        self.shape = (self.D**2, self.D**2)
        
        self.dtype = np.dtype(tdvp.typ)
        
        self.out = np.empty_like(tdvp.r)
        
        self.calls = 0
        
        if left:
            self.eps = tm.eps_l_noop_inplace
        else:
            self.eps = tm.eps_r_noop_inplace
    
    def matvec(self, v):
        x = v.reshape((self.D, self.D))

        Ex = self.eps(x, self.A1, self.A2, self.out)
        
        self.calls += 1
        
        return Ex.ravel()

class EvoMPS_MPS_Uniform(object):   
        
    def __init__(self, D, q, dtype=None):
        
        self.odr = 'C' 
        
        if dtype is None:
            self.typ = np.complex128
        
        self.itr_rtol = 1E-13
        self.itr_atol = 1E-14
        
        self.itr_l = 0
        self.itr_r = 0
        
        self.pow_itr_max = 2000
        self.ev_use_arpack = True
                
        self.symm_gauge = False
        
        self.sanity_checks = False
        self.check_fac = 50
        
        self.userdata = None        
        
        self.eps = np.finfo(self.typ).eps
                
        self._init_arrays(D, q)        
                    
        self.randomize()

    def randomize(self, fac=0.5):
        m.randomize_cmplx(self.A, a=-fac, b=fac)
    
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
                        tol=1E-14, ncv=6):
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
        opE = EOp(self, A1, A2, calc_l)
        x *= n / norm(x.ravel())
        try:
            ev, eV = las.eigs(opE, which='LM', k=1, v0=x.ravel(), tol=tol, ncv=ncv)
            conv = True
        except las.ArpackNoConvergence:
            print "Reset! (l? %s)" % str(calc_l)
            ev, eV = las.eigs(opE, which='LM', k=1, tol=tol, ncv=ncv)
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
        
    def calc_E_gap(self, tol=1E-6, ncv=10):
        """
        Calculates the spectral gap of E by calculating the second-largest eigenvalue
        
        This is the correlation length.
        """
        opE = EOp(self, self.A, self.A, False)
        
        r = np.asarray(self.r)
        
        ev = las.eigs(opE, which='LM', k=2, v0=r.ravel(), tol=tol, ncv=ncv,
                          return_eigenvectors=False)
                          
        ev1 = abs(ev.max())
        ev2 = abs(ev.min())
        return ((ev1 - ev2) / ev1)
                
    def _calc_lr(self, x, tmp, calc_l=False, A1=None, A2=None, rescale=True,
                 max_itr=1000, rtol=1E-14, atol=1E-14):
        """Power iteration to obtain eigenvector corresponding to largest
           eigenvalue.
           
           The contents of the starting vector x is modifed.
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
        tmp = np.empty_like(self.tmp)
        
        #Make sure...
        self.l_before_CF = np.asarray(self.l_before_CF)
        self.r_before_CF = np.asarray(self.r_before_CF)
        
        if self.ev_use_arpack:
            self.l, self.conv_l, self.itr_l = self._calc_lr_ARPACK(self.l_before_CF, tmp,
                                                   calc_l=True,
                                                   tol=self.itr_rtol)
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
                                                   tol=self.itr_rtol)
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
    
    def restore_SCF(self):
        X = la.cholesky(self.r, lower=True)
        Y = la.cholesky(self.l, lower=False)
        
        U, sv, Vh = la.svd(Y.dot(X))
        
        #s contains the Schmidt coefficients,
        lam = sv**2
        self.S_hc = - np.sum(lam * sp.log2(lam))
        
        S = m.simple_diag_matrix(sv, dtype=self.typ)
        Srt = S.sqrt()
        
        g = m.mmul(Srt, Vh, m.invtr(X, lower=True))
        
        g_i = m.mmul(m.invtr(Y, lower=False), U, Srt)
        
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

    def restore_CF(self, ret_g=False, zero_tol=1E-15):
        if self.symm_gauge:
            self.restore_SCF()
        else:

            #First get G such that r = eye
            try:
                G = la.cholesky(self.r, lower=True)
                G_i = m.invtr(G, lower=True)
                new_D_r = self.D
            except la.LinAlgError:
                ev, EV = la.eigh(self.r)
                new_D_r = np.count_nonzero(ev > zero_tol)
                ev_sq = sp.sqrt(ev[-new_D_r:])
                ev_sq_i = m.simple_diag_matrix(
                    np.append(np.zeros(self.A.shape[1] - new_D_r), 1. / ev_sq))
                ev_sq = m.simple_diag_matrix(
                    np.append(np.zeros(self.A.shape[1] - new_D_r), ev_sq))
                G = ev_sq.dot_left(EV)
                G_i = ev_sq_i.dot(m.H(EV))
                print ev
                print new_D_r
                #exit()

            self.l = m.mmul(m.H(G), self.l, G)

            #Now bring l into diagonal form, trace = 1 (guaranteed by r = eye..?)
            ev, EV = la.eigh(self.l)

            new_D_l = np.count_nonzero(ev > zero_tol)
            if new_D_l != self.A.shape[1]:
                print ev
                print new_D_l

            G = G.dot(EV)
            G_i = m.H(EV).dot(G_i)

            for s in xrange(self.q):
                self.A[s] = m.mmul(G_i, self.A[s], G)

            #ev contains the squares of the Schmidt coefficients,
            self.S_hc = - np.sum(ev[-new_D_l:] * sp.log2(ev[-new_D_l:]))

            if min(new_D_r, new_D_l) != self.A.shape[1]:
                self.D = min(new_D_r, new_D_l)
                print self.D
                tmp_A = self.A
                self._init_arrays(self.D, self.q)
                self.l = m.simple_diag_matrix(ev[-self.D:], dtype=self.typ)
                self.A = tmp_A[..., -self.D:, -self.D:]
                print self.A.shape
            else:
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
    
    def calc_AA(self):
        self.AA = tm.calc_AA(self.A, self.A)
        
        
    def update(self, restore_CF=True):
        self.calc_lr()
        if restore_CF:
            self.restore_CF()
        self.calc_AA()
        
        
    def fidelity_per_site(self, other, full_output=False, left=False):
        """
          Returns the per-site fidelity.
          
          Also returns the largest eigenvalue "w" of the overlap transfer
          operator, as well as the corresponding eigenvector "V" in the
          matrix representation.
          
          If the fidelity is 1:
          
            A^s = w g A'^s g^-1      (with g = V r'^-1)
            
        """
        if self.D == 1:
            ev = 0
            for s in xrange(self.q):
                ev += self.A[s] * other.A[s].conj()
            if left:
                ev = ev.conj()
            if full_output:
                return abs(ev), ev, 1
            else:
                return abs(ev)
        else:
            opE = EOp(other, self.A, other.A, left)
            ev, eV = las.eigs(opE, which='LM', k=1, ncv=6, return_eigenvectors=True)
            if full_output:
                return abs(ev[0]), ev[0], eV[:, 0].reshape(self.D, self.D)
            else:
                return abs(ev[0])

    def phase_align(self, other, tol=1E-12, left=False):
        d, phi, gR = self.fidelity_per_site(other, full_output=True, left=left)
        
        self.A *= phi.conj()
        
        return phi, gR

    def gauge_align(self, other, tol=1E-12):
        d, phi, gR = self.fidelity_per_site(other, full_output=True)
        
        if abs(d - 1) > tol:
            return False
            
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
        if callable(op):
            op = np.vectorize(op, otypes=[np.complex128])
            op = np.fromfunction(op, (self.q, self.q, self.q, self.q))        
        
        res = tm.eps_r_op_2s_AA_func_op(self.r, self.AA, self.AA, op)
        
        return m.adot(self.l, res)
        
    def density_1s(self):
        rho = np.empty((self.q, self.q), dtype=self.typ)
        for s in xrange(self.q):
            for t in xrange(self.q):                
                rho[s, t] = m.adot(self.l, m.mmul(self.A[t], self.r, m.H(self.A[s])))
        return rho
        
    def apply_op_1s(self, op):
        if callable(op):
            op = np.vectorize(op, otypes=[np.complex128])
            op = np.fromfunction(op, (self.q, self.q))
            
        newA = sp.zeros_like(self.A)
        
        for s in xrange(self.q):
            for t in xrange(self.q):
                newA[s] += self.A[t] * op[s, t]
                
        self.A = newA
            
            
    def expand_q(self, newq):
        if newq < self.q:
            return False
        
        oldq = self.q
        oldA = self.A
        
        oldl = self.l
        oldr = self.r
        
        self._init_arrays(self.D, newq) 
        
        self.l = oldl
        self.r = oldr
        
        self.A.fill(0)
        self.A[:oldq, :, :] = oldA
        
    def shrink_q(self, newq):
        if newq > self.q:
            return False
        
        oldA = self.A
        
        oldl = self.l
        oldr = self.r
        
        self._init_arrays(self.D, newq) 
        
        self.l = oldl
        self.r = oldr
        
        self.A.fill(0)
        self.A[:] = oldA[:newq, :, :]
            
    def expand_D(self, newD, refac=100, imfac=0):
        """Expands the bond dimension in a simple way.
        
        New matrix entries are (mostly) randomized.
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
        
    def fuzz_state(self, f=1.0):
        norm = la.norm(self.A)
        fac = f*(norm / (self.q * self.D**2))        
        
        R = np.empty_like(self.A)
        m.randomize_cmplx(R, -fac/2.0, fac/2.0)
        
        self.A += R