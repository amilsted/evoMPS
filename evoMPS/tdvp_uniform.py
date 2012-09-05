# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 17:29:27 2011

@author: Ashley Milsted

"""
import numpy as np
import scipy as sp
import scipy.linalg as la
import scipy.sparse.linalg as las
import scipy.optimize as opti
import nullspace as ns
import matmul as m
import math as ma

try:
    import tdvp_common as tc
    import allclose as ac
except ImportError:
    tc = None
    ac = None
    print "Warning! Cython version of Calc_C was not available. Performance may suffer for large q."
        

class EOp:
    def __init__(self, tdvp, A1, A2, left):
        self.tdvp = tdvp
        self.A1 = A1
        self.A2 = A2
        
        self.D = tdvp.D
        
        self.shape = (self.D**2, self.D**2)
        
        self.dtype = np.dtype(tdvp.typ)
        
        self.out = np.empty_like(tdvp.r)
        
        if left:
            self.eps = tdvp._eps_l_noop_dense
        else:
            self.eps = tdvp._eps_r_noop_dense
    
    def matvec(self, v):
        x = v.reshape((self.D, self.D))

        Ex = self.eps(x, self.A1, self.A2, self.out)
        
        return Ex.ravel()
    

class PPInvOp:    
    def __init__(self, tdvp, p, left, pseudo, A1, A2, r):
        self.tdvp = tdvp
        self.A1 = A1
        self.A2 = A2
        self.l = tdvp.l
        self.r = r
        self.p = p
        self.left = left
        self.pseudo = pseudo
        
        self.D = tdvp.D
        
        self.shape = (self.D**2, self.D**2)
        
        self.dtype = np.dtype(tdvp.typ)
        
        self.out = np.empty_like(self.l)
    
    def matvec(self, v):
        x = v.reshape((self.D, self.D))
        
        if self.left: #Multiplying from the left, but x is a col. vector, so use mat_dagger
            Ehx = self.tdvp._eps_l_noop_dense(x, self.A1, self.A2, self.out)
            if self.pseudo:
                QEQhx = Ehx - self.l * m.adot(self.r, x)
                res = x - sp.exp(-1.j * self.p) * QEQhx
            else:
                res = x - sp.exp(-1.j * self.p) * Ehx
        else:
            Ex = self.tdvp._eps_r_noop_dense(x, self.A1, self.A2, self.out)
            if self.pseudo:
                QEQx = Ex - self.r * m.adot(self.l, x)
                res = x - sp.exp(1.j * self.p) * QEQx
            else:
                res = x - sp.exp(1.j * self.p) * Ex
        
        return res.ravel()
        
class Excite_H_Op:
    def __init__(self, tdvp, donor, p):
        self.donor = donor
        self.tdvp = tdvp
        self.p = p
        
        self.D = tdvp.D
        self.q = tdvp.q
        
        d = (self.q - 1) * self.D**2
        self.shape = (d, d)
        
        self.dtype = np.dtype(tdvp.typ)
        
        self.prereq = (tdvp.calc_BHB_prereq(donor))
        
        self.calls = 0
    
    def matvec(self, v):
        x = v.reshape((self.D, (self.q - 1)*self.D))
        
        self.calls += 1
        print "Calls: %u" % self.calls
        
        res = self.tdvp.calc_BHB(x, self.p, self.donor, *self.prereq)
        
        return res.ravel()

class EvoMPS_TDVP_Uniform:
    odr = 'C'
    typ = np.complex128
        
    def __init__(self, D, q):
        
        self.itr_rtol = 1E-13
        self.itr_atol = 1E-14
        
        self.pow_itr_max = 1000
        
        self.h_nn = None    
        self.h_nn_cptr = None
        self.h_nn_mat = None
        
        self.symm_gauge = False
        
        self.sanity_checks = False
        self.check_fac = 50
        
        self.userdata = None        
        
        self.eps = np.finfo(self.typ).eps
        
        self.eta = 0
        
        self._init_arrays(D, q)
        
        #self.A.fill(0)
        #for s in xrange(q):
        #    self.A[s] = np.eye(D)
            
        self.randomize()

    def randomize(self, fac=0.5):
        m.randomize_cmplx(self.A, a=-fac, b=fac)
    
    def _init_arrays(self, D, q):
        self.D = D
        self.q = q
        
        self.A = np.zeros((q, D, D), dtype=self.typ, order=self.odr)
        self.AA = np.zeros((q, q, D, D), dtype=self.typ, order=self.odr)
        
        self.C = np.zeros((q, q, D, D), dtype=self.typ, order=self.odr)
        
        self.K = np.ones_like(self.A[0])
        self.K_left = None
        
        self.l = np.ones_like(self.A[0])
        self.r = np.ones_like(self.A[0])
        self.conv_l = True
        self.conv_r = True
        
        self.tmp = np.zeros_like(self.A[0])
           
    def _eps_r_noop_dense(self, x, A1, A2, out):
        """The right epsilon map, optimized for efficiency.
        """
        out.fill(0)
        dot = np.dot
        for s in xrange(self.q):
            out += dot(A1[s], dot(x, A2[s].conj().T))
        
        return out
        
    def eps_r(self, x, A1=None, A2=None, op=None, out=None):
        """Implements the right epsilon map
        
        FIXME: Ref.
        
        Parameters
        ----------
        op : function
            The single-site operator to use.
        out : ndarray
            A matrix to hold the result (with the same dimensions as r).
        x : ndarray
            The argument matrix.
    
        Returns
        -------
        res : ndarray
            The resulting matrix.
        """
           
        if A1 is None:
            A1 = self.A
        if A2 is None:
            A2 = self.A

        if out is None:
            out = np.zeros((A1.shape[1], A2.shape[1]), dtype=self.typ)
        else:
            out.fill(0.)
            
        if op is None:
            for s in xrange(self.q):
                out += m.mmul(A1[s], x, m.H(A2[s]))
        else:
            for s in xrange(self.q):
                for t in xrange(self.q):
                    o_st = op(s, t)
                    if o_st != 0.:
                        tmp = m.mmul(A1[t], x, m.H(A2[s]))
                        tmp *= o_st
                        out += tmp
        return out
        
    def _eps_l_noop_dense(self, x, A1, A2, out):
        """The left epsilon map, optimized for efficiency.
        """
        out.fill(0)
        dot = np.dot
        for s in xrange(self.q):
            out += dot(A1[s].conj().T, dot(x, A2[s]))
            
        return out
        
    def eps_l(self, x, A1=None, A2=None, out=None):
        if out is None:
            out = np.zeros_like(self.A[0])
        else:
            out.fill(0.)
            
        if A1 is None:
            A1 = self.A
        if A2 is None:
            A2 = self.A
            
        for s in xrange(self.q):
            out += m.mmul(m.H(A1[s]), x, A2[s])
            
        return out
        
    def calc_AA(self):
        dot = np.dot
        A = self.A
        AA = self.AA
        for s in xrange(self.q):
            for t in xrange(self.q):
                AA[s, t] = dot(A[s], A[t])
        
        #Note: This could be cythonized, calling BLAS from C    
        
        #This works too: (just for reference)
        #AA = np.array([dot(A[s], A[t]) for s in xrange(self.q) for t in xrange(self.q)])
        #self.AA = AA.reshape(self.q, self.q, self.D, self.D)
        
    def eps_r_2s(self, x, op, A1=None, A2=None, A3=None, A4=None, C34=None, C=None):
        if A1 is None:
            A1 = self.A
        if A2 is None:
            A2 = self.A
        if A3 is None:
            A3 = self.A
        if A4 is None:
            A4 = self.A
            
        if op is self.h_nn and C is None:
            C = self.C
            op = None
            
        res = np.zeros((A1.shape[1], A3.shape[1]), dtype=self.typ)
        zeros_like = np.zeros_like
        zeros = np.zeros
        
        if (A1 is self.A) and (A2 is self.A) and (A3 is self.A) and (A4 is self.A):
            if op is None and not C is None:
                for s in xrange(self.q):
                    for t in xrange(self.q):
                        res += C[s, t].dot(x.dot(m.H(self.AA[s, t])))
            else:
                for u in xrange(self.q):
                    for v in xrange(self.q):
                        subres = zeros_like(A1[0])
                        for s in xrange(self.q):
                            for t in xrange(self.q):
                                opval = op(u, v, s, t)
                                if opval != 0:
                                    subres += opval * self.AA[s, t]
                        res += subres.dot(x.dot(m.H(self.AA[u, v])))
        elif (A1 is self.A) and (A2 is self.A):
            if op is None and not C is None:
                for s in xrange(self.q):
                    for t in xrange(self.q):
                        AAstH = m.H(A3[s].dot(A4[t]))
                        res += C[s, t].dot(x.dot(AAstH))
            elif op is None and not C34 is None:
                for s in xrange(self.q):
                    for t in xrange(self.q):
                        res += self.AA[s, t].dot(x.dot(m.H(C34[s, t])))
            else:
                for u in xrange(self.q):
                    for v in xrange(self.q):
                        subres = zeros_like(self.A[0])
                        for s in xrange(self.q):
                            for t in xrange(self.q):
                                opval = op(u, v, s, t)
                                if opval != 0:
                                    subres += opval * self.AA[s, t]
                        AAuvH = m.H(A3[u].dot(A4[v]))
                        res += subres.dot(x.dot(AAuvH))
        elif (op is None and not C34 is None) or (A3 is self.A) and (A4 is self.A):
            if op is None and not C34 is None:
                for s in xrange(self.q):
                    for t in xrange(self.q):
                        res += A1[s].dot(A2[t]).dot(x.dot(m.H(C34[s, t])))
            elif op is None and not C is None:
                for s in xrange(self.q):
                    for t in xrange(self.q):
                        res += A1[s].dot(A2[t]).dot(x.dot(m.H(C[s, t])))
            else:
                for u in xrange(self.q):
                    for v in xrange(self.q):
                        subres = zeros((A1.shape[1], A2.shape[2]), dtype=self.typ)
                        for s in xrange(self.q):
                            for t in xrange(self.q):
                                opval = op(u, v, s, t)
                                if opval != 0:
                                    subres += opval * A1[s].dot(A2[t])
                        res += subres.dot(x.dot(m.H(self.AA[u, v])))
        else:
            for u in xrange(self.q):
                for v in xrange(self.q):
                    subres = zeros((A1.shape[1], A2.shape[2]), dtype=self.typ)
                    for s in xrange(self.q):
                        for t in xrange(self.q):
                            opval = op(u, v, s, t)
                            if opval != 0:
                                subres += opval * A1[s].dot(A2[t])
                    AAuvH = m.H(A3[u].dot(A4[v]))
                    res += subres.dot(x.dot(AAuvH))
                    
        return res

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
        print "Left ok?: " + str(np.allclose(self.eps_l(self.l), self.l))
        print "Right ok?: " + str(np.allclose(self.eps_r(self.r), self.r))
        
    def _calc_lr(self, x, eps, tmp, A1=None, A2=None, rescale=True,
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
        except ValueError:
            norm = np.linalg.norm
            
        try:
            allclose = ac.allclose_mat
        except:
            allclose = np.allclose
            print "Falling back to numpy allclose()!"
        
        n = x.size #we will scale x so that stuff doesn't get too small
        
        x *= n / norm(x.ravel())
        tmp[:] = x
        for i in xrange(max_itr):
            x[:] = tmp
            eps(x, A1, A2, tmp)            
            ev_mag = norm(tmp.ravel()) / n
            ev = tmp.mean() / x.mean()
            tmp *= (1 / ev_mag)
#            if norm((tmp - x).ravel()) < tol_vec * self.D**2:
            if allclose(tmp, x, rtol, atol):                
                #print (i, ev, ev_mag, norm((tmp - x).ravel())/norm(x.ravel()), atol, rtol)
                x[:] = tmp
                break            
#        else:
#            print (i, ev, ev_mag, norm((tmp - x).ravel())/norm(x.ravel()), atol, rtol)
            
        ev = abs(ev)

#        opE = EOp(self, A1, A2, x is self.l)
#        x *= n / norm(x.ravel())
#        ev, eV = las.eigs(opE, which='LM', k=10, v0=x.ravel())
#        print ev * ev.conj()
#        ev = ev[0]
#        x[:] = eV[:, 0].reshape(self.D, self.D)        
#        
#        #remove any additional phase factor
#        x *= 1 / sp.sqrt((x / m.H(x)).mean())
#        #x[:] = sp.sqrt(x * m.H(x))
#        print norm((x - m.H(x)).ravel())
#        
#        x *= n / norm(x.ravel())
#        
#        i = 0
#        ev = abs(ev)
#        if self.sanity_checks:
#            y = eps(x, A1, A2, tmp)
#            diff = norm(x.ravel()-y.ravel()/ev)/norm(x.ravel())
#            if diff > 1E-10:
#                print "Sanity check failed: Bad eigenevector, off by %g" % diff        
                    
        if rescale and not abs(ev - 1) < atol:
            A1 *= 1 / sp.sqrt(ev)
            if self.sanity_checks:
                if not A1 is A2:
                    print "Sanity check failed: Re-scaling with A1 <> A2!"
                ev = eps(x, A1, A2, tmp).mean() / x.mean()
                if not abs(ev - 1) < atol:
                    print "Sanity check failed: Largest ev after re-scale = " + str(ev)
        
        return x, i < max_itr - 1, i
    
    def calc_lr(self, reset=False, auto_reset=True):
        tmp = np.empty_like(self.tmp)

        self.l = np.asarray(self.l)

        self.r = np.asarray(self.r)
        
        self.conv_l = False
        if reset:
            i = 1
        else:            
            i = 0
        while not self.conv_l and i < 2:
            if i > 0:
                print "RESETTING l!"
                self.l.fill(1)
            self.l, self.conv_l, self.itr_l = self._calc_lr(self.l, 
                                                        self._eps_l_noop_dense, 
                                                        tmp, 
                                                        max_itr=self.pow_itr_max,
                                                        rtol=self.itr_atol, 
                                                        atol=self.itr_atol)
            i += 1
            if not auto_reset:
                break
        
        self.conv_r = False
        if reset:
            i = 1
        else:            
            i = 0
        while not self.conv_r and i < 2:
            if i > 0:
                print "RESETTING r!"
                self.r.fill(1)        
            self.r, self.conv_r, self.itr_r = self._calc_lr(self.r, 
                                                        self._eps_r_noop_dense, 
                                                        tmp, 
                                                        max_itr=self.pow_itr_max,
                                                        rtol=self.itr_atol, 
                                                        atol=self.itr_atol)
            i += 1
            if not auto_reset:
                break
            
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
            if not np.allclose(self.eps_l(self.l), self.l,
            rtol=self.itr_rtol*self.check_fac, 
            atol=self.itr_atol*self.check_fac):
                print "Sanity check failed: Left eigenvector bad! Off by: " \
                       + str(la.norm(self.eps_l(self.l) - self.l))
                       
            if not np.allclose(self.eps_r(self.r), self.r,
            rtol=self.itr_rtol*self.check_fac,
            atol=self.itr_atol*self.check_fac):
                print "Sanity check failed: Right eigenvector bad! Off by: " \
                       + str(la.norm(self.eps_r(self.r) - self.r))
            
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
                
            l = self.eps_l(Sfull)
            r = self.eps_r(Sfull)
            
            if not np.allclose(Sfull, l, rtol=self.itr_rtol*self.check_fac, 
                               atol=self.itr_atol*self.check_fac):
                print "Sanity check failed: Restorce_SCF, left bad!"
                
            if not np.allclose(Sfull, r, rtol=self.itr_rtol*self.check_fac, 
                               atol=self.itr_atol*self.check_fac):
                print "Sanity check failed: Restorce_SCF, right bad!"

        self.l = S
        self.r = S
    
    def restore_CF(self, ret_g=False):
        if self.symm_gauge:
            self.restore_SCF()
        else:
            #First get G such that r = eye
            G = la.cholesky(self.r, lower=True)
            G_i = m.invtr(G, lower=True)

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
                
                l = self.eps_l(self.l)
                r = self.eps_r(self.r)
                
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
    
    def gen_h_matrix(self):
        """Generates a matrix form for h_nn, which can speed up parts of the
        algorithm by avoiding excess loops and python calls.
        """
        self.h_nn_mat = sp.zeros((self.q, self.q, self.q, self.q), dtype=sp.complex128)
        q = self.q
        for u in xrange(q):
            for v in xrange(q):
                for s in xrange(q):
                    for t in xrange(q):
                        self.h_nn_mat[s, t, u, v] = self.h_nn(s, t, u, v)    
    
    def calc_C(self):
        if not tc is None and not self.h_nn_cptr is None:
            self.C = tc.calc_C(self.AA, self.h_nn_cptr, self.C)
        elif not self.h_nn_mat is None:
            self.C[:] = sp.tensordot(self.h_nn_mat, self.AA, ((2, 3), (0, 1)))
        else:
            self.C.fill(0)
            
            for u in xrange(self.q): #ndindex is just too slow..
                for v in xrange(self.q):
                    for s in xrange(self.q):
                        for t in xrange(self.q):
                            h = self.h_nn(s, t, u, v) #for large q, this executes a lot..
                            if h != 0:
                                self.C[s, t] += h * self.AA[u, v]
    
    def calc_PPinv(self, x, p=0, out=None, left=False, A1=None, A2=None, r=None, pseudo=True):
        if out is None:
            out = np.ones_like(self.A[0])
            
        if A1 is None:
            A1 = self.A
            
        if A2 is None:
            A2 = self.A
            
        if r is None:
            r = self.r
        
        op = PPInvOp(self, p, left, pseudo, A1, A2, r)
        
        res = out.ravel()
        x = x.ravel()
        
        res, info = las.bicgstab(op, x, x0=res, maxiter=2000, 
                                 tol=self.itr_rtol) #tol: norm( b - A*x ) / norm( b )
        
        if info > 0:
            print "Warning: Did not converge on solution for ppinv!"
        
        #Test
        if self.sanity_checks:
            RHS_test = op.matvec(res)
            d = la.norm(RHS_test - x) / la.norm(x)
            if not d < self.itr_rtol*self.check_fac:
                print "Sanity check failed: Bad ppinv solution! Off by: " + str(
                        d)
        
        res = res.reshape((self.D, self.D))
            
        if False and self.sanity_checks and self.D < 16:
            pinvE = self.pinvE_brute(p, A1, A2, r, pseudo)
            
            if left:
                res_brute = (x.reshape((1, self.D**2)).conj().dot(pinvE)).ravel().conj().reshape((self.D, self.D))
                #res_brute = (pinvE.T.dot(x)).reshape((self.D, self.D))
            else:
                res_brute = pinvE.dot(x).reshape((self.D, self.D))
            
            if not np.allclose(res, res_brute):
                print "Sanity Fail in calc_PPinv (left: %s): Bad brute check! Off by: %g" % (str(left), la.norm(res - res_brute))
        
        out[:] = res
        
        return out
        
    def calc_K(self):
        Hr = np.zeros_like(self.A[0])
        
        for s in xrange(self.q):
            for t in xrange(self.q):
                Hr += m.mmul(self.C[s, t], self.r, m.H(self.AA[s, t]))
        
        self.h = m.adot(self.l, Hr)
        
        QHr = Hr - self.r * self.h
        
        self.calc_PPinv(QHr, out=self.K)
        
        if self.sanity_checks:
            Ex = self.eps_r(self.K)
            QEQ = Ex - self.r * m.adot(self.l, self.K)
            res = self.K - QEQ
            if not np.allclose(res, QHr):
                print "Sanity check failed: Bad K!"
                print "Off by: " + str(la.norm(res - QHr))
        
    def calc_K_l(self):
        lH = np.zeros_like(self.A[0])
        
        for s in xrange(self.q):
            for t in xrange(self.q):
                lH += m.mmul(m.H(self.AA[s, t]), self.l, self.C[s, t])
        
        h = m.adot(self.r, lH)
        
        lHQ = lH - self.l * h
        
        self.K_left = self.calc_PPinv(lHQ, left=True, out=self.K_left)
        
        if self.sanity_checks:
            xE = self.eps_l(self.K_left)
            QEQ = xE - self.l * m.adot(self.r, self.K_left)
            res = self.K_left - QEQ
            if not np.allclose(res, lHQ):
                print "Sanity check failed: Bad K_left!"
                print "Off by: " + str(la.norm(res - lHQ))
        
        return self.K_left, h
            
    def calc_Vsh(self, r_sqrt):
        R = np.zeros((self.D, self.q, self.D), dtype=self.typ, order='C')
        
        for s in xrange(self.q):
            R[:,s,:] = m.mmul(r_sqrt, m.H(self.A[s]))
        
        R = R.reshape((self.q * self.D, self.D))
        
        Vconj = ns.nullspace_qr(m.H(R)).T
        #R can be pretty huge for large q and D. The decomp. can take a long time...

        if self.sanity_checks:
            if not np.allclose(np.dot(Vconj, m.H(Vconj)), np.eye(self.q*self.D - self.D)):
                print "Sanity check failed: V . H(V) not eye!"
            if not np.allclose(np.dot(Vconj.conj(), R), 0):
                print "Sanity check failed: V . R not zero!"
        Vconj = Vconj.reshape(((self.q - 1) * self.D, self.D, self.q))
        
        #prepare for using V[s] and already take the adjoint, since we use it more often
        Vsh = Vconj.T
        Vsh = np.asarray(Vsh, order='C')

        return Vsh
        
    def calc_x(self, l_sqrt, l_sqrt_i, r_sqrt, r_sqrt_i, Vsh, out=None):
        if out is None:
            out = np.zeros((self.D, (self.q - 1) * self.D), dtype=self.typ, 
                           order=self.odr)
        
        tmp = np.zeros_like(out)
        for s in xrange(self.q):
            tmp2 = m.mmul(self.A[s], self.K)
            for t in xrange(self.q):
                tmp2 += m.mmul(self.C[s, t], self.r, m.H(self.A[t]))
            tmp += m.mmul(tmp2, r_sqrt_i, Vsh[s])
        out += l_sqrt.dot(tmp)
        
        tmp.fill(0)
        for s in xrange(self.q):
            tmp2.fill(0)
            for t in xrange(self.q):
                tmp2 += m.mmul(m.H(self.A[t]), self.l, self.C[t, s])
            tmp += m.mmul(tmp2, r_sqrt, Vsh[s])
        out += l_sqrt_i.dot(tmp)
        
        return out
        
    def get_B_from_x(self, x, Vsh, l_sqrt_i, r_sqrt_i, out=None):
        if out is None:
            out = np.zeros_like(self.A)
            
        for s in xrange(self.q):
            out[s] = m.mmul(l_sqrt_i, x, m.H(Vsh[s]), r_sqrt_i)
            
        return out
        
    def calc_l_r_roots(self):
        try:
            self.l_sqrt = self.l.sqrt()
            self.l_sqrt_i = self.l_sqrt.inv()
        except AttributeError:
            self.l_sqrt, evd = m.sqrtmh(self.l, ret_evd=True)
            self.l_sqrt_i = m.invmh(self.l_sqrt, evd=evd)
            
        try:
            self.r_sqrt = self.r.sqrt()
            self.r_sqrt_i = self.r_sqrt.inv()
        except AttributeError:
            self.r_sqrt, evd = m.sqrtmh(self.r, ret_evd=True)
            self.r_sqrt_i = m.invmh(self.r_sqrt, evd=evd)
        
        if self.sanity_checks:
            if not np.allclose(self.l_sqrt.dot(self.l_sqrt), self.l):
                print "Sanity check failed: l_sqrt is bad!"
            if not np.allclose(self.l_sqrt.dot(self.l_sqrt_i), np.eye(self.D)):
                print "Sanity check failed: l_sqrt_i is bad!"
            if not np.allclose(self.r_sqrt.dot(self.r_sqrt), self.r):
                print "Sanity check failed: r_sqrt is bad!"
            if (not np.allclose(self.r_sqrt.dot(self.r_sqrt_i), np.eye(self.D))):
                print "Sanity check failed: r_sqrt_i is bad!"
        
    def calc_B(self, set_eta=True):
        self.calc_l_r_roots()
                
        self.Vsh = self.calc_Vsh(self.r_sqrt)
        
        self.x = self.calc_x(self.l_sqrt, self.l_sqrt_i, self.r_sqrt, 
                        self.r_sqrt_i, self.Vsh)
        
        if set_eta:
            self.eta = sp.sqrt(m.adot(self.x, self.x))
        
        B = self.get_B_from_x(self.x, self.Vsh, self.l_sqrt_i, self.r_sqrt_i)
        
        if self.sanity_checks:
            #Test gauge-fixing:
            tst = np.zeros_like(self.A[0])
            for s in xrange(self.q):
                tst += m.mmul(B[s], self.r, m.H(self.A[s]))
            if not np.allclose(tst, 0):
                print "Sanity check failed: Gauge-fixing violation!"

        return B
        
    def update(self, restore_CF=True):
        self.calc_lr()
        if restore_CF:
            self.restore_CF()
        self.calc_AA()
        self.calc_C()
        self.calc_K()
        
    def take_step(self, dtau, B=None):
        if B is None:
            B = self.calc_B()
        
        self.A += -dtau * B
            
    def take_step_RK4(self, dtau, B_i=None):
        def update():
            self.calc_lr()
            #self.restore_CF() #this really messes things up...
            self.calc_AA()
            self.calc_C()
            self.calc_K()            

        A0 = self.A.copy()
            
        B_fin = np.empty_like(self.A)

        if not B_i is None:
            B = B_i
        else:
            B = self.calc_B() #k1
        B_fin = B
        self.A = A0 - dtau/2 * B
        
        update()
        
        B = self.calc_B(set_eta=False) #k2                
        self.A = A0 - dtau/2 * B
        B_fin += 2 * B         
            
        update()
            
        B = self.calc_B(set_eta=False) #k3                
        self.A = A0 - dtau * B
        B_fin += 2 * B

        update()
        
        B = self.calc_B(set_eta=False) #k4
        B_fin += B
            
        self.A = A0 - dtau /6 * B_fin
            
    def pinvE_brute(self, p, A1, A2, r, pseudo=True):
        E = np.zeros((self.D**2, self.D**2), dtype=self.typ)

        for s in xrange(self.q):
            E += np.kron(A1[s], A2[s].conj())
        
        l = np.asarray(self.l)
        r = np.asarray(r)
        
        if pseudo:
            QEQ = E - r.reshape((self.D**2, 1)).dot(l.reshape((1, self.D**2)).conj())
        else:
            QEQ = E
        
        EyemE = np.eye(self.D**2, dtype=self.typ) - sp.exp(1.j * p) * QEQ
        
        return la.inv(EyemE)
        
    def calc_BHB_prereq(self, donor):
        l = self.l
        r_ = donor.r
        r__sqrt = donor.r_sqrt
        r__sqrt_i = donor.r_sqrt_i
        A = self.A
        A_ = donor.A
        AA_ = donor.AA
                    
#        def h_nn(s,t,u,v):
#            h = self.h_nn(s,t,u,v)
#            if s == u and t == v:
#                h -= self.h.real
#            return h
#        h_nn = np.vectorize(h_nn, otypes=[self.typ])
#        h_nn_mat = np.fromfunction(h_nn, (self.q, self.q, self.q, self.q), dtype=self.typ)
        eyed = np.eye(self.q**2)
        eyed = eyed.reshape((self.q, self.q, self.q, self.q))
        h_nn_mat = self.h_nn_mat - self.h.real * eyed
        def h_nn(s,t,u,v):
            return h_nn_mat[s,t,u,v]
            
        V_ = sp.zeros((donor.Vsh.shape[0], donor.Vsh.shape[2], donor.Vsh.shape[1]), dtype=self.typ)
        for s in xrange(donor.q):
            V_[s] = m.H(donor.Vsh[s])
        
        Vri_ = sp.zeros_like(V_)
        for s in xrange(donor.q):
            Vri_[s] = r__sqrt_i.dot_left(V_[s])
            
        Vr_ = sp.zeros_like(V_)
        for s in xrange(donor.q):
            Vr_[s] = r__sqrt.dot_left(V_[s])
            
        C_AhlA = np.empty_like(self.C)
        for u in xrange(self.q):
            for s in xrange(self.q):
                C_AhlA[u, s] = m.H(A[u]).dot(l.dot(A[s]))
        C_AhlA = sp.tensordot(h_nn_mat, C_AhlA, ((2, 0), (0, 1)))
        
        C_A_Vrh_ = np.empty((self.q, self.q, A_.shape[1], Vr_.shape[1]), dtype=self.typ)
        for t in xrange(self.q):
            for v in xrange(self.q):
                C_A_Vrh_[t, v] = A_[t].dot(m.H(Vr_[v]))
        C_A_Vrh_ = sp.tensordot(h_nn_mat, C_A_Vrh_, ((1, 3), (0, 1)))
                
        C_Vri_A_ = np.empty((self.q, self.q, Vri_.shape[1], A_.shape[2]), dtype=self.typ)
        for s in xrange(self.q):
            for t in xrange(self.q):
                C_Vri_A_[s, t] = Vri_[s].dot(A_[t])
        C_Vri_A_ = sp.tensordot(h_nn_mat, C_Vri_A_, ((2, 3), (0, 1)))
        
        C = sp.tensordot(h_nn_mat, self.AA, ((2, 3), (0, 1)))

        C_ = sp.tensordot(h_nn_mat, AA_, ((2, 3), (0, 1)))
        
        rhs10 = donor.eps_r_2s(r_, op=None, A3=Vri_, C34=C_Vri_A_)
        
        return h_nn, h_nn_mat, C, C_, V_, Vr_, Vri_, C_Vri_A_, C_AhlA, C_A_Vrh_, rhs10
            
    def calc_BHB(self, x, p, donor, h_nn, h_nn_mat, C, C_, V_, Vr_, Vri_, C_Vri_A_, C_AhlA, C_A_Vrh_, rhs10): 
        """For a good approx. ground state, H should be Hermitian pos. semi-def.
        """        
        A = self.A
        A_ = donor.A
        
        l = self.l
        r_ = donor.r
        
        l_sqrt = self.l_sqrt
        l_sqrt_i = self.l_sqrt_i
        
        r__sqrt = donor.r_sqrt
        r__sqrt_i = donor.r_sqrt_i
        
        K__r = donor.K
        K_l = self.K_left
        
        pseudo = donor is self
        
        B = donor.get_B_from_x(x, donor.Vsh, l_sqrt_i, r__sqrt_i)
        
        if self.sanity_checks:
            tst = donor.eps_r(r_, A1=B)
            if not np.allclose(tst, 0):
                print "Sanity check failed: Gauge-fixing violation!"

        if self.sanity_checks:
            B2 = np.zeros_like(B)
            for s in xrange(self.q):
                B2[s] = l_sqrt_i.dot(x.dot(Vri_[s]))
            if not sp.allclose(B, B2, rtol=self.itr_rtol*self.check_fac,
                               atol=self.itr_atol*self.check_fac):
                print "Sanity Fail in calc_BHB! Bad Vri!"
            
        y = self.eps_l(l, A1=B)  
        
        if pseudo:
            y = y - m.adot(r_, y) * l #should just = y due to gauge-fixing
        M = self.calc_PPinv(y, p=-p, left=True, A1=A_, r=r_, pseudo=pseudo)
        #print m.adot(r, M)
        if self.sanity_checks:
            y2 = M - sp.exp(+1.j * p) * self.eps_l(M, A1=A_)
            if not sp.allclose(y, y2):
                print "Sanity Fail in calc_BHB! Bad M. Off by: %g" % (la.norm((y - y2).ravel()) / la.norm(y.ravel()))
        Mh = m.H(M)

        res = l_sqrt.dot(
               donor.eps_r_2s(r_, op=None, A1=B, A3=Vri_, C34=C_Vri_A_) #1 OK
               + sp.exp(+1.j * p) * self.eps_r_2s(r_, op=None, A2=B, A3=Vri_, A4=A_, C34=C_Vri_A_) #3 OK with 4
              )
        
        res += sp.exp(-1.j * p) * l_sqrt_i.dot(Mh.dot(rhs10)) #10
        
        exp = sp.exp
        H = m.H
        for s in xrange(self.q):
            for t in xrange(self.q):
                res += l_sqrt_i.dot(C_AhlA[s, t].dot(B[s])).dot(H(Vr_[t])) #2 OK
                res += exp(-1.j * p) * l_sqrt_i.dot(H(A[t]).dot(l.dot(B[s]))).dot(C_A_Vrh_[s, t]) #4 OK with 3
                res += exp(-2.j * p) * l_sqrt_i.dot(H(A[s]).dot(Mh.dot(C_[s, t]))).dot(H(Vr_[t])) #12
        
        res += l_sqrt.dot(self.eps_r(K__r, A1=B, A2=Vri_)) #5 OK
        
        res += l_sqrt_i.dot(m.H(K_l).dot(self.eps_r(r__sqrt, A1=B, A2=V_))) #6
        
        res += sp.exp(-1.j * p) * l_sqrt_i.dot(Mh.dot(donor.eps_r(K__r, A2=Vri_))) #8
        
        y1 = sp.exp(+1.j * p) * donor.eps_r(K__r, A1=B) #7
        y2 = sp.exp(+1.j * p) * donor.eps_r_2s(r_, op=None, A1=B, C34=C_) #9
        y3 = sp.exp(+2.j * p) * donor.eps_r_2s(r_, op=None, A1=A, A2=B, C34=C_) #11
        
        y = y1 + y2 + y3
        if pseudo:
            y = y - m.adot(l, y) * r_
        y_pi = self.calc_PPinv(y, p=p, A2=A_, r=r_, pseudo=pseudo)
        #print m.adot(l, y_pi)
        if self.sanity_checks:
            y2 = y_pi - sp.exp(+1.j * p) * self.eps_r(y_pi, A2=A_)
            if not sp.allclose(y, y2):
                print "Sanity Fail in calc_BHB! Bad x_pi. Off by: %g" % (la.norm((y - y2).ravel()) / la.norm(y.ravel()))
        
        res += l_sqrt.dot(self.eps_r(y_pi, A2=Vri_))
        
        if self.sanity_checks:
            expval = m.adot(x, res) / m.adot(x, x)
            #print "expval = " + str(expval)
            if expval < 0:
                print "Sanity Fail in calc_BHB! H is not pos. semi-definite (" + str(expval) + ")"
            if not abs(expval.imag) < 1E-9:
                print "Sanity Fail in calc_BHB! H is not Hermitian (" + str(expval) + ")"
        
        return res
    
    def _prepare_excite_op_top_triv(self, p):
        self.calc_K_l()
        self.calc_l_r_roots()
        self.Vsh = self.calc_Vsh(self.r_sqrt)
        
        op = Excite_H_Op(self, self, p)

        return op        
    
    def excite_top_triv(self, p, k=6, tol=0, max_itr=None, v0=None,
                        which='SM', return_eigenvectors=False):
        self.calc_K_l()
        self.calc_l_r_roots()
        self.Vsh = self.calc_Vsh(self.r_sqrt)
        
        op = Excite_H_Op(self, self, p)
        res = las.eigsh(op, which=which, k=k, v0=v0,
                         return_eigenvectors=return_eigenvectors, 
                         maxiter=max_itr, tol=tol)
                          
        return res
    
    def excite_top_triv_brute(self, p):
        self.calc_K_l()
        self.calc_l_r_roots()
        self.Vsh = self.calc_Vsh(self.r_sqrt)
        
        op = Excite_H_Op(self, self, p)
        
        x = np.empty(((self.q - 1)*self.D**2), dtype=self.typ)
        y = np.empty_like(x)
        
        H = np.zeros((x.shape[0], x.shape[0]), dtype=self.typ)
        
        #Only fill the lower triangle
        for i in xrange(x.shape[0]):
            x.fill(0)
            x[i] = 1
            for j in xrange(i + 1):
                y.fill(0)
                y[j] = 1
                
                H[i, j] = x.dot(op.matvec(y))

        #print np.allclose(H, m.H(H))
               
        return la.eigvalsh(H)

    def excite_top_nontriv(self, donor, p, k=6, tol=0, max_itr=None, v0=None,
                           which='SM', return_eigenvectors=False):
        self.gen_h_matrix()
        donor.gen_h_matrix()
        self.calc_lr()
        self.restore_CF()
        donor.calc_lr()
        donor.restore_CF()
        
        #Phase-alignment
        opE = EOp(donor, self.A, donor.A, False)
        ev = las.eigs(opE, which='LM', k=1)
        donor.A *= ev[0] / abs(ev[0])
        
        self.update()
        donor.update()

        self.calc_K_l()
        self.calc_l_r_roots()
        donor.calc_l_r_roots()
        donor.Vsh = donor.calc_Vsh(donor.r_sqrt)
        
        op = Excite_H_Op(self, donor, p)
        res = las.eigsh(op, which=which, k=k, v0=v0,
                        return_eigenvectors=return_eigenvectors, 
                        maxiter=max_itr, tol=tol)
                
        return res
            
    def find_min_h(self, B, dtau_init, tol=5E-2):
        dtau = dtau_init
        d = 1.0
        #dh_dtau = 0
        
        tau_min = 0
        
        A0 = self.A.copy()                
        AA0 = self.AA.copy()
        C0 = self.C.copy()
        
        try:
            l0 = self.l
            self.l = self.l.A
        except:
            l0 = self.l.copy()
            pass
        
        try:
            r0 = self.r
            self.r = self.r.A
        except:
            r0 = self.r.copy()
            pass
        
        h_min = self.h
        A_min = self.A.copy()
        
        #l_min = np.array(self.l, copy=True)
        #r_min = np.array(self.r, copy=True)
        
        itr = 0
        while itr == 0 or itr < 30 and (abs(dtau) / tau_min > tol or tau_min == 0):
            itr += 1
            for s in xrange(self.q):
                self.A[s] = A_min[s] -d * dtau * B[s]
            
            #self.l[:] = l_min
            #self.r[:] = r_min
            
            self.calc_lr(reset=True)
            self.calc_AA()
            self.calc_C()
            
            self.h = self.expect_2s(self.h_nn)
            
            #dh_dtau = d * (self.h - h_min) / dtau
            
            print (tau_min + dtau, self.h.real, tau_min)
            
            if self.h.real < h_min.real:
                #self.restore_CF()
                h_min = self.h
                A_min[:] = self.A
                #l_min[:] = self.l
                #r_min[:] = self.r
                
                dtau = min(dtau * 1.1, dtau_init * 10)
                
#                if tau + d * dtau > 0:
                tau_min += d * dtau
#                else:
#                    d = -1.0
#                    dtau = tau
            else:
                d *= -1.0
                dtau = dtau / 2.0
                
#                if tau + d * dtau < 0:
#                    dtau = tau #only happens if dtau is -ive
                
        #Must restore everything needed for take_step
        self.A = A0
        self.l = l0
        self.r = r0
        self.AA = AA0
        self.C = C0
        
        return tau_min
        
    def find_min_h_brent(self, B, dtau_init, tol=5E-2, skipIfLower=False, 
                         taus=[], hs=[], trybracket=True):
        def f(tau, *args):
            if tau == 0:
                return self.h.real
                
            try:
                i = taus.index(tau)
                return hs[i]
            except ValueError:
                for s in xrange(self.q):
                    self.A[s] = A0[s] - tau * B[s]
                
                self.l.fill(1)
                self.r.fill(1)
                self.calc_lr(reset=False, auto_reset=False)
                self.calc_AA()
                self.calc_C()
                
                h = self.expect_2s(self.h_nn)
                
                print (tau, h.real)
                
                res = h.real
                
                taus.append(tau)
                hs.append(res)
                
                return res
        
        A0 = self.A.copy()                
        AA0 = self.AA.copy()
        C0 = self.C.copy()
        
        try:
            l0 = self.l
            self.l = self.l.A
        except:
            l0 = self.l.copy()
            pass
        
        try:
            r0 = self.r
            self.r = self.r.A
        except:
            r0 = self.r.copy()
            pass
        
        if skipIfLower:
            if f(dtau_init) < self.h.real:
                return dtau_init
        
        fb_brack = (dtau_init * 0.9, dtau_init * 1.1)
        if trybracket:
            brack = (dtau_init * 0.1, dtau_init, dtau_init * 2.0)
        else:
            brack = fb_brack
                
        try:
            tau_opt, h_min, itr, calls = opti.brent(f, 
                                                    brack=brack, 
                                                    tol=tol,
                                                    maxiter=20,
                                                    full_output=True)
        except ValueError:
            print "Bracketing attempt failed..."
            tau_opt, h_min, itr, calls = opti.brent(f, 
                                                    brack=fb_brack, 
                                                    tol=tol,
                                                    maxiter=20,
                                                    full_output=True)
        
        #Must restore everything needed for take_step
        self.A = A0
        self.l = l0
        self.r = r0
        self.AA = AA0
        self.C = C0
        
        return tau_opt, h_min
        
    def step_reduces_h(self, B, dtau):
        A0 = self.A.copy()
        AA0 = self.AA.copy()
        C0 = self.C.copy()
        
        try:
            l0 = self.l
            self.l = self.l.A
        except:
            l0 = self.l.copy()
            pass
        
        try:
            r0 = self.r
            self.r = self.r.A
        except:
            r0 = self.r.copy()
            pass
        
        for s in xrange(self.q):
            self.A[s] = A0[s] - dtau * B[s]
        
        self.calc_lr()
        self.calc_AA()
        self.calc_C()
        
        h = self.expect_2s(self.h_nn)
        
        #Must restore everything needed for take_step
        self.A = A0
        self.l = l0
        self.r = r0
        self.AA = AA0
        self.C = C0
        
        return h.real < self.h.real, h

    def calc_B_CG(self, B_CG_0, x_0, eta_0, dtau_init, reset=False,
                 skipIfLower=False, brent=True):
        B = self.calc_B()
        eta = self.eta
        x = self.x
        
        if reset:
            beta = 0.
            print "RESET CG"
            
            B_CG = B
        else:
            beta = (eta**2) / eta_0**2
            
            #xy = m.adot(x_0, x)
            #betaPR = (eta**2 - xy) / eta_0**2
        
            print "BetaFR = " + str(beta)
            #print "BetaPR = " + str(betaPR)
        
            beta = max(0, beta.real)
        
            B_CG = B + beta * B_CG_0
        
        taus = []
        hs = []
        
        if skipIfLower:
            stepRedH, h = self.step_reduces_h(B_CG, dtau_init)
            taus.append(dtau_init)
            hs.append(h)
        
        if skipIfLower and stepRedH:
            tau = self.find_min_h(B_CG, dtau_init)
        else:
            if brent:
                tau, h_min = self.find_min_h_brent(B_CG, dtau_init, taus=taus, hs=hs,
                                            trybracket=False)
            else:
                tau = self.find_min_h(B_CG, dtau_init)
        
        if tau < 0:
            print "RESET due to negative dtau!"
            B_CG = B
            tau, h_min = self.find_min_h_brent(B_CG, dtau_init)
            
        if self.h.real < h_min:
            print "RESET due to energy rise!"
            B_CG = B
            tau, h_min = self.find_min_h_brent(B_CG, dtau_init)
        
        if self.h.real < h_min:
            print "RESET FAILED: Setting tau=0!"
            tau = 0
        
        return B_CG, B, x, eta, tau
        
            
    def expect_1s(self, op):
        Or = self.eps_r(self.r, op=op)
        
        return m.adot(self.l, Or)
            
    def expect_2s(self, op):
        res = self.eps_r_2s(self.r, op)
        
        return m.adot(self.l, res)
        
    def density_1s(self):
        rho = np.empty((self.q, self.q), dtype=self.typ)
        for s in xrange(self.q):
            for t in xrange(self.q):                
                rho[s, t] = m.adot(self.l, m.mmul(self.A[t], self.r, m.H(self.A[s])))
        return rho
        
    def apply_op_1s(self, o):
        newA = sp.zeros_like(self.A)
        
        for s in xrange(self.q):
            for t in xrange(self.q):
                newA[s] += self.A[t] * o(s, t)
                
        self.A = newA
            
    def save_state(self, file, userdata=None):
        if userdata is None:
            userdata = self.userdata

        l = np.asarray(self.l)
        r = np.asarray(self.r)
            
        tosave = np.empty((5), dtype=np.ndarray)
        tosave[0] = self.A
        tosave[1] = l
        tosave[2] = r
        tosave[3] = self.K
        tosave[4] = np.asarray(userdata)
        
        np.save(file, tosave)
        
    def load_state(self, file, expand=False, expand_q=False):
        state = np.load(file)
        
        newA = state[0]
        newl = state[1]
        newr = state[2]
        newK = state[3]
        if state.shape[0] > 4:
            self.userdata = state[4]
        
        if (newA.shape == self.A.shape):
            self.A[:] = newA
            self.K[:] = newK

            self.l = np.asarray(newl)
            self.r = np.asarray(newr)
                
            return True
        elif expand and (len(newA.shape) == 3) and (newA.shape[0] == 
        self.A.shape[0]) and (newA.shape[1] == newA.shape[2]) and (newA.shape[1]
        <= self.A.shape[1]):
            newD = self.D
            savedD = newA.shape[1]
            self._init_arrays(savedD, self.q)
            self.A[:] = newA
            self.l = newl
            self.r = newr
            self.K[:] = newK
            self.expand_D(newD)
            print "EXPANDED!"
        elif expand_q and (len(newA.shape) == 3) and (newA.shape[0] <= 
        self.A.shape[0]) and (newA.shape[1] == newA.shape[2]) and (newA.shape[1]
        == self.A.shape[1]):
            newQ = self.q
            savedQ = newA.shape[0]
            self._init_arrays(self.D, savedQ)
            self.A[:] = newA
            self.l = newl
            self.r = newr
            self.K[:] = newK
            self.expand_q(newQ)
            print "EXPANDED in q!"
        else:
            return False
            
    def expand_q(self, newq):
        if newq < self.q:
            return False
        
        oldq = self.q
        oldA = self.A
        oldK = self.K
        
        oldl = self.l
        oldr = self.r
        
        self._init_arrays(self.D, newq) 
        
        self.l = oldl
        self.r = oldr
        self.K = oldK
        
        self.A.fill(0)
        self.A[:oldq, :, :] = oldA
            
    def expand_D(self, newD):
        """Expands the bond dimension in a simple way.
        
        New matrix entries are (mostly) randomized.
        """
        if newD < self.D:
            return False
        
        oldD = self.D
        oldA = self.A
        oldK = self.K
        
        oldl = np.asarray(self.l)
        oldr = np.asarray(self.r)
        
        self._init_arrays(newD, self.q)
        
        realnorm = la.norm(oldA.real)
        imagnorm = la.norm(oldA.imag)
        realfac = (realnorm / (self.q * oldD**2)) * 100.
        imagfac = (imagnorm / (self.q * oldD**2)) * 0.
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
        
        self.K[:oldD, :oldD] = oldK
        self.K[oldD:, :oldD].fill(la.norm(oldK) / oldD**2)
        self.K[:oldD, oldD:].fill(la.norm(oldK) / oldD**2)
        self.K[oldD:, oldD:].fill(la.norm(oldK) / oldD**2)
        
    def fuzz_state(self, f=1.0):
        norm = la.norm(self.A)
        fac = f*(norm / (self.q * self.D**2))        
        
        R = np.empty_like(self.A)
        m.randomize_cmplx(R, -fac/2.0, fac/2.0)
        
        self.A += R