# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 17:29:27 2011

@author: Ashley Milsted

TODO:
    - Think about whether it is better to move back to RCF before
      applying B, since B does *right* gauge fixing. Then we would
      switch back to symm. form before calculating the next B.
       0. Restore RCF if needed
       1. RCF to SCF (with 4th root of l etc.)
       2. Calc B
       3. SCF to RCF (can do this quickly?)
       4. Apply B (take step)
    - Also, find out what happens in theory when this is *not* done...
      Should cause the gauge choice to drift... right?

"""
import scipy as sp
import scipy.linalg as la
import scipy.sparse.linalg as las
import scipy.optimize as op
import nullspace as ns
import matmul as m

import pyximport; pyximport.install()
import tdvp_common as tc

def adot(a, b, tmp=None):
    """
    Calculates the scalar product for the ancilla, expecting
    the arguments in matrix form.
    Equivalent to trace(dot(H(a), b))
    """
    if tmp is None:
        return sp.inner(a.ravel().conj(), b.ravel())
    else:
        return sp.inner(sp.conj(a, out=tmp).ravel(), b.ravel())
        
#This class allows us to use scipy's bicgstab implementation
class PPInv_op:
    tdvp = None
    l = None
    r = None
    A = None
    p = 0
    
    shape = (0)
    
    dtype = None
    
    def __init__(self, tdvp, p=0):
        self.tdvp = tdvp
        self.l = tdvp.l
        self.r = tdvp.r
        self.p = 0
        
        self.D = tdvp.D
        
        self.shape = (self.D**2, self.D**2)
        
        self.dtype = tdvp.typ
    
    def matvec(self, v):
        x = v.reshape((self.D, self.D))
        
        Ex = self.tdvp.EpsR(x)
            
        QEQ = Ex - self.r * adot(self.l, x)
        
        if not self.p == 0:
            QEQ *= sp.exp(1.j * self.p)
        
        res = x - QEQ
            
        return res.reshape((self.D**2))

class H_tangeant_op:
    tdvp = None
    ppinvop = None
    p = 0
    
    def __init__(self, tdvp, p):
        self.tdvp = tdvp
        self.p = p
        self.ppinvop = PPInv_op(tdvp, p)
        
        self.shape = (tdvp.D**2, tdvp.D**2)
        self.dtype = tdvp.typ        
        
    def matvec(self, v):
        x = v.reshape((self.tdvp.D, self.tdvp.D * (self.tdvp.q)))
        
        self.tdvp.Calc_BHB(x)                
        
class evoMPS_TDVP_Uniform:
    odr = 'C'
    typ = sp.complex128
    eps = 0
    
    itr_rtol = 1E-13
    itr_atol = 1E-14
    
    h_nn = None    
    h_nn_cptr = None
    
    symm_gauge = False
    
    sanity_checks = False
    check_fac = 50
    
    userdata = None
    
    def __init__(self, D, q):
        self.eps = sp.finfo(self.typ).eps
        
        self.eta = 0
        
        self._init_arrays(D, q)
        
        #self.A.fill(0)
        #for s in xrange(q):
        #    self.A[s] = sp.eye(D)
            
        self.Randomize()

    def Randomize(self, fac=0.5):
        m.randomize_cmplx(self.A, a=-fac, b=fac)
    
    def _init_arrays(self, D, q):
        self.D = D
        self.q = q
        
        self.A = sp.empty((q, D, D), dtype=self.typ, order=self.odr)
        
        self.C = sp.empty((q, q, D, D), dtype=self.typ, order=self.odr)
        
        self.K = sp.ones_like(self.A[0])
        
        self.l = sp.ones_like(self.A[0])
        self.r = sp.ones_like(self.A[0])
        self.conv_l = True
        self.conv_r = True
        
        self.tmp = sp.empty_like(self.A[0])
           
    def EpsR(self, x, A1=None, A2=None, op=None, out=None):
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
        if out is None:
            out = sp.zeros_like(self.r)
        else:
            out.fill(0.)
            
        if A1 is None:
            A1 = self.A
        if A2 is None:
            A2 = self.A
            
        if op is None:
            for s in xrange(self.q):
                out += m.matmul(self.tmp, A1[s], x, m.H(A2[s]))
        else:
            for (s, t) in sp.ndindex(self.q, self.q):
                o_st = op(s, t)
                if o_st != 0.:
                    m.matmul(self.tmp, A1[t], x, m.H(A2[s]))
                    self.tmp *= o_st
                    out += self.tmp
        return out
        
    def EpsL(self, x, out=None):
        if out is None:
            out = sp.zeros_like(self.l)
        else:
            out.fill(0.)
            
        for s in xrange(self.q):
            out += m.matmul(self.tmp, m.H(self.A[s]), x, self.A[s])        
            
        return out
        
    def EpsR_2(self, x, op, A1=None, A2=None, A3=None, A4=None):
        AAuvH = sp.empty_like(self.A[0])
        res = sp.zeros_like(self.r)
        
        if A1 is None:
            A1 = self.A
        if A2 is None:
            A2 = self.A
        if A3 is None:
            A3 = self.A
        if A4 is None:
            A4 = self.A
        
        for u in xrange(self.q):
            for v in xrange(self.q):
                m.matmul(AAuvH, A3[u], A4[v])
                AAuvH = m.H(AAuvH, out=AAuvH)
                for s in xrange(self.q):
                    for t in xrange(self.q):
                        res += op(u, v, s, t) * m.matmul(None, A1[s], 
                                                         A2[t], x, AAuvH)
        return res

    def _Calc_lr_brute(self):
        E = sp.zeros((self.D**2, self.D**2), dtype=self.typ, order='C')
        
        for s in xrange(self.q):
            E += sp.kron(self.A[s], self.A[s].conj())
            
        ev, eVL, eVR = la.eig(E, left=True, right=True)
        
        i = sp.argmax(ev)
        
        self.A *= 1 / sp.sqrt(ev[i])        
        
        self.l = eVL[:,i].reshape((self.D, self.D))
        self.r = eVR[:,i].reshape((self.D, self.D))
        
        norm = adot(self.l, self.r, tmp=self.tmp)
        self.l *= 1 / sp.sqrt(norm)
        self.r *= 1 / sp.sqrt(norm)        
        
        print "Sledgehammer:"
        print "Left ok?: " + str(sp.allclose(self.EpsL(self.l), self.l))
        print "Right ok?: " + str(sp.allclose(self.EpsR(self.r), self.r))
        
    def _Calc_lr(self, x, e, tmp, max_itr=1000, rtol=1E-14, atol=1E-14):        
        for i in xrange(max_itr):
            e(x, out=tmp)
            ev = la.norm(tmp)
            tmp *= (1 / ev)
            if sp.allclose(tmp, x, rtol=rtol, atol=atol):
                x[:] = tmp
                break
            x[:] = tmp
        
        #re-scale
        if not sp.allclose(ev, 1.0, rtol=rtol, atol=atol):
            self.A *= 1 / sp.sqrt(ev)
            ev = la.norm(e(x, out=tmp))
        
        return i < max_itr - 1, i
    
    def Calc_lr(self, renorm=True, force_r_CF=False):        
        tmp = sp.empty_like(self.tmp)
        
        self.conv_l, self.itr_l = self._Calc_lr(self.l, self.EpsL, tmp, 
                                                rtol=self.itr_rtol, 
                                                atol=self.itr_atol)
        
        self.conv_r, self.itr_r = self._Calc_lr(self.r, self.EpsR, tmp, 
                                                rtol=self.itr_rtol, 
                                                atol=self.itr_atol)
        #normalize eigenvectors:

        if self.symm_gauge and not force_r_CF:
            norm = adot(self.l, self.r, tmp=tmp).real
            itr = 0 
            while not sp.allclose(norm, 1, atol=1E-13, rtol=0) and itr < 10:
                self.l *= 1 / sp.sqrt(norm)
                self.r *= 1 / sp.sqrt(norm)
                
                norm = adot(self.l, self.r, tmp=tmp).real
                
                itr += 1
                
            if itr == 10:
                print "Warning: Max. iterations reached during normalization!"
        else:
            fac = self.D / sp.trace(self.r)
            self.l *= 1 / fac
            self.r *= fac

            norm = adot(self.l, self.r, tmp=tmp).real
            itr = 0 
            while not sp.allclose(norm, 1, atol=1E-13, rtol=0) and itr < 10:
                self.l *= 1 / norm
                norm = adot(self.l, self.r, tmp=tmp).real
                itr += 1
                
            if itr == 10:
                print "Warning: Max. iterations reached during normalization!"

        if self.sanity_checks:
            if not sp.allclose(self.EpsL(self.l), self.l,
            rtol=self.itr_rtol*self.check_fac, 
            atol=self.itr_atol*self.check_fac):
                print "Sanity check failed: Left eigenvector bad! Off by: " \
                       + str(la.norm(self.EpsL(self.l) - self.l))
                       
            if not sp.allclose(self.EpsR(self.r), self.r,
            rtol=self.itr_rtol*self.check_fac,
            atol=self.itr_atol*self.check_fac):
                print "Sanity check failed: Right eigenvector bad! Off by: " \
                       + str(la.norm(self.EpsR(self.r) - self.r))
            
            if not sp.allclose(self.l, m.H(self.l),
            rtol=self.itr_rtol*self.check_fac, 
            atol=self.itr_atol*self.check_fac):
                print "Sanity check failed: l is not hermitian!"

            if not sp.allclose(self.r, m.H(self.r),
            rtol=self.itr_rtol*self.check_fac, 
            atol=self.itr_atol*self.check_fac):
                print "Sanity check failed: r is not hermitian!"
            
            if not sp.all(la.eigvalsh(self.l) > 0):
                print "Sanity check failed: l is not pos. def.!"
                
            if not sp.all(la.eigvalsh(self.r) > 0):
                print "Sanity check failed: r is not pos. def.!"
            
            norm = adot(self.l, self.r, tmp=tmp)
            if not sp.allclose(norm, 1.0, atol=1E-13, rtol=0):
                print "Sanity check failed: Bad norm = " + str(norm)
    
    def Restore_SCF(self):
        X = la.cholesky(self.l, lower=True)
        Y = la.cholesky(self.l, lower=False)
        
        U, s, Vh = la.svd(sp.dot(Y, X))
        
        S = la.diagsvd(s, self.D, self.D)
        Srt = la.diagsvd(sp.sqrt(s), self.D, self.D)
        
        g = m.matmul(None, Srt, Vh, m.invtr(X, lower=True))
        
        g_i = m.matmul(None, m.invtr(Y, lower=False), U, Srt)
        
        for s in xrange(self.q):
            m.matmul(self.A[s], g, self.A[s], g_i)
                
        if self.sanity_checks:
            l = m.matmul(None, m.H(g_i), self.l, g_i)
            r = m.matmul(None, g, self.l, m.H(g))
            
            if not sp.allclose(S, l):
                print "Sanity check failed: Restorce_SCF, left failed!"
                
            if not sp.allclose(S, r):
                print "Sanity check failed: Restorce_SCF, right failed!"

        self.l[:] = S
        self.r[:] = self.l
        
        self.Calc_lr()
    
    def RCF_to_SCF(self):
        sqrt_l = m.sqrtmh(self.l)

        G = m.sqrtmh(sqrt_l)
        G_i = la.inv(G)
        
        if self.sanity_checks and not sp.allclose(
            m.matmul(None, G_i, G_i, G_i, G_i, self.l), sp.eye(self.D)):
                print "Sanity check failed: 4th root of l is bad!"
        
        for s in xrange(self.q):
            m.matmul(self.A[s], G, self.A[s], G_i)
            
        self.l[:] = sqrt_l
        self.r[:] = self.l
        
        self.Calc_lr()
        
        if self.sanity_checks:
            if not sp.allclose(self.l, self.r):
                print "Sanity check failed: Could not achieve S-CF."        
    
    def Restore_CF(self, force_r_CF=False):
        if self.symm_gauge and not force_r_CF:
            self.Restore_SCF()
        else:
            M = sp.zeros_like(self.r)
            for s in xrange(self.q):
                M += m.matmul(None, self.A[s], m.H(self.A[s]))
            
            G = m.H(la.cholesky(self.r))
            G_i = m.invtr(G, lower=True)
            
            for s in xrange(self.q):
                m.matmul(self.A[s], G_i, self.A[s], G)
                
            m.matmul(self.l, m.H(G), self.l, G)
            self.r[:] = sp.eye(self.D) #this will be corrected in Calc_lr if needed
            #m.matmul(self.r, G_i, self.r, m.H(G_i))
                
            self.Calc_lr(force_r_CF=True)
            
            if self.sanity_checks:
                M.fill(0)
                for s in xrange(self.q):
                    M += m.matmul(None, self.A[s], m.H(self.A[s]))            
                    
                if not sp.allclose(M, sp.eye(M.shape[0])) or not sp.allclose(self.r,
                sp.eye(self.D)):
                    print "Sanity check failed: Could not achieve R-CF."

#        if self.symm_gauge and not force_r_CF:
#            self.RCF_to_SCF()
    
    def Calc_C(self):
        if not self.h_nn_cptr is None:
            self.C = tc.calc_C(self.A, self.A, self.h_nn_cptr, self.C)
        else:
            self.C.fill(0)
            
            AA = sp.empty_like(self.A[0])
            
            for u in xrange(self.q): #ndindex is just too slow..
                for v in xrange(self.q):
                    m.matmul(AA, self.A[u], self.A[v])
                    for s in xrange(self.q):
                        for t in xrange(self.q):
                            h = self.h_nn(s, t, u, v) #for large q, this executes a lot..
                            if h != 0:
                                self.C[s, t] += h * AA
    
    def Calc_PPinv(self, x, p=0, out=None):
        op = PPInv_op(self, p)
        
        res = out.reshape((self.D**2))
        x = x.reshape((self.D**2))
        
        res, info = las.bicgstab(op, x, x0=res, maxiter=1000, 
                                    tol=self.itr_rtol)
        
        if info > 0:
            print "Warning: Did not converge on solution for ppinv!"
        
        #Test
        if self.sanity_checks:
            RHS_test = op.matvec(res)
            if not sp.allclose(RHS_test, x, rtol=self.itr_rtol*self.check_fac,
                                atol=self.itr_atol*self.check_fac):
                print "Sanity check failed: Bad ppinv solution! Off by: " + str(
                        la.norm(RHS_test - x))
        
        out[:] = res.reshape((self.D, self.D))
        
        return out
        
    def Calc_K(self):
        Hr = sp.zeros_like(self.A[0])
        
        for s in xrange(self.q):
            for t in xrange(self.q):
                Hr += m.matmul(None, self.C[s, t], self.r, m.H(self.A[t]), 
                               m.H(self.A[s]))
        
        self.h = adot(self.l, Hr, tmp=self.tmp)
        
        QHr = Hr - self.r * self.h
        
        self.Calc_PPinv(QHr, out=self.K)
            
    def Calc_Vsh(self, r_sqrt):
        R = sp.zeros((self.D, self.q, self.D), dtype=self.typ, order='C')
        
        for s in xrange(self.q):
            R[:,s,:] = m.matmul(None, r_sqrt, m.H(self.A[s]))
        
        R = R.reshape((self.q * self.D, self.D))
        
        V = m.H(ns.nullspace_qr(m.H(R))) 
        #R can be pretty huge for large q and D. The decomp. can take a long time...

#        print "V Checks..."
#        print sp.allclose(sp.dot(V, m.H(V)), sp.eye(self.q*self.D - self.D))
#        print sp.allclose(sp.dot(V, R), 0)
        V = V.reshape(((self.q - 1) * self.D, self.D, self.q)) 
        
        #prepare for using V[s] and already take the adjoint, since we use it more often
        Vsh = sp.empty((self.q, self.D, (self.q - 1) * self.D), dtype=self.typ, 
                       order=self.odr)
        for s in xrange(self.q):
            Vsh[s] = m.H(V[:,:,s])
        
        return Vsh
        
    def Calc_x(self, l_sqrt, l_sqrt_i, r_sqrt, r_sqrt_i, Vsh, out=None):
        if out is None:
            out = sp.zeros((self.D, (self.q - 1) * self.D), dtype=self.typ, 
                           order=self.odr)
        
        tmp = sp.zeros_like(out)
        tmp2 = sp.zeros_like(self.A[0])
        for s in xrange(self.q):
            tmp += m.matmul(None, self.A[s], self.K, r_sqrt_i, Vsh[s])
            
            tmp2.fill(0)
            for t in xrange(self.q):
                tmp2 += m.matmul(None, self.C[s, t], self.r, m.H(self.A[t]))
            tmp += m.matmul(None, tmp2, r_sqrt_i, Vsh[s])
            
        out += sp.dot(l_sqrt, tmp)
        
        tmp.fill(0)
        for s in xrange(self.q):
            tmp2.fill(0)
            for t in xrange(self.q):
                tmp2 += m.matmul(None, m.H(self.A[t]), self.l, self.C[t, s])
            tmp += m.matmul(None, tmp2, r_sqrt, Vsh[s])
        out += sp.dot(l_sqrt_i, tmp)
        
        return out
        
    def B_from_x(self, x, Vsh, l_sqrt_i, r_sqrt_i, out=None):
        if out is None:
            out = sp.zeros_like(self.A)
            
        for s in xrange(self.q):
            m.matmul(out[s], l_sqrt_i, x, m.H(Vsh[s]), r_sqrt_i)
            
        return out
        
    def Calc_B(self):
        #print "sqrts and inverses start"
        self.l_sqrt = m.sqrtmh(self.l)
        self.l_sqrt_i = la.inv(self.l_sqrt)
        self.r_sqrt = m.sqrtmh(self.r)
        self.r_sqrt_i = la.inv(self.r_sqrt)
        #print "sqrts and inverses stop"
        
        if self.sanity_checks:
            if not sp.allclose(sp.dot(self.l_sqrt, self.l_sqrt), self.l):
                print "Sanity check failed: l_sqrt is bad!"
            if not sp.allclose(sp.dot(self.l_sqrt, self.l_sqrt_i), sp.eye(self.D)):
                print "Sanity check failed: l_sqrt_i is bad!"
            if not sp.allclose(sp.dot(self.r_sqrt, self.r_sqrt), self.r):
                print "Sanity check failed: l_sqrt is bad!"
            if not sp.allclose(sp.dot(self.r_sqrt, self.r_sqrt_i), sp.eye(self.D)):
                print "Sanity check failed: l_sqrt_i is bad!"
                
        #print "Vsh start"
        self.Vsh = self.Calc_Vsh(self.r_sqrt)
        #print "Vsh stop"
        
        #print "x start"
        self.x = self.Calc_x(self.l_sqrt, self.l_sqrt_i, self.r_sqrt, 
                        self.r_sqrt_i, self.Vsh)
        #print "x stop"
        
        self.eta = sp.sqrt(adot(self.x, self.x))
        
        B = self.B_from_x(self.x, self.Vsh, self.l_sqrt_i, self.r_sqrt_i)
        
        if self.sanity_checks:
            #Test gauge-fixing:
            tst = sp.zeros_like(self.l)
            for s in xrange(self.q):
                tst += m.matmul(None, B[s], self.r, m.H(self.A[s]))
            if not sp.allclose(tst, 0):
                print "Sanity check failed: Gauge-fixing violation!"

        return B
        
    def TakeStep(self, dtau, B=None):
        if B is None:
            B = self.Calc_B()
        
        for s in xrange(self.q):
            self.A[s] += -dtau * B[s]
            
    def Calc_BHB(self, x):
        B = self.B_from_x(x, self.Vsh, self.l_sqrt_i, self.r_sqrt_i)
        
        rVsh = self.Vsh.copy()
        for s in xrange(self.q):
            rVsh[s] = sp.dot(self.r_sqrt_i, rVsh[s])
        
        res = sp.dot(self.l_sqrt, 
                     self.EpsR_2(self.r, op=self.h_nn, A1=B, A3=rVsh))
        
        return None
            
    def Find_min_h(self, B, dtau_init, tol=5E-2):
        dtau = dtau_init
        d = 1.0
        #dh_dtau = 0
        
        tau_min = 0
        
        A0 = self.A.copy()
        
        h_min = self.h
        A_min = self.A.copy()
        l_min = self.l.copy()
        r_min = self.r.copy()
        
        itr = 0
        while itr == 0 or itr < 30 and (abs(dtau) / tau_min > tol or tau_min == 0):
            itr += 1
            for s in xrange(self.q):
                self.A[s] = A_min[s] -d * dtau * B[s]
            
            self.l[:] = l_min
            self.r[:] = r_min
            
            self.Calc_lr()
            
            self.h = self.Expect_2S(self.h_nn)
            
            #dh_dtau = d * (self.h - h_min) / dtau
            
            print (tau_min + dtau, self.h.real, tau_min)
            
            if self.h.real < h_min.real:
                #self.Restore_CF()
                h_min = self.h
                A_min[:] = self.A
                l_min[:] = self.l
                r_min[:] = self.r
                
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
                
        self.A = A0
        
        return tau_min
        
    def Find_min_h_Brent(self, B, dtau_init, tol=5E-2, skipIfLower=False, 
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
                
                self.Calc_lr()
                
                h = self.Expect_2S(self.h_nn)
                
                print (tau, h.real)
                
                res = h.real
                
                taus.append(tau)
                hs.append(res)
                
                return res
        
        A0 = self.A.copy()
        
        if skipIfLower:
            if f(dtau_init) < self.h.real:
                return dtau_init
        
        fb_brack = (dtau_init * 0.9, dtau_init * 1.1)
        if trybracket:
            brack = (dtau_init * 0.1, dtau_init, dtau_init * 2.0)
        else:
            brack = fb_brack
                
        try:
            tau_opt = op.brent(f, 
                               brack=brack, 
                               tol=tol)
        except ValueError:
            print "Bracketing attempt failed..."
            tau_opt = op.brent(f, 
                               brack=fb_brack, 
                               tol=tol)            
        
        self.A = A0
        
        return tau_opt
        
    def Step_Reduces_h(self, B, dtau):
        A0 = self.A.copy()
        
        for s in xrange(self.q):
            self.A[s] = A0[s] - dtau * B[s]
        
        self.Calc_lr()
        
        h = self.Expect_2S(self.h_nn)
        
        self.A = A0
        
        return h.real < self.h.real, h

    def CalcB_CG(self, B_CG_0, x_0, eta_0, dtau_init, reset=False,
                 skipIfLower=False, brent=True):
        #self.Calc_lr()
        #self.Calc_C()
        #self.Calc_K()
        
        B = self.Calc_B()
        eta = self.eta
        x = self.x
        
        if reset:
            beta = 0.
            print "RESET CG"
            
            B_CG = B
        else:
            beta = (eta**2) / eta_0**2
            
            #xy = adot(x_0, x)
            #betaPR = (eta**2 - xy) / eta_0**2
        
            print "BetaFR = " + str(beta)
            #print "BetaPR = " + str(betaPR)
        
            beta = max(0, beta.real)
        
            B_CG = B + beta * B_CG_0
        
        taus = []
        hs = []
        
        if skipIfLower:
            stepRedH, h = self.Step_Reduces_h(B_CG, dtau_init)
            taus.append(dtau_init)
            hs.append(h)
        
        if skipIfLower and stepRedH:
            tau = self.Find_min_h(B_CG, dtau_init)
        else:
            if brent:
                tau = self.Find_min_h_Brent(B_CG, dtau_init, taus=taus, hs=hs,
                                            trybracket=False)
            else:
                tau = self.Find_min_h(B_CG, dtau_init)
        
        if tau < 0:
            print "RESET due to negative dtau!"
            B_CG = B
            tau = self.Find_min_h_Brent(B_CG, dtau_init)
        
        return B_CG, B, x, eta, tau
        
            
    def Expect_SS(self, op):
        Or = self.EpsR(self.r, op=op)
        
        return adot(self.l, Or)
            
    def Expect_2S(self, op):
        res = self.EpsR_2(self.r, op)
        
        return adot(self.l, res)
        
    def Density_SS(self):
        rho = sp.empty((self.q, self.q), dtype=self.typ)
        for (s, t) in sp.ndindex(self.q, self.q):
            m.matmul(self.tmp, self.A[t], self.r, m.H(self.A[s]))
            rho[s, t] = adot(self.l, self.tmp)
        return rho
            
    def SaveState(self, file, userdata=None):
        if userdata is None:
            userdata = self.userdata
        tosave = sp.empty((5), dtype=sp.ndarray)
        tosave[0] = self.A
        tosave[1] = self.l
        tosave[2] = self.r
        tosave[3] = self.K
        tosave[4] = sp.asarray(userdata)
        sp.save(file, tosave)
        
    def LoadState(self, file, expand=False):
        state = sp.load(file)
        
        newA = state[0]
        newl = state[1]
        newr = state[2]
        newK = state[3]
        if state.shape[0] > 4:
            self.userdata = state[4]
        
        if (newA.shape == self.A.shape) and (newl.shape == self.l.shape) and (
        newr.shape == self.r.shape) and (newK.shape == self.K.shape):
            self.A[:] = newA
            self.l[:] = newl
            self.r[:] = newr
            self.K[:] = newK
            return True
        elif expand and (len(newA.shape) == 3) and (newA.shape[0] == 
        self.A.shape[0]) and (newA.shape[1] == newA.shape[2]) and (newA.shape[1]
        <= self.A.shape[1]):
            newD = self.D
            savedD = newA.shape[1]
            self._init_arrays(savedD, self.q)
            self.A[:] = newA
            self.l[:] = newl
            self.r[:] = newr
            self.K[:] = newK
            self.Expand_D(newD)
            print "EXPANDED"
        else:
            return False
            
    def Expand_D(self, newD):
        if newD < self.D:
            return False
        
        oldD = self.D
        oldA = self.A
        oldl = self.l
        oldr = self.r
        oldK = self.K
        
        self._init_arrays(newD, self.q)
        
        norm = la.norm(oldA)
        fac = (norm / (self.q * oldD**2))
#        m.randomize_cmplx(newA[:, self.D:, self.D:], a=-fac, b=fac)
        m.randomize_cmplx(self.A[:, :oldD, oldD:], a=-fac, b=fac)
        m.randomize_cmplx(self.A[:, oldD:, :oldD], a=-fac, b=fac)
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
        
    def Fuzz_State(self, f=1.0):
        norm = la.norm(self.A)
        fac = f*(norm / (self.q * self.D**2))        
        
        R = sp.empty_like(self.A)
        m.randomize_cmplx(R, -fac/2.0, fac/2.0)
        
        self.A += R