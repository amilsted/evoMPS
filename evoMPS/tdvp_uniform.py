# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 17:29:27 2011

@author: Ashley Milsted

"""
import scipy as sp
import scipy.linalg as la
import nullspace as ns
import matmul as m


def myMVop(opData, x):
    l = opData[0]
    r = opData[1]
    A = opData[2]
    
    opres = sp.zeros_like(A[0])
    for s in xrange(A.shape[0]):
        opres += m.matmul(None, A[s], x, m.H(A[s]))
        
    return x - opres  + r * sp.trace(sp.dot(l, x))

def myVVop(a, b):
    return sp.trace(sp.dot(a, b))
        
class evoMPS_TDVP_Uniform:
    odr = 'C'
    typ = sp.complex128
    eps = 0
    
    h_nn = None    
    
    symm_gauge = True
    
    sanity_checks = True
    
    def __init__(self, D, q):
        self.eps = sp.finfo(self.typ).eps
        
        self.eta = 0
        
        self.D = D
        self.q = q
        
        self.A = sp.zeros((q, D, D), dtype=self.typ, order=self.odr)
        
        self.C = sp.empty((q, q, D, D), dtype=self.typ, order=self.odr)
        
        self.K = sp.ones_like(self.A[0])
        
        self.l = sp.zeros_like(self.A[0])
        self.r = sp.zeros_like(self.A[0])
        
        self.l.real = sp.eye(self.D)
        self.r.real = sp.eye(self.D)
        
        #for s in xrange(q):
        #    self.A[s] = sp.eye(D)
            
        self.A.real = sp.rand(q, D, D) - 0.5
        self.A.imag = sp.rand(q, D, D) - 0.5
        
        self.tmp = sp.empty_like(self.A[0])
            
    def EpsR(self, x, op=None, out=None):
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
            
        if op is None:
            for s in xrange(self.q):
                out += m.matmul(self.tmp, self.A[s], x, m.H(self.A[s]))            
        else:
            for (s, t) in sp.ndindex(self.q, self.q):
                o_st = op(s, t)
                if o_st != 0.:
                    m.matmul(self.tmp, self.A[t], x, m.H(self.A[s]))
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
        
    def _Calc_lr(self, x, e, tmp, max_itr=5000, rtol=1E-14, atol=1E-14):        
        for i in xrange(5000):
            e(x, out=tmp)
            ev = la.norm(tmp)
            tmp *= (1 / ev)
            if sp.allclose(tmp, x, rtol=rtol, atol=atol):
                x[:] = tmp
                break
            x[:] = tmp
        
        print "Iterations: " + str(i + 1)
        print "Found ev: " + str(ev)        
        
        #re-scale
        if not sp.allclose(ev, 1.0, rtol=rtol, atol=atol):
            self.A *= 1 / sp.sqrt(ev)
            ev = la.norm(self.EpsL(self.l, out=tmp))
            print "After re-scale: " + str(ev) 
        
        return x
    
    def Calc_lr(self, renorm=True, force_r_CF=False):
#        E = sp.zeros((self.D**2, self.D**2), dtype=self.typ, order='C')
#        
#        for s in xrange(self.q):
#            E += sp.kron(self.A[s], self.A[s].conj())
#            
#        ev, eVL, eVR = la.eig(E, left=True, right=True)
#        
#        i = sp.argmax(ev)
#        
#        self.l = eVL[i].reshape((self.D, self.D))
#        self.r = eVR[i].reshape((self.D, self.D))
#        
#        #Test!
#        print "Sledgehammer:"
#        print ev[i]
#        print sp.allclose(self.EpsL(self.l), self.l * ev[i])
#        print sp.allclose(self.EpsR(self.r), self.r * ev[i])
#        
#        #Method using eps maps... Depends on max. ev = 1
        
        tmp = sp.empty_like(self.tmp)
        
        print "Left..."
        self._Calc_lr(self.l, self.EpsL, tmp)
        
        print "Right..."
        self._Calc_lr(self.r, self.EpsR, tmp)
                    
        #normalize eigenvectors:
        norm = sp.trace(sp.dot(self.l, self.r, out=tmp))
        while not sp.allclose(norm, 1, atol=1E-14, rtol=0):
            self.l *= 1 / sp.sqrt(norm)
            self.r *= 1 / sp.sqrt(norm)
            
            norm = sp.trace(sp.dot(self.l, self.r, out=tmp))
        
        if force_r_CF or not self.symm_gauge: #right to do this every time?
            fac = self.D / sp.trace(self.r)
            self.l *= 1 / fac
            self.r *= fac

        #Test!
        if self.sanity_checks:
            if not sp.allclose(self.EpsL(self.l), self.l, rtol=1E-12, atol=1E-12):
                print "Sanity check failed: Left eigenvector bad! Off by: " \
                       + str(la.norm(self.EpsL(self.l) - self.l))
                       
            if not sp.allclose(self.EpsR(self.r), self.r, rtol=1E-12, atol=1E-12):
                print "Sanity check failed: Right eigenvector bad! Off by: " \
                       + str(la.norm(self.EpsR(self.r) - self.r))
            
            if not sp.allclose(self.l, m.H(self.l), rtol=1E-12, atol=1E-12):
                print "Sanity check failed: l is not hermitian!"

            if not sp.allclose(self.r, m.H(self.r), rtol=1E-12, atol=1E-12):
                print "Sanity check failed: r is not hermitian!"
            
            if not sp.all(la.eigvalsh(self.l) > 0):
                print "Sanity check failed: l is not pos. def.!"
                
            if not sp.all(la.eigvalsh(self.r) > 0):
                print "Sanity check failed: r is not pos. def.!"
            
        print "Norm = " + str(sp.trace(sp.dot(self.l, self.r, out=tmp)))
            
        print "IsCF symm: " + str(sp.allclose(self.l, self.r))
    
    def Restore_CF(self):      
        M = sp.zeros_like(self.r)
        for s in xrange(self.q):
            M += m.matmul(None, self.A[s], m.H(self.A[s]))
        
        G = m.H(la.cholesky(self.r))
        G_i = m.invtr(G, lower=True)
        
        for s in xrange(self.q):
            m.matmul(self.A[s], G_i, self.A[s], G)
            
        m.matmul(self.l, m.H(G), self.l, G)
        self.r[:] = sp.eye(self.D)
            
        self.Calc_lr(force_r_CF=True)
        
        if self.sanity_checks:
            M.fill(0)
            for s in xrange(self.q):
                M += m.matmul(None, self.A[s], m.H(self.A[s]))            
                
            if not sp.allclose(M, sp.eye(M.shape[0])) or not sp.allclose(self.r, sp.eye(self.D)):
                print "Sanity check failed: Could not achieve R-CF."

        if self.symm_gauge:    #Move to symmetrical gauge.
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
    
    def Calc_C(self):
        self.C.fill(0)
        
        AA = sp.empty_like(self.A[0])
        
        for (u, v) in sp.ndindex(self.q, self.q):
            m.matmul(AA, self.A[u], self.A[v])
            for (s, t) in sp.ndindex(self.q, self.q):
                self.C[s, t] += self.h_nn(s, t, u, v) * AA
    
    def Calc_K(self):
        Hr = sp.zeros_like(self.A[0])
        
        for (s, t) in sp.ndindex(self.q, self.q):
            Hr += m.matmul(None, self.C[s, t], self.r, m.H(self.A[t]), m.H(self.A[s]))
        
        self.h = sp.trace(sp.dot(self.l, Hr))
        
        QHr = Hr - self.r * sp.trace(m.matmul(None, self.l, Hr))
        
        opData = (self.l, self.r, self.A)
        #self.K.fill(1)
        
        self.K = m.bicgstab_iso(opData, self.K, QHr, myMVop, myVVop)
        
        #Test
        if self.sanity_checks and not sp.allclose(myMVop(opData, self.K),
                                                  QHr, atol=1E-13, rtol=1E-13):
            print "Sanity check failed: Bad K solution!"
            
    def Calc_Vsh(self, r_sqrt): #this really is just the same as for the generic case
        R = sp.zeros((self.D, self.q, self.D), dtype=self.typ, order='C')
        
        for s in xrange(self.q):
            R[:,s,:] = m.matmul(None, r_sqrt, m.H(self.A[s]))

        R = R.reshape((self.q * self.D, self.D))
        V = m.H(ns.nullspace(m.H(R)))
        #print (q[n]*D[n] - D[n-1], q[n]*D[n])
        #print V.shape
#        print "V Checks..."
#        print sp.allclose(sp.dot(V, m.H(V)), sp.eye(self.q*self.D - self.D))
#        print sp.allclose(sp.dot(V, R), 0)
        V = V.reshape(((self.q - 1) * self.D, self.D, self.q)) #this works with the above form for R
        
        #prepare for using V[s] and already take the adjoint, since we use it more often
        Vsh = sp.empty((self.q, self.D, (self.q - 1) * self.D), dtype=self.typ, order=self.odr)
        for s in xrange(self.q):
            Vsh[s] = m.H(V[:,:,s])
        
        return Vsh
        
    def Calc_x(self, l_sqrt, l_sqrt_i, r_sqrt, r_sqrt_i, Vsh, out=None):
        if out is None:
            out = sp.zeros(((self.q - 1) * self.D, self.D), dtype=self.typ, order=self.odr)
            
        for (s, t) in sp.ndindex(self.q, self.q):
            out += m.matmul(None, l_sqrt, self.C[s, t], self.r, m.H(self.A[t]), r_sqrt_i, Vsh[s])
            
        for (s, t) in sp.ndindex(self.q, self.q):
            out += m.matmul(None, l_sqrt_i, m.H(self.A[t]), self.l, self.C[t, s], r_sqrt, Vsh[s])
            
        for s in xrange(self.q):
            out += m.matmul(None, l_sqrt, self.A[s], self.K, r_sqrt_i, Vsh[s])
        
        return out
        
    def Calc_B(self, x, Vsh, l_sqrt_i, r_sqrt_i, out=None):
        if out is None:
            out = sp.zeros_like(self.A)
            
        for s in xrange(self.q):
            m.matmul(out[s], l_sqrt_i, x, m.H(Vsh[s]), r_sqrt_i)
            
        return out
        
    def TakeStep(self, dtau):
        l_sqrt = m.sqrtmh(self.l)
        l_sqrt_i = la.inv(l_sqrt)
        r_sqrt = m.sqrtmh(self.r)
        r_sqrt_i = la.inv(r_sqrt)
        
        if self.sanity_checks:
            if not sp.allclose(sp.dot(l_sqrt, l_sqrt), self.l):
                print "Sanity check failed: l_sqrt is bad!"
            if not sp.allclose(sp.dot(l_sqrt, l_sqrt_i), sp.eye(self.D)):
                print "Sanity check failed: l_sqrt_i is bad!"
            if not sp.allclose(sp.dot(r_sqrt, r_sqrt), self.r):
                print "Sanity check failed: l_sqrt is bad!"
            if not sp.allclose(sp.dot(r_sqrt, r_sqrt_i), sp.eye(self.D)):
                print "Sanity check failed: l_sqrt_i is bad!"
                
        Vsh = self.Calc_Vsh(r_sqrt)
        
        x = self.Calc_x(l_sqrt, l_sqrt_i, r_sqrt, r_sqrt_i, Vsh)
        
        self.eta = sp.sqrt(sp.trace(sp.dot(m.H(x), x)))
        
        B = self.Calc_B(x, Vsh, l_sqrt_i, r_sqrt_i)
        
        if self.sanity_checks:
            #Test gauge-fixing:
            tst = sp.zeros_like(self.l)
            for s in xrange(self.q):
                tst += m.matmul(None, B[s], self.r, m.H(self.A[s]))
            if not sp.allclose(tst, 0):
                print "Sanity check failed: Gauge-fixing violation!"
        
        for s in xrange(self.q):
            self.A[s] += -dtau * B[s]
            
    def Expect_SS(self, op):
        Or = self.EpsR(self.r, op=op)
        
        return sp.trace(sp.dot(self.l, Or))        
            
    def Expect_2S(self, op):
        AAuv = sp.empty_like(self.A[0])
        res = sp.zeros_like(self.r)
        
        for (u, v) in sp.ndindex(self.q, self.q):
            m.matmul(AAuv, self.A[u], self.A[v])
            for (s, t) in sp.ndindex(self.q, self.q):
                res += op(u, v, s, t) * m.matmul(None, self.A[s], self.A[t], 
                                                       self.r, m.H(AAuv))
        
        return sp.trace(sp.dot(self.l, res))
            
    def SaveState(self, file):
        sp.save(file, self.A)
        
    def LoadState(self, file):
        self.A = sp.load(file)            