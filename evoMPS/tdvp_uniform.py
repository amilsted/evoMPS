# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 17:29:27 2011

@author: Ashley Milsted

"""
import scipy as sp
import scipy.linalg as la
import nullspace as ns
import matmul as m


def myMVop(A, x):
    opres = sp.zeros_like(A[2][0])
    for i in xrange(A[2].shape[0]):
        opres += m.matmul(None, A[2][i], x, m.H(A[2][i]))
        
    return x - opres  + A[1] * sp.trace(sp.dot(A[0], x))

def myVVop(a, b):
    return sp.trace(sp.dot(a, b))
        
class evoMPS_TDVP_Uniform:
    odr = 'C'
    typ = sp.complex128
    
    h_nn = None
    
    def __init__(self, D, q):
        self.D = D
        self.q = q
        
        self.A = sp.zeros((q, D, D), dtype=self.typ, order=self.odr)
        
        self.C = sp.empty((q, q, D, D), dtype=self.typ, order=self.odr)
        
        self.K = sp.ones_like(A[0])
        
        self.l = sp.empty_like(A[0])
        self.r = sp.empty_like(A[0])
        
        for s in xrange(q):
            self.A[s] = sp.eye(D)
            
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
            
        tmp = sp.empty_like(out)
        if op is None:
            for s in xrange(self.q):
                out += m.matmul(tmp, self.A[s], x, m.H(self.A[s]))            
        else:
            for (s, t) in sp.ndindex(self.q, self.q):
                o_st = op(s, t)
                if o_st != 0.:
                    m.matmul(tmp, self.A[t], x, m.H(self.A[s]))
                    tmp *= o_st
                    out += tmp
        return out
        
    def EpsL(self, x, out=None):
        if out is None:
            out = sp.zeros_like(self.r)
        else:
            out.fill(0.)
            
        tmp = sp.empty_like(out)
        for s in xrange(self.q):
            out += m.matmul(tmp, self.A[s], x, m.H(self.A[s]))
            
        return out
    
    def Calc_rl(self, renorm=True):
        E = sp.zeros((D**2, D**2), dtype=self.typ, order='C')
        
        for s in xrange(self.q):
            E += sp.kron(self.A[s], self.A[s].conj())
            
        ev, eVL, eVR = la.eig(E, left=True, right=True)
        
        i = sp.argmax(ev)
        
        self.l = eVL[i].reshape((self.D, self.D))
        self.r = eVR[i].reshape((self.D, self.D))
        
        #Test!
        print "Sledgehammer:"
        print sp.allclose(self.EpsL(self.l), self.l * ev[i])
        print sp.allclose(self.EpsR(self.r), self.r * ev[i])
        
        #Method using eps maps... Depends on max. ev = 1
        
        ev = 1
        
        self.l.fill(0)
        self.l.real = sp.eye(self.D)
        
        l_new = sp.empty_like(self.l)
        
        for i in xrange(100):
            l_new = self.EpsL(self.l, out=l_new)
            ev = la.norm(l_new)
            l_new = l_new * (1 / ev)
            if sp.allclose(l_new, self.l):
                break
            self.l = l_new
            
        if renorm:
            self.A *= 1 / sp.sqrt(ev)
            ev = 1 #FIXME
        
        self.r.fill(0)
        self.r.real = sp.eye(self.D)
        
        r_new = sp.empty_like(self.r)
        
        for i in xrange(100):
            r_new = self.EpsR(self.r, out=r_new)
            r_new = r_new * (1 / la.norm(r_new))
            if sp.allclose(r_new, self.r):
                break
            self.r = r_new
            
        #Test!
        print "Flipside:"
        print sp.allclose(self.EpsL(self.l), self.l * ev)
        print sp.allclose(self.EpsR(self.r), self.r * ev)
            
    
    def Restore_CF(self):
        M = sp.zeros_like(self.r)
        for s in xrange(self.q):
            M += m.matmul(None, self.A[s], H(self.A[s]))     
        
        try:
            tu = la.cholesky(M) #Assumes M is pos. def.. It should raise LinAlgError if not.
            G = m.H(m.invtr(tu, overwrite=True), out=tu) #G is now lower-triangular
            G_i = m.invtr(G, overwrite=True, lower=True)
        except sp.linalg.LinAlgError:
            print "Restore_ON_R_n: Falling back to eigh()!"
            e,Gh = la.eigh(M)
            G = m.H(m.matmul(None, Gh, sp.diag(1/sp.sqrt(e) + 0.j)))
            G_i = la.inv(G)        
            
        
        for s in xrange(self.q):                
            m.matmul(self.A[s], G, self.A[s], G_i)
            #It's ok to use the same matrix as out and as an operand here
            #since there are > 2 matrices in the chain and it is not the last argument.
            
        #TODO: Move to symmetrical gauge.
    
    def Calc_C(self):
        self.C.fill(0)
        
        AA = sp.empty_like(self.A[0])
        
        for (u, v) in sp.ndindex(self.q, self.q):
            m.matmul(AA, self.A[u], self.A[v])
            for (s, t) in sp.ndindex(self.q, self.q):
                self.C[s, t] += self.h_nn(s, t, u, v) * AA
    
    def Calc_K(self):
        Hr = sp.empty_like(self.A[0])
        
        AAst = sp.empty_like(self.A[0])
        
        for (s, t) in ndindex(self.q, self.q):
            m.matmul(AAst, self.A[s], self.A[t])
            for (u, v) in sp.ndindex(self.q, self.q):
                Hr += self.h_nn(u, v, s, t) * m.matmul(None, AAst, self.r, m.H(self.A[v]), m.H(self.A[u]))
                
        QHr = Hr - self.r * sp.trace(m.matmul(None, self.l, Hr))
        
        Amod = (self.l, self.r, self.A)
        self.K.fill(1)
        
        self.K = m.bicgstab_iso(Amod, self.K, QHr, myMVop, myVVop)
        
    def Calc_Vsh(self, r_sqrt): #this really is just the same as for the generic case
        R = sp.zeros((self.D, self.q, self.D), dtype=self.typ, order='C')
        
        for s in xrange(self.q):
            R[:,s,:] = m.matmul(None, r_sqrt, m.H(self.A[s]))

        R = R.reshape((self.q * self.D, self.D))
        V = m.H(ns.nullspace(m.H(R)))
        #print (q[n]*D[n] - D[n-1], q[n]*D[n])
        #print V.shape
        #print allclose(mat(V) * mat(V).H, eye(q[n]*D[n] - D[n-1]))
        #print allclose(mat(V) * mat(Rh).H, 0)
        V = V.reshape(((self.q - 1) * self.D, self.D, self.q)) #this works with the above form for R
        
        #prepare for using V[s] and already take the adjoint, since we use it more often
        Vsh = sp.empty((self.q, self.D, (self.q - 1) * self.D), dtype=self.typ, order=self.odr)
        for s in xrange(self.q):
            Vsh[s] = m.H(V[:,:,s])
        
        return Vsh
        
    def Calc_x(self, l_sqrt, l_sqrt_i, r_sqrt, r_sqrt_i, Vsh, out=None):
        if out is None:
            out = sp.zeros(((self.q - 1) * self.D, self.D), dtype=self.typ, order=self.odr)
            
        for (s, t) in sp.nditer(self.q, self.q):
            out += m.matmul(None, l_sqrt, self.C[s, t], self.r, m.H(self.A[t]), r_sqrt_i, Vsh[s])
            
        for (s, t) in sp.nditer(self.q, self.q):
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
        l_sqrt = la.sqrtm(self.l)
        l_sqrt_i = la.inv(l_sqrt)
        r_sqrt = la.sqrtm(self.r)
        r_sqrt_i = la.inv(r_sqrt)
        
        Vsh = self.Calc_Vsh(r_sqrt)
        
        x = self.Calc_x(l_sqrt, l_sqrt_i, r_sqrt, r_sqrt_i, Vsh)
        
        B = self.Calc_B(x, Vsh, l_sqrt_i, r_sqrt_i)
        
        for s in xrange(self.q):
            self.A[s] += -dtau * B[s]