# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 17:29:27 2011

@author: Ashley Milsted

"""
import scipy as sp
import scipy.linalg as la
from scipy import *
import nullspace as ns
from matmul import *


def myMVop(A, x):
    opres = zeros_like(A[2][0])
    for i in xrange(A[2].shape[0]):
        opres += matmul(None, A[2][i], x, H(A[2][i]))
        
    return x - opres  + A[1] * sp.trace(dot(A[0], x))

def myVVop(a, b):
    return sp.trace(dot(a, b))
        
class evoMPS_TDVP_Uniform:
    odr = 'C'
    typ = complex128
    
    h_nn = None
    
    def __init__(self, D, q):
        self.D = D
        self.q = q
        
        self.A = zeros((q, D, D), dtype=self.typ, order=self.odr)
        
        self.C = empty((q, q, D, D), dtype=self.typ, order=self.odr)
        
        self.K = ones_like(A[0])
        
        self.l = empty_like(A[0])
        self.r = empty_like(A[0])
        
        for s in xrange(q):
            self.A[s] = eye(D)
            
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
            out = zeros_like(self.r)
        else:
            out.fill(0.)
            
        tmp = empty_like(out)
        if o is None:
            for s in xrange(self.q):
                out += matmul(tmp, self.A[s], x, H(self.A[s]))            
        else:
            for (s, t) in sp.ndindex(self.q, self.q):
                o_st = o(n, s, t)
                if o_st != 0.:
                    matmul(tmp, self.A[n][t], x, H(self.A[n][s]))
                    tmp *= o_st
                    out += tmp
        return out
        
    def EpsL(self, x, out=None):
        if out is None:
            out = zeros_like(self.r)
        else:
            out.fill(0.)
            
        tmp = empty_like(out)
        for s in xrange(self.q):
            out += matmul(tmp, self.A[s], x, H(self.A[s]))
            
        return out
    
    def Calc_rl(self, renorm=True):
        E = zeros((D**2, D**2), dtype=self.typ, order='C')
        
        for s in xrange(self.q):
            E += sp.kron(self.A[s], self.A[s].conj())
            
        ev, eVL, eVR = la.eig(E, left=True, right=True)
        
        i = argmax(ev)
        
        self.l = eVL[i].reshape((self.D, self.D))
        self.r = eVR[i].reshape((self.D, self.D))
        
        #Test!
        print "Sledgehammer:"
        print sp.allclose(self.EpsL(self.l), self.l * ev[i])
        print sp.allclose(self.EpsR(self.r), self.r * ev[i])
        
        #Method using eps maps... Depends on max. ev = 1
        
        ev = 1
        
        self.l.fill(0)
        self.l.real = eye(self.D)
        
        l_new = empty_like(self.l)
        
        for i in xrange(100):
            l_new = self.EpsL(self.l, out=l_new)
            ev = la.norm(l_new)
            l_new = l_new * (1 / ev)
            if allclose(l_new, self.l):
                break
            self.l = l_new
            
        if renorm:
            self.A *= 1 / sp.sqrt(ev)
            ev = 1 #FIXME
        
        self.r.fill(0)
        self.r.real = eye(self.D)
        
        r_new = empty_like(self.r)
        
        for i in xrange(100):
            r_new = self.EpsR(self.r, out=r_new)
            r_new = r_new * (1 / la.norm(r_new))
            if allclose(r_new, self.r):
                break
            self.r = r_new
            
        #Test!
        print "Flipside:"
        print sp.allclose(self.EpsL(self.l), self.l * ev)
        print sp.allclose(self.EpsR(self.r), self.r * ev)
            
    
    def Restore_CF(self):
        M = zeros_like(self.r)
        for s in xrange(self.q):
            M += matmul(None, self.A[s], H(self.A[s]))     
        
        try:
            tu = la.cholesky(M) #Assumes M is pos. def.. It should raise LinAlgError if not.
            G = H(invtr(tu, overwrite=True), out=tu) #G is now lower-triangular
            G_i = invtr(G, overwrite=True, lower=True)
        except sp.linalg.LinAlgError:
            print "Restore_ON_R_n: Falling back to eigh()!"
            e,Gh = la.eigh(M)
            G = H(matmul(None, Gh, diag(1/sqrt(e) + 0.j)))
            G_i = la.inv(G_nm1)        
            
        
        for s in xrange(self.q[n]):                
            matmul(self.A[n][s], G, self.A[n][s], G_i)
            #It's ok to use the same matrix as out and as an operand here
            #since there are > 2 matrices in the chain and it is not the last argument.
            
        #TODO: Move to symmetrical gauge.
    
    def Calc_C(self):
        self.C.fill(0)
        
        AA = empty_like(self.A[0])
        
        for (u, v) in ndindex(self.q, self.q):
            matmul(AA, self.A[u], self.A[v])
            for (s, t) in ndindex(self.q, self.q):
                C[s, t] += h_nn(n, s, t, u, v) * AA
    
    def Calc_K(self):
        Hr = empty_like(self.A[0])
        
        AAst = empty_like(self.A[0])
        
        for (s, t) in ndindex(self.q, self.q):
            matmul(AAst, self.A[s], self.A[t])
            for (u, v) in ndindex(self.q, self.q):
                Hr += h_nn(n, u, v, s, t) * matmul(None, AAst, self.r, H(self.A[v]), H(self.A[u]))
                
        QHr = Hr - self.r * sp.trace(matmul(None, self.l, Hr))
        
        Amod = (self.l, self.r, self.A)
        self.K.fill(1)
        
        self.K = bicgstab_iso(Amod, self.K, QHr, myMVop, myVVop)
        
    def Calc_Vsh(self, r_sqrt): #this really is just the same as for the generic case
        R = zeros((self.D, self.q, self.D), dtype=self.typ, order='C')
        
        for s in xrange(self.q):
            R[:,s,:] = matmul(None, r_sqrt, H(self.A[s]))

        R = R.reshape((self.q * self.D, self.D))
        V = H(ns.nullspace(H(R)))
        #print (q[n]*D[n] - D[n-1], q[n]*D[n])
        #print V.shape
        #print allclose(mat(V) * mat(V).H, eye(q[n]*D[n] - D[n-1]))
        #print allclose(mat(V) * mat(Rh).H, 0)
        V = V.reshape(((self.q - 1) * self.D, self.D, self.q)) #this works with the above form for R
        
        #prepare for using V[s] and already take the adjoint, since we use it more often
        Vsh = empty((self.q, self.D, (self.q - 1) * self.D), dtype=self.typ, order=self.odr)
        for s in xrange(self.q):
            Vsh[s] = H(V[:,:,s])
        
        return Vsh
        
    def Calc_x(self, l_sqrt, l_sqrt_i, r_sqrt, r_sqrt_i, Vsh, out=None):
        if out is None:
            out = zeros(((self.q - 1) * self.D, self.D), dtype=self.typ, order=self.odr)
            
        for (s, t) in sp.nditer(self.q, self.q):
            out += matmul(None, l_sqrt, self.C[s, t], self.r, H(self.A[t]), r_sqrt_i, Vsh[s])
            
        for (s, t) in sp.nditer(self.q, self.q):
            out += matmul(None, l_sqrt_i, H(self.A[t]), self.l, self.C[t, s], r_sqrt, Vsh[s])
            
        for s in xrange(self.q):
            out += matmul(None, l_sqrt, self.A[s], self.K, r_sqrt_i, Vsh[s])
        
        return out
        
    def Calc_B(self, x, Vsh, l_sqrt_i, out=None):
        if out is None:
            out = zeros_like(self.A)
            
        for s in xrange(self.q):
            matmul(out[s], l_sqrt_i, x, H(Vsh[s]), r_sqrt_i)
            
        return out
        
    def TakeStep(self, dtau):
        l_sqrt = la.sqrtm(self.l)
        l_sqrt_i = la.inv(l_sqrt)
        r_sqrt = la.sqrtm(self.r)
        r_sqrt_i = la.inv(r_sqrt)
        
        Vsh = self.Calc_Vsh(r_sqrt)
        
        x = self.Calc_x(l_sqrt, l_sqrt_i, r_sqrt, r_sqrt_i, Vsh)
        
        B = self.Calc_B(x, Vsh, l_sqrt_i)
        
        self.A += -dtau * B[s]