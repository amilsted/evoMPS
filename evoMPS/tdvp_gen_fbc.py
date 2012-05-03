# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 17:29:27 2011

@author: Ashley Milsted

Issues:
    - Getting stuck: Especially transverse Ising. More noise -> higher "residual energy"
    - Dropping below the uniform solution's energy: phi^4 with low noise
    - Wild energy fluctuations after that don't seem to go away after a fuzz
    - Expectation value discontinuities at the joins
    - We currently ignore contributions to the total energy when getting it from K[0]
      - We should probably be calculating further K's into the left region until K
        stops changing...

"""
import scipy as sp
import scipy.linalg as la
import nullspace as ns
import matmul as mm
import tdvp_uniform as uni

def go(sfbc, tau, steps, dbg=False):
    h_prev = 0
    for i in xrange(steps):
        sfbc.Restore_RCF(dbg=dbg)
        sfbc.Upd_l()
        sfbc.Upd_r()
        h, eta = sfbc.TakeStep(tau)
        norm_uni = uni.adot(sfbc.uGnd.l, sfbc.uGnd.r).real
        h_uni = sfbc.uGnd.h.real / norm_uni
        print "\t".join(map(str, (eta.real, h.real - h_uni, (h - h_prev).real)))
        print sfbc.eta.real
#        if i > 0 and (h - h_prev).real > 0:
#            break
        h_prev = h

class evoMPS_TDVP_Generic_FBC:
    odr = 'C'
    typ = sp.complex128
    
    #Numsites
    N = 0
    
    D = None
    q = None
    
    h_nn = None
    h_ext = None
    
    eps = 0
    
    sanity_checks = True
    
    uGnd = uni.evoMPS_TDVP_Uniform(10, 2)
    
    def TrimD(self):
        qacc = 1
        for n in reversed(xrange(self.N)):
            qacc *= self.q[n + 1]
            if self.D[n] > qacc:
                self.D[n] = qacc
                
        qacc = 1
        for n in xrange(1, self.N + 1):
            qacc *= self.q[n - 1]
            if self.D[n] > qacc:
                self.D[n] = qacc
            
    def _init_arrays(self):
        self.D = sp.repeat(self.uGnd.D, self.N + 2)
        self.q = sp.repeat(self.uGnd.q, self.N + 2)
        
        #Make indicies correspond to the thesis
        #Deliberately add a None to the end to catch [-1] indexing!
        self.K = sp.empty((self.N + 3), dtype=sp.ndarray) #Elements 1..N
        self.C = sp.empty((self.N + 2), dtype=sp.ndarray) #Elements 1..N-1
        self.A = sp.empty((self.N + 3), dtype=sp.ndarray) #Elements 1..N
        
        self.r = sp.empty((self.N + 3), dtype=sp.ndarray) #Elements 0..N
        self.l = sp.empty((self.N + 3), dtype=sp.ndarray)
        
        self.eta = sp.zeros((self.N + 1), dtype=self.typ)
        
        if (self.D.ndim != 1) or (self.q.ndim != 1):
            raise NameError('D and q must be 1-dimensional!')
            
        
        #Don't do anything pointless
        self.D[0] = self.uGnd.D
        self.D[self.N + 1] = self.uGnd.D
        
        #self.TrimD()
        
        print self.D
        
        self.l[0] = sp.zeros((self.D[0], self.D[0]), dtype=self.typ, order=self.odr)
        self.r[0] = sp.zeros((self.D[0], self.D[0]), dtype=self.typ, order=self.odr)
        self.K[0] = sp.zeros((self.D[0], self.D[0]), dtype=self.typ, order=self.odr) 
        self.C[0] = sp.empty((self.q[0], self.q[1], self.D[0], self.D[1]), dtype=self.typ, order=self.odr)
        self.A[0] = sp.empty((self.q[0], self.D[0], self.D[0]), dtype=self.typ, order=self.odr)
        for n in xrange(1, self.N + 2):
            self.K[n] = sp.zeros((self.D[n-1], self.D[n-1]), dtype=self.typ, order=self.odr)    
            self.r[n] = sp.zeros((self.D[n], self.D[n]), dtype=self.typ, order=self.odr)
            self.l[n] = sp.zeros((self.D[n], self.D[n]), dtype=self.typ, order=self.odr)
            self.A[n] = sp.empty((self.q[n], self.D[n-1], self.D[n]), dtype=self.typ, order=self.odr)
            if n < self.N + 1:
                self.C[n] = sp.empty((self.q[n], self.q[n+1], self.D[n-1], self.D[n+1]), dtype=self.typ, order=self.odr)
                
        self.r[self.N + 1] = self.r[self.N]
            
    def __init__(self, numsites, uGnd):
        uGnd.symm_gauge = False
        uGnd.Calc_lr()
        uGnd.Restore_CF()
        uGnd.Calc_lr()
        self.uGnd = uGnd
        
        self.h_nn = self.Wrap_h

        self.eps = sp.finfo(self.typ).eps
        
        self.N = numsites
        
        self._init_arrays()

        mm.matmul_init(dtype=self.typ, order=self.odr)
                
        for n in xrange(self.N + 2):
            self.A[n][:] = uGnd.A
            
        self.r[self.N][:] = uGnd.r
        self.l[0][:] = uGnd.l
        
        self.l_r = uGnd.l.copy()
        self.r_l = uGnd.r.copy()
        
    def Randomize(self):
        """Set A's randomly, trying to keep the norm reasonable.
        
        We need to ensure that the M matrix in EpsR() is positive definite. How?
        Does it really have to be?
        """
        for n in xrange(1, self.N + 1):
            self.A[n].real = (sp.rand(self.D[n - 1], self.D[n]) - 0.5) / sp.sqrt(self.q[n]) #/ sp.sqrt(self.N) #/ sp.sqrt(self.D[n])
            self.A[n].imag = 0#(sp.rand(self.D[n - 1], self.D[n]) - 0.5) / sp.sqrt(self.q[n])
    
    def Wrap_h(self, n, s, t, u, v):
        return self.uGnd.h_nn(s, t, u, v)
    
    def BuildC(self, n_low=-1, n_high=-1):
        """Generates the C matrices used to calculate the K's and ultimately the B's
        
        These are to be used on one side of the super-operator when applying the
        nearest-neighbour Hamiltonian, similarly to C in eqn. (44) of 
        arXiv:1103.0936v2 [cond-mat.str-el], except being for the non-norm-preserving case.

        Makes use only of the nearest-neighbour hamiltonian, and of the A's.
        
        C[n] depends on A[n] and A[n + 1].
        
        """
        if self.h_nn is None:
            return 0
        
        if n_low < 1:
            n_low = 0
        if n_high < 1:
            n_high = self.N + 1
        
        for n in xrange(n_low, n_high):
            self.C[n].fill(0)
            AA = sp.empty_like(self.C[n][0][0], order='A')
            for u in xrange(self.q[n]):
                for v in xrange(self.q[n + 1]):
                    mm.matmul(AA, self.A[n][u], self.A[n + 1][v]) #only do this once for each 
                    for s in xrange(self.q[n]):
                        for t in xrange(self.q[n + 1]):                
                            h_nn_stuv = self.h_nn(n, s, t, u, v)
                            if h_nn_stuv != 0:
                                self.C[n][s, t] += h_nn_stuv * AA
    
    def CalcK(self, n_low=-1, n_high=-1):
        """Generates the K matrices used to calculate the B's
        
        K[n] is recursively defined. It depends on C[m] and A[m] for all m >= n.
        
        It directly depends on A[n], A[n + 1], r[n], r[n + 1], C[n] and K[n + 1].
        
        This is equivalent to K on p. 14 of arXiv:1103.0936v2 [cond-mat.str-el], except 
        that it is for the non-gauge-preserving case, and includes a single-site
        Hamiltonian term.
        
        K[1] is, assuming a normalized state, the expectation value H of Ĥ.
        
        Instead of an explicit single-site term here, one could also include the 
        single-site Hamiltonian in the nearest-neighbour term, which may be more 
        efficient.
        """
        if n_low < 1:
            n_low = 0
        if n_high < 1:
            n_high = self.N + 1
            
        for n in reversed(xrange(n_low, n_high)):
            self.K[n].fill(0)
            tmp = sp.empty_like(self.K[n])
            
            for s in xrange(self.q[n]): 
                for t in xrange(self.q[n+1]):
                    self.K[n] += mm.matmul(tmp, self.C[n][s, t],
                                          self.r[n + 1], mm.H(self.A[n + 1][t]), 
                                          mm.H(self.A[n][s]))
                
                self.K[n] += mm.matmul(tmp, self.A[n][s], self.K[n + 1], 
                                      mm.H(self.A[n][s]))
        
    def Calc_K_luni(self, C, h_l):
        K_np1 = self.K[0]
        K_n = sp.empty_like(K_np1)
        r_np1 = None
        
        print (uni.adot(self.l[0], K_np1) / (self.N + 1)).real
        
        for n in reversed(xrange(-10000, 0)):
            K_n.fill(0)
            
            if not r_np1 is None and sp.allclose(r_np1/la.norm(r_np1),
                                                 self.r_l/la.norm(self.r_l),
                                                 atol=1E-10, rtol=1E-10):
                break
            else:
                r_np1 = self.Get_r(n + 1, r_np1)
            
            for s in xrange(self.q[0]): 
                for t in xrange(self.q[0]):
                    K_n += mm.matmul(None, C[s, t], r_np1, mm.H(self.A[0][t]), 
                                     mm.H(self.A[0][s]))
                
                K_n += mm.matmul(None, self.A[0][s], K_np1, mm.H(self.A[0][s]))
            
            h = (uni.adot(self.l[0], K_n) - h_l * abs(n)) / (self.N + 1.0)
            #Note: This becomes ever more inaccurate due to numerical error!!
            
            h_is_stable = sp.allclose(uni.adot(self.l[0], K_n), uni.adot(self.l[0], K_np1) + h_l,
                                      atol=1E-14, rtol=1E-14)
            
            if n < -1:
                print (n, h.real, uni.adot(self.l[0], K_n).real, 
                       (uni.adot(self.l[0], K_np1) + h_l).real, h_is_stable, 
                       la.norm(K_n/la.norm(K_n) - K_np1/la.norm(K_np1)), 
                       la.norm(r_np1/la.norm(r_np1) - self.r_l/la.norm(self.r_l)))
                if h_is_stable:
                    break
            
            K_np1 = K_n.copy()
            
        return h
    
    def BuildVsh(self, n, sqrt_r):
        """Generates mm.H(V[n][s]) for a given n, used for generating B[n][s]
        
        This is described on p. 14 of arXiv:1103.0936v2 [cond-mat.str-el] for left 
        gauge fixing. Here, we are using right gauge fixing.
        
        Array slicing and reshaping is used to manipulate the indices as necessary.
        
        Each V[n] directly depends only on A[n] and r[n].
        
        We return the conjugate mm.H(V) because we use it in more places than V.
        """
        R = sp.zeros((self.D[n], self.q[n], self.D[n-1]), dtype=self.typ, order='C')
        
        for s in xrange(self.q[n]):
            R[:,s,:] = mm.matmul(None, sqrt_r, mm.H(self.A[n][s]))

        R = R.reshape((self.q[n] * self.D[n], self.D[n-1]))
        V = mm.H(ns.nullspace(mm.H(R)))
        
        if self.sanity_checks:
            if not sp.allclose(mm.matmul(None, V, R), 0):
                print "Sanity Fail in BuildVsh!: VR_%u != 0" % (n)
            if not sp.allclose(mm.matmul(None, V, mm.H(V)), sp.eye(V.shape[0])):
                print "Sanity Fail in BuildVsh!: V H(V)_%u != eye" % (n)
        #print (q[n]*D[n] - D[n-1], q[n]*D[n])
        #print V.shape
        #print sp.allclose(mat(V) * mat(V).H, sp.eye(q[n]*D[n] - D[n-1]))
        #print sp.allclose(mat(V) * mat(Rh).H, 0)
        V = V.reshape((self.q[n] * self.D[n] - self.D[n - 1], self.D[n], self.q[n])) #this works with the above form for R
        
        #prepare for using V[s] and already take the adjoint, since we use it more often
        Vsh = sp.empty((self.q[n], self.D[n], self.q[n] * self.D[n] - self.D[n - 1]), dtype=self.typ, order=self.odr)
        for s in xrange(self.q[n]):
            Vsh[s] = mm.H(V[:,:,s])
        
        if self.sanity_checks:
            M = sp.zeros((self.q[n] * self.D[n] - self.D[n - 1], self.D[n]), dtype=self.typ)
            for s in xrange(self.q[n]):
                M += mm.matmul(None, mm.H(Vsh[s]), sqrt_r, mm.H(self.A[n][s]))
            if not sp.allclose(M, 0):
                print "Sanity Fail in BuildVsh!: Bad Vsh_%u" % (n)
        
        return Vsh
        
    def CalcOpt_x(self, n, Vsh, sqrt_l, sqrt_r, sqrt_l_inv, sqrt_r_inv):
        """Calculate the parameter matrix x* giving the desired B.
        
        This is equivalent to eqn. (49) of arXiv:1103.0936v2 [cond-mat.str-el] except 
        that, here, norm-preservation is not enforced, such that the optimal 
        parameter matrices x*_n (for the parametrization of B) are given by the 
        derivative w.r.t. x_n of <Phi[B, A]|Ĥ|Psi[A]>, rather than 
        <Phi[B, A]|Ĥ - H|Psi[A]> (with H = <Psi|Ĥ|Psi>).
        
        An additional sum was added for the single-site hamiltonian.
        
        Some multiplications have been pulled outside of the sums for efficiency.
        
        Direct dependencies: 
            - A[n - 1], A[n], A[n + 1]
            - r[n], r[n + 1], l[n - 2], l[n - 1]
            - C[n], C[n - 1]
            - K[n + 1]
            - V[n]
        """
        x = sp.zeros((self.D[n - 1], self.q[n] * self.D[n] - self.D[n - 1]), dtype=self.typ, order=self.odr)
        x_part = sp.empty_like(x)
        x_subpart = sp.empty_like(self.A[n][0])
        x_subsubpart = sp.empty_like(self.A[n][0])
        tmp = sp.empty_like(x_subpart)
        
        x_part.fill(0)
        for s in xrange(self.q[n]):
            x_subpart.fill(0)    
            
            if n < self.N + 1:
                x_subsubpart.fill(0)
                for t in xrange(self.q[n + 1]):
                    x_subsubpart += mm.matmul(tmp, self.C[n][s,t], self.r[n + 1], mm.H(self.A[n + 1][t])) #~1st line
                    
                x_subsubpart += mm.matmul(tmp, self.A[n][s], self.K[n + 1]) #~3rd line               
                
                x_subpart += mm.matmul(tmp, x_subsubpart, sqrt_r_inv)
            
            if not self.h_ext is None:
                x_subsubpart.fill(0)
                for t in xrange(self.q[n]):                         #Extra term to take care of h_ext..
                    x_subsubpart += self.h_ext(n, s, t) * self.A[n][t] #it may be more effecient to squeeze this into the nn term...
                x_subpart += mm.matmul(tmp, x_subsubpart, sqrt_r)
            
            x_part += mm.matmul(None, x_subpart, Vsh[s])
                
        x += mm.matmul(None, sqrt_l, x_part)
            
        if n > 0:
            l_nm2 = self.Get_l(n - 2)
            x_part.fill(0)
            for s in xrange(self.q[n]):     #~2nd line
                x_subsubpart.fill(0)
                for t in xrange(self.q[n + 1]):
                    x_subsubpart += mm.matmul(tmp, mm.H(self.A[n - 1][t]), l_nm2, self.C[n - 1][t, s])
                x_part += mm.matmul(None, x_subsubpart, sqrt_r, Vsh[s])
            x += mm.matmul(None, sqrt_l_inv, x_part)
                
        return x
        
    def GetB(self, n):
        """Generates the B[n] tangent vector corresponding to physical evolution of the state.
        
        In other words, this returns B[n][x*] (equiv. eqn. (47) of 
        arXiv:1103.0936v2 [cond-mat.str-el]) 
        with x* the parameter matrices satisfying the Euler-Lagrange equations
        as closely as possible.
        """
        if self.q[n] * self.D[n] - self.D[n - 1] > 0:
            l_sqrt, r_sqrt, l_sqrt_inv, r_sqrt_inv = self.Get_l_r_roots(n)
            
            Vsh = self.BuildVsh(n, r_sqrt)
            
            x = self.CalcOpt_x(n, Vsh, l_sqrt, r_sqrt, l_sqrt_inv, r_sqrt_inv)
            
            self.eta[n] = sp.sqrt(uni.adot(x, x))
            
            B = sp.empty_like(self.A[n])
            for s in xrange(self.q[n]):
                mm.matmul(B[s], l_sqrt_inv, x, mm.H(Vsh[s]), r_sqrt_inv)
                
            if self.sanity_checks:
                M = sp.zeros_like(self.r[n - 1])
                for s in xrange(self.q[n]):
                    M += mm.matmul(None, B[s], self.r[n], mm.H(self.A[n][s]))
                    
                if not sp.allclose(M, 0):
                    print "Sanity Fail in GetB!: B_%u does not satisfy GFC!" % n
            
            #print "eta_%u = %g" % (n, eta.real)
            
            return B
        else:
            return None, 0
        
    def Get_l_r_roots(self, n):
        """Returns the matrix square roots (and inverses) needed to calculate B.
        
        Hermiticity of l[n] and r[n] is used to speed this up.
        If an exception occurs here, it is probably because these matrices
        are no longer Hermitian (enough).
        """
        l_sqrt, evd = mm.sqrtmh(self.l[n - 1], ret_evd=True)
        l_sqrt_inv = mm.invmh(l_sqrt, evd=evd)

        r_sqrt, evd =  mm.sqrtmh(self.r[n], ret_evd=True)
        r_sqrt_inv = mm.invmh(r_sqrt, evd=evd)
        
        if self.sanity_checks:
            if not sp.allclose(mm.matmul(None, l_sqrt, l_sqrt), self.l[n - 1]):
                print "Sanity Fail in Get_l_r_roots: Bad l_sqrt_%u" % (n - 1)
            if not sp.allclose(mm.matmul(None, r_sqrt, r_sqrt), self.r[n]):
                print "Sanity Fail in Get_l_r_roots: Bad r_sqrt_%u" % (n)
            if not sp.allclose(mm.matmul(None, l_sqrt, l_sqrt_inv), sp.eye(l_sqrt.shape[0])):
                print "Sanity Fail in Get_l_r_roots: Bad l_sqrt_inv_%u" % (n - 1)
            if not sp.allclose(mm.matmul(None, r_sqrt, r_sqrt_inv), sp.eye(r_sqrt.shape[0])):
                print "Sanity Fail in Get_l_r_roots: Bad r_sqrt_inv_%u" % (n)
        
        return l_sqrt, r_sqrt, l_sqrt_inv, r_sqrt_inv
    
    def Update(self):
        self.BuildC()
        
        self.uGnd.A = self.A[self.N + 1]
        self.uGnd.r = self.r[self.N]
        self.uGnd.l = self.l_r
        self.uGnd.Calc_AA()
        self.uGnd.Calc_C()
        self.uGnd.Calc_K()
        self.K[self.N + 1][:] = self.uGnd.K
        
        self.CalcK()

        self.uGnd.A = self.A[0]
        self.uGnd.r = self.r_l
        self.uGnd.l = self.l[0]
        self.uGnd.Calc_AA()
        self.uGnd.Calc_C()
        K_left, h_left_uni = self.uGnd.Calc_K_left()
        
        h = (uni.adot(K_left, self.r[0]) + uni.adot(self.l[0], self.K[0])) / (self.N + 1)

        return h        
    
    def TakeStep(self, dtau): #simple, forward Euler integration     
        """Performs a complete forward-Euler step of imaginary time dtau.
        
        If dtau is itself imaginary, real-time evolution results.
        
        Here, the A's are updated as the sites are visited. Since we want all
        tangent vectors to be generated using the old state, we must delay
        updating each A[n] until we are *two* steps away (due to the direct
        dependency on A[n - 1] in CalcOpt_x).
        
        The dependencies on l, r, C and K are not a problem because we store
        all these matrices separately and do not update them at all during TakeStep().
        
        Parameters
        ----------
        dtau : complex
            The (imaginary or real) amount of imaginary time (tau) to step.
        """
        h = self.Update()
        
        eta_tot = 0
        
        B_prev = None
        for n in xrange(1, self.N + 2):
            #V is not always defined (e.g. at the right boundary vector, and possibly before)
            if n <= self.N:
                B = self.GetB(n)
                eta_tot += self.eta[n]
            
            if n > 1 and not B_prev is None:
                self.A[n - 1] += -dtau * B_prev
                
            B_prev = B
        
        return h, eta_tot
            
    def AddNoise(self, fac, n_i=-1, n_f=-1):
        """Adds some random noise of a given order to the state matrices A
        This can be used to determine the influence of numerical innaccuracies
        on quantities such as observables.
        """
        if n_i < 1:
            n_i = 1
        
        if n_f > self.N:
            n_f = self.N
        
        for n in xrange(n_i, n_f + 1):
            for s in xrange(self.q[n]):
                self.A[n][s].real += (sp.rand(self.D[n - 1], self.D[n]) - 0.5) * 2 * fac
                #self.A[n][s].imag += (sp.rand(self.D[n - 1], self.D[n]) - 0.5) * 2 * fac
                
    
    def Upd_l(self, start=-1, finish=-1):
        """Updates the l matrices using the current state.
        Implements step 5 of the TDVP algorithm or, equivalently, eqn. (41).
        (arXiv:1103.0936v2 [cond-mat.str-el])
        """
        if start < 0:
            start = 1
        if finish < 0:
            finish = self.N + 1
        for n in xrange(start, finish + 1):
            self.l[n].fill(0)
            tmp = sp.empty_like(self.l[n])
            for s in xrange(self.q[n]):
                self.l[n] += mm.matmul(tmp, mm.H(self.A[n][s]), self.l[n - 1], self.A[n][s])
    
    def Upd_r(self, n_low=-1, n_high=-1):
        """Updates the r matrices using the current state.
        Implements step 5 of the TDVP algorithm or, equivalently, eqn. (41).
        (arXiv:1103.0936v2 [cond-mat.str-el])
        """
        if n_low < 0:
            n_low = 0
        if n_high < 0:
            n_high = self.N - 1
        for n in reversed(xrange(n_low, n_high + 1)):
            self.EpsR(self.r[n], n + 1, self.r[n + 1], None)
    
    def EpsR(self, res, n, x, o):
        """Implements the right epsilon map
        
        FIXME: Ref.
        
        Parameters
        ----------
        res : ndarray
            A matrix to hold the result (with the same dimensions as r[n - 1]). May be None.
        n : int
            The site number.
        x : ndarray
            The argument matrix. For example, using r[n] (and o=None) gives a result r[n - 1]
        o : function
            The single-site operator to use. May be None.
    
        Returns
        -------
        res : ndarray
            The resulting matrix.
        """
        if n > self.N + 1:
            n = self.N + 1
        elif n < 0:
            n = 0
        
        if res is None:
            res = sp.zeros((self.D[n - 1], self.D[n - 1]), dtype=self.typ)
        else:
            res.fill(0)
        tmp = sp.empty_like(res)
        if o is None:
            for s in xrange(self.q[n]):
                res += mm.matmul(tmp, self.A[n][s], x, mm.H(self.A[n][s]))            
        else:
            for s in xrange(self.q[n]):
                for t in xrange(self.q[n]):
                    o_st = o(n, s, t)
                    if o_st != 0.:
                        mm.matmul(tmp, self.A[n][t], x, mm.H(self.A[n][s]))
                        tmp *= o_st
                        res += tmp
        return res
        
    def EpsR_2(self, n, x, op, A1=None, A2=None, A3=None, A4=None):
        if n > self.N:
            n = self.N + 1
            m = self.N + 1
        elif n < 0:
            n = 0
            m = 0
        else:
            m = n + 1
            
        if A1 is None:
            A1 = self.A[n]
        if A2 is None:
            A2 = self.A[m]
        if A3 is None:
            A3 = self.A[n]
        if A4 is None:
            A4 = self.A[m]
            
        res = sp.zeros((A1.shape[1], A1.shape[1]), dtype=self.typ)
        
        AAuvH = sp.empty_like(A3[0])
        for u in xrange(self.q[n]):
            for v in xrange(self.q[m]):
                mm.matmul(AAuvH, A3[u], A4[v])
                AAuvH = mm.H(AAuvH, out=AAuvH)
                subres = sp.zeros_like(A1[0])
                for s in xrange(self.q[n]):
                    for t in xrange(self.q[m]):
                        opval = op(n, u, v, s, t)
                        if opval != 0:
                            subres += opval * sp.dot(A1[s], A2[t])
                res += mm.matmul(subres, subres, x, AAuvH)
        
        return res
        
    def EpsL(self, res, n, x):
        """Implements the left epsilon map
        
        FIXME: Ref.
        
        Parameters
        ----------
        res : ndarray
            A matrix to hold the result (with the same dimensions as l[n]). May be None.
        n : int
            The site number.
        x : ndarray
            The argument matrix. For example, using l[n - 1] gives a result l[n]
    
        Returns
        -------
        res : ndarray
            The resulting matrix.
        """
        if n > self.N + 1:
            n = self.N + 1
        elif n < 0:
            n = 0        
        
        if res is None:
            res = sp.zeros_like(self.l[n])
        else:
            res.fill(0.)
        tmp = sp.empty_like(res)

        for s in xrange(self.q[n]):
            res += mm.matmul(tmp, mm.H(self.A[n][s]), x, self.A[n][s])
        return res
    
    def Restore_ON_R_n(self, n, G_n_i):
        """Transforms a single A[n] to obtain right orthonormalization.
        
        Implements the condition for right-orthonormalization from sub-section
        3.1, theorem 1 of arXiv:quant-ph/0608197v2.
        
        This function must be called for each n in turn, starting at N + 1,
        passing the gauge transformation matrix from the previous step
        as an argument.
        
        Finds a G[n-1] such that ON_R is fulfilled for n.
        
        Eigenvalues = 0 are a problem here... IOW rank-deficient matrices. 
        Apparently, they can turn up during a run, but if they do we're screwed.    
        
        The fact that M should be positive definite is used to optimize this.
        
        Parameters
        ----------
        n : int
            The site number.
        G_n_i : ndarray
            The inverse gauge transform matrix for site n obtained in the previous step (for n + 1).
    
        Returns
        -------
        G_n_m1_i : ndarray
            The inverse gauge transformation matrix for the site n - 1.
        """
        if G_n_i is None:
            GGh_n_i = self.r[n]
        else:
            GGh_n_i = mm.matmul(None, G_n_i, self.r[n], mm.H(G_n_i))
        
        M = self.EpsR(None, n, GGh_n_i, None)            
                    
        #The following should be more efficient than eigh():
        try:
            tu = la.cholesky(M) #Assumes M is pos. def.. It should raise LinAlgError if not.
            G_nm1 = mm.H(mm.invtr(tu)) #G is now lower-triangular
            G_nm1_i = mm.H(tu)
        except sp.linalg.LinAlgError:
            print "Restore_ON_R_n: Falling back to eigh()!"
            e,Gh = la.eigh(M)
            G_nm1 = mm.H(mm.matmul(None, Gh, sp.diag(1/sp.sqrt(e) + 0.j)))
            G_nm1_i = la.inv(G_nm1)
        
        if G_n_i is None:
            G_n_i = G_nm1_i
            
        if self.sanity_checks:
            if not sp.allclose(sp.dot(G_nm1, G_nm1_i), sp.eye(G_nm1.shape[0]), atol=1E-13, rtol=1E-13):
                print "Sanity Fail in Restore_ON_R_n!: Bad GT at n=%u" % n
        
        for s in xrange(self.q[n]):                
            mm.matmul(self.A[n][s], G_nm1, self.A[n][s], G_n_i)
            #It's ok to use the same matrix as out and as an operand here
            #since there are > 2 matrices in the chain and it is not the last argument.

        return G_nm1_i, G_nm1
        
    
    def Restore_RCF_dbg(self):
            for n in xrange(self.N + 2):
                print (n, sp.trace(self.l[n]).real, sp.trace(self.r[n]).real, uni.adot(self.l[n], self.r[n]).real)
                
            norm_r = uni.adot(self.l_r, self.r[self.N])
            norm_l = uni.adot(self.l[0], self.r_l)
            print norm_r
            print norm_l
            
            #r_m1 = self.EpsR(None, 0, self.r[0], None)
            #print uni.adot(self.l[0], r_m1).real            
            
            norm = uni.adot(self.l[0], self.r[0]).real
            
            h = sp.empty((self.N + 1), dtype=self.typ)
            for n in xrange(self.N + 1):
                h[n] = self.Expect_2S(self.h_nn, n)
            h *= 1/norm
            
            self.uGnd.A = self.A[0].copy()
            self.uGnd.l = self.l[0].copy()
            self.uGnd.r = self.r_l.copy()
            self.uGnd.Calc_AA()
            h_left = self.uGnd.Expect_2S(self.uGnd.h_nn) / norm_l

            self.uGnd.A = self.A[self.N + 1].copy()
            self.uGnd.l = self.l_r.copy()
            self.uGnd.r = self.r[self.N].copy()
            self.uGnd.Calc_AA()
            h_right = self.uGnd.Expect_2S(self.uGnd.h_nn) / norm_r
                
            return h, h_left, h_right
    
    def Restore_RCF(self, update_l=True, normalize=True, diag_l=True, dbg=False):
        if dbg:
            self.Upd_l()
            self.Upd_r()
            print "BEFORE..."
            h_before, h_left_before, h_right_before = self.Restore_RCF_dbg()
                
            print (h_left_before, h_before, h_right_before)
        
        G_n_i = None
        for n in reversed(xrange(1, self.N + 2)):
            G_n_i, G_n = self.Restore_ON_R_n(n, G_n_i)

            self.r[n - 1][:] = sp.eye(self.D[n - 1])

            if self.sanity_checks: #and not diag_l:
                r_n = mm.eyemat(self.D[n], self.typ)
                    
                r_nm1 = self.EpsR(None, n, r_n, None)
                if not sp.allclose(r_nm1, self.r[n - 1], atol=1E-13, rtol=1E-13):
                    print "Sanity Fail in Restore_RCF! p1: r_%u is bad" % (n - 1)
                    print la.norm(r_nm1 - self.r[n - 1])
                    
        self.r[self.N + 1] = self.r[self.N]
        
        #Now G_n_i contains g_0_i
        if self.sanity_checks:
            if not sp.allclose(sp.dot(G_n, G_n_i), sp.eye(G_n.shape[0]), atol=1E-13, rtol=1E-13):
                print "Sanity Fail in Restore_RCF!: Bad GT at n=0"
                
        for s in xrange(self.q[0]): #Note: This does not change the scale of A[0]
            self.A[0][s] = mm.matmul(None, G_n, self.A[0][s], G_n_i)            
            
        self.r_l[:] = mm.matmul(None, G_n, self.r_l, mm.H(G_n))
        self.l[0][:] = mm.matmul(None, mm.H(G_n_i), self.l[0], G_n_i)
        
        if dbg:
            self.Upd_l()
            print "MIDDLE..."
            h_mid, h_left_mid, h_right_mid = self.Restore_RCF_dbg()
                
            print (h_left_mid, h_mid, h_right_mid)     
        
        fac = 1 / sp.trace(self.l[0]).real
        if dbg:
            print "Scale l[0]: %g" % fac
        self.l[0] *= fac
        self.r_l *= 1/fac
                        
        if not diag_l:
            self.Upd_l()
        else:
            G_nm1 = None
            l_nm1 = self.l[0]
            for n in xrange(self.N + 1):
                if G_nm1 is None:
                    x = l_nm1
                else:
                    x = mm.matmul(None, mm.H(G_nm1), l_nm1, G_nm1)
                M = self.EpsL(None, n, x)
                ev, EV = la.eigh(M)
                
                self.l[n][:] = sp.diag(ev)
                G_n_i = EV
                
                if G_nm1 is None:
                    G_nm1 = mm.H(EV) #for left uniform case
                    
                for s in xrange(self.q[n]):                
                    mm.matmul(self.A[n][s], G_nm1, self.A[n][s], G_n_i)
                
                if self.sanity_checks:
                    l = self.EpsL(None, n, l_nm1)
                    if not sp.allclose(l, self.l[n], atol=1E-12, rtol=1E-12):
                        print "Sanity Fail in Restore_RCF!: l_%u is bad" % n
                
                G_nm1 = mm.H(EV)
                l_nm1 = self.l[n]
                
                if self.sanity_checks:
                    if not sp.allclose(sp.dot(G_nm1, G_n_i), sp.eye(G_n_i.shape[0]), atol=1E-12, rtol=1E-12):
                        print "Sanity Fail in Restore_RCF!: Bad GT for l_%u" % n
            
            #Now G_nm1 = G_N
            G_nm1_i = mm.H(G_nm1)
            for s in xrange(self.q[self.N + 1]):
                self.A[self.N + 1][s] = mm.matmul(None, G_nm1, self.A[self.N + 1][s], G_nm1_i)
                
            self.r[self.N][:] = mm.matmul(None, G_nm1, self.r[self.N], mm.H(G_nm1))            
            self.l_r[:] = mm.matmul(None, mm.H(G_nm1_i), self.l_r, G_nm1_i)
            
            self.uGnd.A = self.A[0]
            self.uGnd.r = self.r_l
            self.uGnd.l = self.l[0]
            self.uGnd.Calc_lr() #Ensures largest ev of E=1
            fac = 1 / sp.trace(self.l[0]).real
            if dbg:
                print "Scale l[0]: %g" % fac
            self.l[0] *= fac
            self.r_l *= 1/fac
            
            self.uGnd.A = self.A[self.N + 1]
            self.uGnd.r = self.r[self.N]
            self.uGnd.l = self.l_r
            self.uGnd.Calc_lr() #Ensures largest ev of E=1
            
            #Calc_lr() enforces trace(r) == 1, so we don't need to re-scale r[N]
                       
            self.l[self.N + 1][:] = self.EpsL(None, self.N + 1, self.l[self.N])
            
        if self.sanity_checks:
            l_n = self.l[0]
            for n in xrange(0, self.N + 1):
                l_n = self.EpsL(None, n, l_n)
                if not sp.allclose(l_n, self.l[n], atol=1E-12, rtol=1E-12):
                    print "Sanity Fail in Restore_RCF! p2: l_%u is bad" % n
                    
            r_nm1 = self.r[self.N + 1]
            for n in reversed(xrange(1, self.N + 2)):
                r_nm1 = self.EpsR(None, n, r_nm1, None)
                if not sp.allclose(r_nm1, self.r[n - 1], atol=1E-12, rtol=1E-12):
                    print "Sanity Fail in Restore_RCF! p2: r_%u is bad" % (n - 1)
        if dbg:
            print "AFTER..."
            h_after, h_left_after, h_right_after = self.Restore_RCF_dbg()
                
            print (h_left_after, h_after, h_right_after)  
            
            print h_after - h_before
                
            print h_after.sum() - h_before.sum()
    
    def CheckCanonical_R(self):
        """Tests for right canonical form.
        Uses the criteria listed in sub-section 3.1, theorem 1 of arXiv:quant-ph/0608197v2.
        """
        rnsOK = True
        ls_trOK = True
        ls_herm = True
        ls_pos = True
        ls_diag = True
        
        for n in xrange(1, self.N):
            rnsOK = rnsOK and sp.allclose(self.r[n], sp.eye(self.r[n].shape[0]), atol=self.eps*2, rtol=0)
        for n in xrange(2, self.N + 1):
            ls_herm = ls_herm and sp.allclose(self.l[n] - mm.H(self.l[n]), 0, atol=self.eps*2)
            ls_trOK = ls_trOK and sp.allclose(sp.trace(self.l[n]), 1, atol=self.eps*2, rtol=0)
            ls_pos = ls_pos and all(la.eigvalsh(self.l[n]) > 0)
            ls_diag = ls_diag and sp.allclose(self.l[n], sp.diag(self.l[n].diagonal()))
        
        normOK = sp.allclose(self.l[self.N], 1., atol=self.eps, rtol=0)
        
        return (rnsOK, ls_trOK, ls_pos, ls_diag, normOK)
    
    def Grow_left(self, m):
        oldA = self.A
        oldl_0 = self.l[0]
        oldr_N = self.r[self.N]
            
        self.N = self.N + m
        self._init_arrays()
        
        for n in xrange(m + 1):
            self.A[n][:] = oldA[0]
            
        for n in xrange(m + 1, self.N + 2):
            self.A[n][:] = oldA[n - m]
            
        self.l[0][:] = oldl_0
        self.r[self.N][:] = oldr_N
    
    def Get_l(self, n):
        if 0 <= n <= self.N + 1:
            return self.l[n]
        elif n < 0:
            return self.l[0]
        else:
            l_m = self.l[self.N + 1]
            for m in xrange(self.N + 2, n + 1):
                l_m = self.EpsL(None, self.N + 1, l_m)
            return l_m
            
    def Get_r(self, n, r_np1=None):
        if 0 <= n <= self.N + 1:
            return self.r[n]
        elif n > self.N + 1:
            return self.r[self.N + 1]
        else:
            if r_np1 is None:
                r_m = self.r[0]
                for m in reversed(xrange(n, 0)):
                    r_m = self.EpsR(None, 0, r_m, None)
                return r_m
            else:
                return self.EpsR(None, 0, r_np1, None)
    
    def Expect_SS(self, o, n):
        """Computes the expectation value of a single-site operator.
        
        A single-site operator is represented as a function taking three
        integer arguments (n, s, t) where n is the site number and s, t 
        range from 0 to q[n] - 1 and define the requested matrix element <s|o|t>.
        
        Assumes that the state is normalized.
        
        Parameters
        ----------
        o : function
            The operator.
        n : int
            The site number.
        """
        res = self.EpsR(None, n, self.Get_r(n), o)
        return uni.adot(self.Get_l(n - 1), res)
        
    def Expect_2S(self, o, n):
        res = self.EpsR_2(n, self.Get_r(n + 1), o)
        return uni.adot(self.Get_l(n - 1), res)
        
    def Expect_SS_Cor(self, o1, o2, n1, n2):
        """Computes the correlation of two single site operators acting on two different sites.
        
        See Expect_SS().
        
        n1 must be smaller than n2.
        
        Assumes that the state is normalized.
        
        Parameters
        ----------
        o1 : function
            The first operator, acting on the first site.
        o2 : function
            The second operator, acting on the second site.
        n1 : int
            The site number of the first site.
        n2 : int
            The site number of the second site (must be > n1).
        """        
        r_n = self.EpsR(None, n2, self.r[n2], o2)

        for n in reversed(xrange(n1 + 1, n2)):
            r_n = self.EpsR(None, n, r_n, None)

        r_n = self.EpsR(None, n1, r_n, o1)   
         
        res = mm.matmul(None, self.l[n1 - 1], r_n)
        return res.trace()
        
    def DensityMatrix_2S(self, n1, n2):
        """Returns a reduced density matrix for a pair of sites.
        
        Parameters
        ----------
        n1 : int
            The site number of the first site.
        n2 : int
            The site number of the second site (must be > n1).        
        """
        rho = sp.empty((self.q[n1] * self.q[n2], self.q[n1] * self.q[n2]), dtype=sp.complex128)
        r_n2 = sp.empty_like(self.r[n2 - 1])
        r_n1 = sp.empty_like(self.r[n1 - 1])
        tmp = sp.empty_like(self.r[n1 - 1])
        
        for s2 in xrange(self.q[n2]):
            for t2 in xrange(self.q[n2]):
                mm.matmul(r_n2, self.A[n2][t2], self.r[n2], mm.H(self.A[n2][s2]))
                
                r_n = r_n2
                for n in reversed(xrange(n1 + 1, n2)):
                    r_n = self.EpsR(None, n, r_n, None)        
                    
                for s1 in xrange(self.q[n1]):
                    for t1 in xrange(self.q[n1]):
                        mm.matmul(r_n1, self.A[n1][t1], r_n, mm.H(self.A[n1][s1]))
                        mm.matmul(tmp, self.l[n1 - 1], r_n1)
                        rho[s1 * self.q[n1] + s2, t1 * self.q[n1] + t2] = tmp.trace()
        return rho
    
    def SaveState(self, file):
        sp.save(file, self.A)
        
    def LoadState(self, file):
        self.A = sp.load(file)
