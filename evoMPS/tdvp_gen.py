# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 17:29:27 2011

@author: Ashley Milsted

TODO:
    - Implement evaluation of the error due to restriction to bond dim.
    - Add an algorithm for expanding the bond dimension.
    - Adaptive step size.

"""
import scipy as sp
import scipy.linalg as la
import scipy.optimize as opti
import matmul as m
import tdvp_common as tm
import copy
from mps_gen import EvoMPS_MPS_Generic

class EvoMPS_TDVP_Generic(EvoMPS_MPS_Generic):
            
    def __init__(self, N, D, q, h_nn):
        """Creates a new EvoMPS_TDVP_Generic object.
        
        This class implements the time-dependent variational principle (TDVP) for
        matrix product states (MPS) of a finite spin chain with open boundary
        conditions.
        
        It is derived from EvoMPS_MPS_Generic, which implements basic operations
        on the state, adding the ability to integrate the TDVP flow equations
        given a nearest-neighbour Hamiltonian.
        
        Performs EvoMPS_MPS_Generic.__init__().
        
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
        h_nn : array or callable
            Hamiltonian term for each site ham(n, s, t, u, v) or 
            ham[n][s, t, u, v] for site n.
         
        """       
        self.h_nn = h_nn
        self.ham_sites = 2
            
        super(EvoMPS_TDVP_Generic, self).__init__(N, D, q)
    
    
    def _init_arrays(self):
        super(EvoMPS_TDVP_Generic, self)._init_arrays()
        
        #Make indicies correspond to the thesis
        self.K = sp.empty((self.N + 1), dtype=sp.ndarray) #Elements 1..N
        self.C = sp.empty((self.N), dtype=sp.ndarray) #Elements 1..N-1 

        for n in xrange(1, self.N + 1):
            self.K[n] = sp.zeros((self.D[n - 1], self.D[n - 1]), dtype=self.typ, order=self.odr)    
            if n <= self.N - self.ham_sites + 1:
                ham_shape = []
                for i in xrange(self.ham_sites):
                    ham_shape.append(self.q[n + i])
                C_shape = tuple(ham_shape + [self.D[n - 1], self.D[n - 1 + self.ham_sites]])
                self.C[n] = sp.empty(C_shape, dtype=self.typ, order=self.odr)
        
        self.eta = sp.zeros((self.N + 1), dtype=self.typ)
        
        self.h_expect = sp.zeros((self.N + 1), dtype=self.typ)
        self.H_expect = 0
    
    
    def calc_C(self, n_low=-1, n_high=-1):
        """Generates the C tensors used to calculate the K's and ultimately the B's.
        
        This is called automatically by self.update().
        
        C[n] contains a contraction of the Hamiltonian self.h_nn with the parameter
        tensors A[n] and A[n + 1] over the local basis indices.
        
        This is prerequisite for calculating the tangent vector parameters B,
        which optimally approximate the exact time evolution.
        
        These are to be used on one side of the super-operator when applying the
        nearest-neighbour Hamiltonian, similarly to C in eqn. (44) of 
        arXiv:1103.0936v2 [cond-mat.str-el], except being for the non-norm-preserving case.

        Makes use only of the nearest-neighbour Hamiltonian, and of the A's.
        
        C[n] depends on A[n] and A[n + 1].
        
        """
        if self.h_nn is None:
            return 0
        
        if n_low < 1:
            n_low = 1
        if n_high < 1:
            n_high = self.N - self.ham_sites + 1
        
        for n in xrange(n_low, n_high + 1):
            if callable(self.h_nn):
                h_nn = lambda *args: self.h_nn(n, *args)
                h_nn = sp.vectorize(h_nn, otypes=[sp.complex128])
                h_nn = sp.fromfunction(h_nn, tuple(self.C[n].shape[:-2] * 2))
            else:
                h_nn = self.h_nn[n]
            
            AA = tm.calc_AA(self.A[n], self.A[n + 1])
            self.C[n] = tm.calc_C_mat_op_AA(h_nn, AA)
         
    
    def calc_K(self, n_low=-1, n_high=-1):
        """Generates the K matrices used to calculate the B's.
        
        This is called automatically by self.update().
        
        K[n] is contains the action of the Hamiltonian on sites n to N.
        
        K[n] is recursively defined. It depends on C[m] and A[m] for all m >= n.
        
        It directly depends on A[n], A[n + 1], r[n], r[n + 1], C[n] and K[n + 1].
        
        This is equivalent to K on p. 14 of arXiv:1103.0936v2 [cond-mat.str-el], except 
        that it is for the non-norm-preserving case.
        
        K[1] is, assuming a normalized state, the expectation value H of Ĥ.
        """
        if n_low < 1:
            n_low = 1
        if n_high < 1:
            n_high = self.N
            
        for n in reversed(xrange(n_low, n_high + 1)):
            if n <= self.N - self.ham_sites + 1:
                self.K[n], ex = tm.calc_K(self.K[n + 1], self.C[n], self.l[n - 1], 
                                          self.r[n + 1], self.A[n], self.A[n + 1], 
                                          sanity_checks=self.sanity_checks)                   
                self.h_expect[n] = ex
            else:
                self.K[n].fill(0)
                
        if n_low == 1:
            self.H_expect = sp.asscalar(self.K[1])
    
    def update(self, restore_RCF=True):
        """Updates secondary quantities to reflect the state parameters self.A.
        
        Must be used after taking a step or otherwise changing the 
        parameters self.A before calculating
        physical quantities or taking the next step.
        
        Also (optionally) restores the right canonical form.
        
        Parameters
        ----------
        restore_RCF : bool (True)
            Whether to restore right canonical form.
        """
        super(EvoMPS_TDVP_Generic, self).update(restore_RCF)
        self.calc_C()
        self.calc_K()
        
    def calc_x(self, n, Vsh, sqrt_l, sqrt_r, sqrt_l_inv, sqrt_r_inv):
        """Calculates the matrix x* that results in the TDVP tangent vector B.
        
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
        if n > 1:
            lm2 = self.l[n - 2]
            Cm1 = self.C[n - 1]
            Am1 = self.A[n - 1]
        else:
            lm2 = None
            Cm1 = None
            Am1 = None
                        
        if n <= self.N - self.ham_sites + 1:
            C = self.C[n]            
        else:
            C = None
            
        if n < self.N:
            Kp1 = self.K[n + 1]
            rp1 = self.r[n + 1]
            Ap1 = self.A[n + 1]
        else:
            Kp1 = None
            rp1 = None
            Ap1 = None

        x = tm.calc_x(Kp1, C, Cm1, rp1,
                      lm2, Am1, self.A[n], Ap1,
                      sqrt_l, sqrt_l_inv, sqrt_r, sqrt_r_inv, Vsh)
                
        return x
        
    def calc_B(self, n, set_eta=True):
        """Generates the TDVP tangent vector parameters for a single site B[n].
        
        A TDVP time step is defined as: A[n] -= dtau * B[n]
        where dtau is an infinitesimal imaginary time step.
        
        In other words, this returns B[n][x*] (equiv. eqn. (47) of 
        arXiv:1103.0936v2 [cond-mat.str-el]) 
        with x* the parameter matrices satisfying the Euler-Lagrange equations
        as closely as possible.
        
        Returns
        -------
            B_n : ndarray or None
                The TDVP tangent vector parameters for site n or None
                if none is defined.
        """
        if self.q[n] * self.D[n] - self.D[n - 1] > 0:
            l_sqrt, l_sqrt_inv, r_sqrt, r_sqrt_inv = tm.calc_l_r_roots(self.l[n - 1], 
                                                                   self.r[n], 
                                                                   self.sanity_checks)
            
            Vsh = tm.calc_Vsh(self.A[n], r_sqrt, sanity_checks=self.sanity_checks)
            
            x = self.calc_x(n, Vsh, l_sqrt, r_sqrt, l_sqrt_inv, r_sqrt_inv)
            
            if set_eta:
                self.eta[n] = sp.sqrt(m.adot(x, x))
    
            B = sp.empty_like(self.A[n])
            for s in xrange(self.q[n]):
                B[s] = m.mmul(l_sqrt_inv, x, m.H(Vsh[s]), r_sqrt_inv)
            return B
        else:
            return None

    def take_step(self, dtau, B=None):   
        """Performs a complete forward-Euler step of imaginary time dtau.
        
        The operation is A[n] -= dtau * B[n] with B[n] form self.calc_B(n).
        
        If dtau is itself imaginary, real-time evolution results.
        
        We avoid storing all the B[n] at once by updating self.A[n] as
        we move along the chain. Dependencies must be carefully considered.
        Since we assume a nearest-neighbour Hamiltonian, we must not
        alter A[n - 1] before calculating B[n]. Hence we must store
        two B[n] tensors at any one time.
        
        The dependencies on l, r, C and K are not a problem because we store
        all these matrices separately and do not update them at all during take_step().
        
        Parameters
        ----------
        dtau : complex
            The (imaginary or real) amount of imaginary time (tau) to step.
        """
        if B is None:
            eta_tot = 0
            
            B_prev = None
            for n in xrange(1, self.N + 2):
                #V is not always defined (e.g. at the right boundary vector, and possibly before)
                if n <= self.N:
                    B = self.calc_B(n)
                    eta_tot += self.eta[n]
                
                if n > 1 and not B_prev is None:
                    self.A[n - 1] += -dtau * B_prev
                    
                B_prev = B
                
            return eta_tot
        else:
            for n in xrange(1, self.N + 1):
                if not B[n] is None:
                    self.A[n] += -dtau * B[n]

        
    def take_step_RK4(self, dtau):
        """Take a step using the fourth-order explicit Runge-Kutta method.
        
        This requires more memory than a simple forward Euler step. 
        It is, however, far more accurate with a per-step error of
        order dtau**4.
        
        Parameters
        ----------
        dtau : complex
            The (imaginary or real) amount of imaginary time (tau) to step.
        """
        def upd():
            self.calc_l()
            self.calc_r()
            self.calc_C()
            self.calc_K()            

        eta_tot = 0

        #Take a copy of the current state
        A0 = sp.empty_like(self.A)
        for n in xrange(1, self.N + 1):
            A0[n] = self.A[n].copy()

        B_fin = sp.empty_like(self.A)

        B_prev = None
        for n in xrange(1, self.N + 2):
            if n <= self.N:
                B = self.calc_B(n) #k1
                eta_tot += self.eta[n]
                B_fin[n] = B

            if not B_prev is None:
                self.A[n - 1] = A0[n - 1] - dtau/2 * B_prev

            B_prev = B

        upd()

        B_prev = None
        for n in xrange(1, self.N + 2):
            if n <= self.N:
                B = self.calc_B(n, set_eta=False) #k2

            if not B_prev is None:
                self.A[n - 1] = A0[n - 1] - dtau/2 * B_prev
                B_fin[n - 1] += 2 * B_prev

            B_prev = B

        upd()

        B_prev = None
        for n in xrange(1, self.N + 2):
            if n <= self.N:
                B = self.calc_B(n, set_eta=False) #k3

            if not B_prev is None:
                self.A[n - 1] = A0[n - 1] - dtau * B_prev
                B_fin[n - 1] += 2 * B_prev

            B_prev = B

        upd()

        for n in xrange(1, self.N + 1):
            B = self.calc_B(n, set_eta=False) #k4
            if not B is None:
                B_fin[n] += B

        for n in xrange(1, self.N + 1):
            if not B_fin[n] is None:
                self.A[n] = A0[n] - dtau /6 * B_fin[n]

        return eta_tot
        
    def calc_B_CG(self, B_CG_0, eta_0, dtau_init):
        """Calculates a tangent vector using the non-linear conjugate gradient method.
        
        Parameters:
            B_CG_0 : ndarray
                Tangent vector used to make the previous step. Setting to None
                triggers reset.
            eta_0 : float
                Norm of the previous tangent vector.
            dtau_init : float
                Initial step-size for the line-search.
                
        Returns:
            B_CG, B, eta, tau
        """
        B = sp.empty_like(self.A)
        for n in xrange(1, self.N + 1):
            B[n] = self.calc_B(n)
            
        eta = self.eta.real.sum()
        
        if B_CG_0 is None:
            beta = 0.
            print "RESET CG"
            
            B_CG = B
        else:
            beta = (eta**2) / eta_0**2
        
            print "BetaFR = " + str(beta)
        
            beta = max(0, beta.real)
            
            B_CG = sp.empty_like(self.A)
            for n in xrange(1, self.N + 1):
                if not B[n] is None:
                    B_CG[n] = B[n] + beta * B_CG_0[n]

        old_h = self.H_expect.real
        tau, h_min = self.find_min_h_brent(B_CG, dtau_init,
                                           trybracket=False)
            
        if old_h < h_min:
            print "RESET due to energy rise!"
            B_CG = B
            tau, h_min = self.find_min_h_brent(B_CG, dtau_init * 0.1, trybracket=False)
        
            if old_h < h_min:
                print "RESET FAILED: Setting tau=0!"
                tau = 0
        
        return B_CG, B, eta, tau
        
    def find_min_h_brent(self, B, dtau_init, tol=5E-2, skipIfLower=False, 
                         trybracket=True):
        taus=[]
        hs=[]
        
        h_before = self.H_expect.real
        
        def f(tau, *args):
            if tau == 0:
                print (0, "tau=0")
                return h_before              
            try:
                i = taus.index(tau)
                print (tau, hs[i], hs[i] - h_before, "from stored")
                return hs[i]
            except ValueError:
                for n in xrange(1, self.N + 1):
                    if not B[n] is None:
                        self.A[n] = A0[n] - tau * B[n]

                self.calc_l()
                self.calc_r()
                self.simple_renorm()
                self.calc_C()
                self.calc_K()
                h = self.H_expect
                
                print (tau, h.real, h.real - h_before)
                
                res = h.real
                
                taus.append(tau)
                hs.append(res)
                
                return res
        
        #TODO: Generalize this?
        A0 = copy.deepcopy(self.A)
        C0 = copy.deepcopy(self.C)
        K0 = copy.deepcopy(self.K)
        l0 = copy.deepcopy(self.l)
        r0 = copy.deepcopy(self.r)
        
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
        self.C = C0
        self.K = K0
        
        return tau_opt, h_min
