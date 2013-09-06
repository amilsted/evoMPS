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
import matmul as m
import tdvp_common as tm
import tdvp_common_cuda as tmc
from mps_gen import EvoMPS_MPS_Generic

class EvoMPS_TDVP_Generic(EvoMPS_MPS_Generic):
            
    def __init__(self, N, D, q, ham, ham_sites=None):
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
        ham : array or callable
            Hamiltonian term for each site ham(n, *physical_indices) or 
            ham[n][*physical indices] for site n.
         
        """       

        self.ham = ham
        """The Hamiltonian. Can be changed, for example, to perform
           a quench. The number of neighbouring sites acted on must be 
           specified in ham_sites."""
        
        if ham_sites is None:
            if not callable(ham):
                self.ham_sites = len(ham[1].shape) / 2
            else:
                self.ham_sites = 2
        else:
            self.ham_sites = ham_sites
        
        if not (self.ham_sites == 2 or self.ham_sites == 3):
            raise ValueError("Only 2 or 3 site Hamiltonian terms supported!")
            
        super(EvoMPS_TDVP_Generic, self).__init__(N, D, q)
        
        self.gauge_fixing = self.canonical_form
        
    
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
        """The per-site contributions to the norm of the TDVP tangent vector 
           (projection of the exact time evolution onto the MPS tangent plane. 
           Only available after calling take_step()."""
        self.eta.fill(sp.NaN)
        
        self.h_expect = sp.zeros((self.N + 1), dtype=self.typ)
        """The local energy expectation values (of each Hamiltonian term), 
           available after calling update() or calc_K()."""
        self.h_expect.fill(sp.NaN)
           
        self.H_expect = sp.NaN
        """The energy expectation value, available after calling update()
           or calc_K()."""

    
    def calc_C(self, n_low=-1, n_high=-1):
        """Generates the C tensors used to calculate the K's and ultimately the B's.
        
        This is called automatically by self.update().
        
        C[n] contains a contraction of the Hamiltonian self.ham with the parameter
        tensors over the local basis indices.
        
        This is prerequisite for calculating the tangent vector parameters B,
        which optimally approximate the exact time evolution.
        
        These are to be used on one side of the super-operator when applying the
        nearest-neighbour Hamiltonian, similarly to C in eqn. (44) of 
        arXiv:1103.0936v2 [cond-mat.str-el], for the non-norm-preserving case.

        Makes use only of the nearest-neighbour Hamiltonian, and of the A's.
        
        C[n] depends on A[n] through A[n + self.ham_sites - 1].
        
        """
        if self.ham is None:
            return 0
        
        if n_low < 1:
            n_low = 1
        if n_high < 1:
            n_high = self.N - self.ham_sites + 1
        
        for n in xrange(n_low, n_high + 1):
            if callable(self.ham):
                ham_n = lambda *args: self.ham(n, *args)
                ham_n = sp.vectorize(ham_n, otypes=[sp.complex128])
                ham_n = sp.fromfunction(ham_n, tuple(self.C[n].shape[:-2] * 2))
            else:
                ham_n = self.ham[n]
                
            if self.ham_sites == 2:
                AA = tm.calc_AA(self.A[n], self.A[n + 1])
                self.C[n] = tm.calc_C_mat_op_AA(ham_n, AA)
            else:
                AAA = tm.calc_AAA(self.A[n], self.A[n + 1], self.A[n + 2])
                self.C[n] = tm.calc_C_3s_mat_op_AAA(ham_n, AAA)                
    
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
                if self.ham_sites == 2:
                    self.K[n], ex = tm.calc_K(self.K[n + 1], self.C[n], self.l[n - 1], 
                                              self.r[n + 1], self.A[n], self.A[n + 1], 
                                              sanity_checks=self.sanity_checks)
                else:
                    self.K[n], ex = tm.calc_K_3s(self.K[n + 1], self.C[n], self.l[n - 1], 
                                              self.r[n + 2], self.A[n], self.A[n + 1], 
                                              self.A[n + 2],
                                              sanity_checks=self.sanity_checks)

                self.h_expect[n] = ex
            else:
                self.K[n].fill(0)
                
        if n_low == 1:
            self.H_expect = sp.asscalar(self.K[1])
            
            
    def calc_K_l(self, n_low=-1, n_high=-1):
        """Generates the K matrices used to calculate the B's.
        For the left gauge-fixing case.
        """
        if n_low < 2:
            n_low = 2
        if n_high < 1:
            n_high = self.N
            
        self.K[1] = sp.zeros((self.D[1], self.D[1]), dtype=self.typ)
            
        for n in xrange(n_low, n_high + 1):
            #if n <= self.N - self.ham_sites + 1:
            if self.ham_sites == 2:
                self.K[n], ex = tm.calc_K_l(self.K[n - 1], self.C[n - 1], self.l[n - 2], 
                                          self.r[n], self.A[n], self.A[n - 1], 
                                          sanity_checks=self.sanity_checks)
            else:
                assert False, 'left gauge fixing not yet supported for three-site Hamiltonians'

            self.h_expect[n - 1] = ex
            #else:
            #    self.K[n].fill(0)
                
        if n_high == self.N:
            self.H_expect = sp.asscalar(self.K[self.N])
            

    def update(self, restore_CF=True, normalize=True, auto_truncate=False, restore_CF_after_trunc=True):
        """Updates secondary quantities to reflect the state parameters self.A.
        
        Must be used after taking a step or otherwise changing the 
        parameters self.A before calculating
        physical quantities or taking the next step.
        
        Also (optionally) restores the canonical form.
        
        Parameters
        ----------
        restore_CF : bool
            Whether to restore canonical form.
        normalize : bool
            Whether to normalize the state in case restore_CF is False.
        auto_truncate : bool
            Whether to automatically truncate the bond-dimension if
            rank-deficiency is detected. Requires restore_CF.
        restore_CF_after_trunc : bool
            Whether to restore_CF after truncation.

        Returns
        -------
        truncated : bool (only if auto_truncate == True)
            Whether truncation was performed.
        """
        trunc = super(EvoMPS_TDVP_Generic, self).update(restore_CF=restore_CF, 
                                                        normalize=normalize,
                                                        auto_truncate=auto_truncate,
                                                        restore_CF_after_trunc=restore_CF_after_trunc)
        self.calc_C()
        if self.gauge_fixing == 'right':
            self.calc_K()
        else:
            self.calc_K_l()
        return trunc
        
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
            
        if n > 2:
            lm3 = self.l[n - 3]
            Cm2 = self.C[n - 2]
            Am2 = self.A[n - 2]
        else:
            lm3 = None
            Cm2 = None
            Am2 = None
            
        if n <= self.N - self.ham_sites + 1:
            C = self.C[n]            
        else:
            C = None
            
        if n + 1 <= self.N - self.ham_sites + 1:
            Kp1 = self.K[n + 1]            
        else:
            Kp1 = None
            
        if n < self.N - 1:
            Ap2 = self.A[n + 2]
            rp2 = self.r[n + 2]
        else:
            Ap2 = None
            rp2 = None
            
        if n < self.N:
            rp1 = self.r[n + 1]
            Ap1 = self.A[n + 1]
        else:
            rp1 = None
            Ap1 = None
        
        if self.ham_sites == 2:
            x = tmc.calc_x(Kp1, C, Cm1, rp1,
                          lm2, Am1, self.A[n], Ap1,
                          sqrt_l, sqrt_l_inv, sqrt_r, sqrt_r_inv, Vsh)
            #x_ = tm.calc_x(Kp1, C, Cm1, rp1,
            #              lm2, Am1, self.A[n], Ap1,
            #              sqrt_l, sqrt_l_inv, sqrt_r, sqrt_r_inv, Vsh)
            #assert sp.allclose(x_, x)
        else:
            x = tm.calc_x_3s(Kp1, C, Cm1, Cm2, rp1, rp2, lm2, 
                             lm3, Am2, Am1, self.A[n], Ap1, Ap2,
                             sqrt_l, sqrt_l_inv, sqrt_r, sqrt_r_inv, Vsh)
                
        return x
    
    def calc_x_l(self, n, Vsh, sqrt_l, sqrt_r, sqrt_l_inv, sqrt_r_inv):
        if n > 1:
            Km1 = self.K[n - 1]
            Cm1 = self.C[n - 1]
            Am1 = self.A[n - 1]
            lm2 = self.l[n - 2]
        else:
            Km1 = None
            Cm1 = None
            Am1 = None
            lm2 = None
            
        if n < self.N:
            Ap1 = self.A[n + 1]
            rp1 = self.r[n + 1]
        else:
            Ap1 = None
            rp1 = None
            
        if n <= self.N - self.ham_sites + 1:
            C = self.C[n]            
        else:
            C = None
        
        if self.ham_sites == 2:
            x = tm.calc_x_l(Km1, C, Cm1, rp1,
                          lm2, Am1, self.A[n], Ap1,
                          sqrt_l, sqrt_l_inv, sqrt_r, sqrt_r_inv, Vsh)
        else:
            assert False, "left gauge-fixing not yet supported for three-site Hamiltonians"
            
        return x        
        

    def calc_B_all(self, set_eta=True):
        l_s = []
        l_si = []
        r_s = [None]
        r_si = [None]
        Vsh = [None]
        for n in xrange(1, self.N + 1):
            if self.q[n] * self.D[n] - self.D[n - 1] > 0:
                l_s_nm1, l_si_nm1, r_s_n, r_si_n = tm.calc_l_r_roots(self.l[n - 1], 
                                                                     self.r[n], 
                                                                     self.sanity_checks)
                l_s.append(l_s_nm1)
                l_si.append(l_si_nm1)
                r_s.append(r_s_n)
                r_si.append(r_si_n)
                            
                Vsh_n = tm.calc_Vsh(self.A[n], r_s_n, sanity_checks=self.sanity_checks)
                
                Vsh.append(Vsh_n)
            else:
                l_s.append(None)
                l_si.append(None)
                r_s.append(None)
                r_si.append(None)
                
                Vsh.append(None)
               
        l_s.append(None)
        l_si.append(None)
               
        Bs = tmc.calc_Bs(self.N, self.A, self.l, l_s, l_si, self.r, r_s, r_si,
                         self.C, self.K, Vsh)
                         
        self.eta.fill(1)
                
        return Bs
        
    
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
        if self.gauge_fixing == 'right':
            return self._calc_B_r(n, set_eta=set_eta)
        else:
            return self._calc_B_l(n, set_eta=set_eta)
    
    def _calc_B_r(self, n, set_eta=True):
        if self.q[n] * self.D[n] - self.D[n - 1] > 0:
            l_sqrt, l_sqrt_inv, r_sqrt, r_sqrt_inv = tm.calc_l_r_roots(self.l[n - 1], 
                                                                   self.r[n], 
                                                                   zero_tol=self.zero_tol,
                                                                   sanity_checks=self.sanity_checks,
                                                                   sc_data=('site', n))
            
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

    def _calc_B_l(self, n, set_eta=True):
        if self.q[n] * self.D[n - 1] - self.D[n] > 0:
            l_sqrt, l_sqrt_inv, r_sqrt, r_sqrt_inv = tm.calc_l_r_roots(self.l[n - 1], 
                                                                   self.r[n], 
                                                                   zero_tol=self.zero_tol,
                                                                   sanity_checks=self.sanity_checks,
                                                                   sc_data=('site', n))
            
            Vsh = tm.calc_Vsh_l(self.A[n], l_sqrt, sanity_checks=self.sanity_checks)
            
            x = self.calc_x_l(n, Vsh, l_sqrt, r_sqrt, l_sqrt_inv, r_sqrt_inv)
            
            if set_eta:
                self.eta[n] = sp.sqrt(m.adot(x, x))
    
            B = sp.empty_like(self.A[n])
            for s in xrange(self.q[n]):
                B[s] = m.mmul(l_sqrt_inv, m.H(Vsh[s]), x, r_sqrt_inv)
            return B
        else:
            return None

    
    def take_step(self, dtau, save_memory=False):   
        """Performs a complete forward-Euler step of imaginary time dtau.
        
        The operation is A[n] -= dtau * B[n] with B[n] from self.calc_B(n).
        
        If dtau is itself imaginary, real-time evolution results.
        
        Parameters
        ----------
        dtau : complex
            The (imaginary or real) amount of imaginary time (tau) to step.
        save_memory : bool
            Whether to save memory by avoiding storing all B[n] at once.
        """
        eta_tot = 0
        
        if self.gauge_fixing == 'right' and save_memory:
            B = [None] * (self.N + 1)
            for n in xrange(1, self.N + self.ham_sites):
                #V is not always defined (e.g. at the right boundary vector, and possibly before)
                if n <= self.N:
                    B[n] = self.calc_B(n)
                    eta_tot += self.eta[n]
                
                #Only change an A after the next B no longer depends on it!
                if n >= self.ham_sites:
                    m = n - self.ham_sites + 1
                    if not B[m] is None:
                        self.A[m] += -dtau * B[m]
                        B[m] = None
             
            assert all(x is None for x in B), "take_step update incomplete!"
        else:
#            B = [None] #There is no site zero
#            for n in xrange(1, self.N + 1):
#                B.append(self.calc_B(n))
#                eta_tot += self.eta[n]
            B = self.calc_B_all()
            
            for n in xrange(1, self.N + 1):
                if not B[n] is None:
                    self.A[n] += -dtau * B[n]
                
        return eta_tot
        
    def take_step_RK4(self, dtau):
        """Take a step using the fourth-order explicit Runge-Kutta method.
        
        This requires more memory than a simple forward Euler step. 
        It is, however, far more accurate with a per-step error of
        order dtau**5.
        
        Parameters
        ----------
        dtau : complex
            The (imaginary or real) amount of imaginary time (tau) to step.
        """
        eta_tot = 0

        #Take a copy of the current state
        A0 = sp.empty_like(self.A)
        for n in xrange(1, self.N + 1):
            A0[n] = self.A[n].copy()

        B_fin = [None]

        B = [None]
        for n in xrange(1, self.N + 1):
            B.append(self.calc_B(n)) #k1
            eta_tot += self.eta[n]
            B_fin.append(B[-1])
        
        for n in xrange(1, self.N + 1):
            if not B[n] is None:
                self.A[n] = A0[n] - dtau/2 * B[n]
                B[n] = None

        self.update(restore_CF=False, normalize=False)
        
        B = [None]
        for n in xrange(1, self.N + 1):
            B.append(self.calc_B(n, set_eta=False)) #k2

        for n in xrange(1, self.N + 1):
            if not B[n] is None:
                self.A[n] = A0[n] - dtau/2 * B[n]
                B_fin[n] += 2 * B[n]
                B[n] = None

        self.update(restore_CF=False, normalize=False)

        B = [None]
        for n in xrange(1, self.N + 1):
            B.append(self.calc_B(n, set_eta=False)) #k3
            
        for n in xrange(1, self.N + 1):
            if not B[n] is None:
                self.A[n] = A0[n] - dtau * B[n]
                B_fin[n] += 2 * B[n]
                B[n] = None

        self.update(restore_CF=False, normalize=False)

        for n in xrange(1, self.N + 1):
            B = self.calc_B(n, set_eta=False) #k4
            if not B is None:
                B_fin[n] += B

        for n in xrange(1, self.N + 1):
            if not B_fin[n] is None:
                self.A[n] = A0[n] - dtau /6 * B_fin[n]

        return eta_tot
        