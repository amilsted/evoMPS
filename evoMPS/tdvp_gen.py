# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 17:29:27 2011

@author: Ashley Milsted

TODO:
    - Adaptive step size.

"""
import copy as cp
import scipy as sp
import scipy.linalg as la
import scipy.optimize as opti
import scipy.sparse.linalg as las
import matmul as m
import tdvp_common as tm
from mps_gen import EvoMPS_MPS_Generic
import logging

log = logging.getLogger(__name__)

class Vari_Opt_Single_Site_Op:
    def __init__(self, tdvp, n, KLnm1, tau=1, sanity_checks=False):
        """
        """
        self.D = tdvp.D
        self.q = tdvp.q
        self.tdvp = tdvp
        self.n = n
        self.KLnm1 = KLnm1
        
        self.sanity_checks = sanity_checks
        self.sanity_tol = 1E-12
        
        d = self.D[n - 1] * self.D[n] * self.q[n]
        self.shape = (d, d)
        
        self.dtype = sp.dtype(tdvp.typ)
        
        self.calls = 0
        
        self.tau = tau
        
        if n > 1:
            try:
                self.lm1_i = tdvp.l[n - 1].inv()
            except AttributeError:
                self.lm1_i = m.invmh(tdvp.l[n - 1])
        else:
            self.lm1_i = tdvp.l[n - 1]
        
    def matvec(self, x):
        self.calls += 1
        #print self.calls
        
        t = self.tdvp
        n = self.n
        
        An = x.reshape((self.q[n], self.D[n - 1], self.D[n]))
        
        if n > 1:
            AAnm1 = tm.calc_AA(t.A[n - 1], An)
            Cnm1 = tm.calc_C_mat_op_AA(t.ham[n - 1], AAnm1)
        else:
            AAnm1 = None
            Cnm1 = None
            
        if n < t.N:
            AAn = tm.calc_AA(An, t.A[n + 1])
            Cn = tm.calc_C_mat_op_AA(t.ham[n], AAn)
        else:
            AAn = None
            Cn = None
        
        res = sp.zeros_like(An)
        
        #Assuming RCF        
        if not Cnm1 is None:
            for s in xrange(t.q[n]):
                res[s] += tm.eps_l_noop(t.l[n - 2], t.A[n - 1], Cnm1[:, s])
                res[s] += self.KLnm1.dot(An[s])
            res[s] = self.lm1_i.dot(res[s])
                
        if not Cn is None:
            for s in xrange(t.q[n]):                
                res[s] += tm.eps_r_noop(t.r[n + 1], Cn[s, :], t.A[n + 1])
                res[s] += An[s].dot(t.K[n + 1])
                
        #print "en = ", (sp.inner(An.conj().ravel(), res.ravel())
        #                / sp.inner(An.conj().ravel(), An.ravel()))
        
        return res.reshape(x.shape) * self.tau
        
class Vari_Opt_SC_op:
    def __init__(self, tdvp, n, KLn, tau=1, sanity_checks=False):
        """
        """
        self.D = tdvp.D
        self.q = tdvp.q
        self.tdvp = tdvp
        self.n = n
        self.KLn = KLn
        
        self.sanity_checks = sanity_checks
        self.sanity_tol = 1E-12
        
        d = self.D[n] * self.D[n]
        self.shape = (d, d)
        
        self.dtype = sp.dtype(tdvp.typ)
        
        self.calls = 0
        
        self.tau = tau
        
    def matvec(self, x):
        self.calls += 1
        #print self.calls
        
        t = self.tdvp
        n = self.n
        
        Gn = x.reshape((self.D[n], self.D[n]))

        res = self.KLn.dot(Gn) + Gn.dot(t.K[n + 1])
        
        if n < t.N:
            Ap1 = sp.array([Gn.dot(As) for As in t.A[n + 1]])
            AAn = tm.calc_AA(t.A[n], Ap1)
            Cn = tm.calc_C_mat_op_AA(t.ham[n], AAn)
            for s in xrange(t.q[n]):
                sres = tm.eps_r_noop(t.r[n + 1], Cn[s, :], t.A[n + 1])
                res += t.A[n][s].conj().T.dot(t.l[n - 1].dot(sres))
        
        return res.reshape(x.shape) * self.tau

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
        self.AA = sp.empty((self.N), dtype=sp.ndarray)
        self.AAA = sp.empty((self.N - 1), dtype=sp.ndarray)
        
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
        
        self.eta_sq = sp.zeros((self.N + 1), dtype=self.typ)
        """The per-site contributions to the norm-squared of the TDVP tangent vector 
           (projection of the exact time evolution onto the MPS tangent plane. 
           Only available after calling take_step()."""
        self.eta_sq.fill(sp.NaN)
        
        self.etaBB_sq = sp.zeros((self.N + 1), dtype=self.typ)
        """Per-site contributions to the norm-squared of the evolution captured 
           by the two-site tangent plane but not by the one-site tangent plane. 
           Only available after calling take_step() with calc_Y_2s or dynexp."""
        self.etaBB_sq.fill(sp.NaN)
        
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
            
        for n in xrange(1, self.N):
            self.AA[n] = tm.calc_AA(self.A[n], self.A[n + 1])
        
        if self.ham_sites == 3:
            for n in xrange(1, self.N - 1):
                self.AAA[n] = tm.calc_AAA_AA(self.AA[n], self.A[n + 2])
        else:
            self.AAA.fill(None)
        
        for n in xrange(n_low, n_high + 1):
            if callable(self.ham):
                ham_n = lambda *args: self.ham(n, *args)
                ham_n = sp.vectorize(ham_n, otypes=[sp.complex128])
                ham_n = sp.fromfunction(ham_n, tuple(self.C[n].shape[:-2] * 2))
            else:
                ham_n = self.ham[n]
            
            if self.ham_sites == 2:
                self.C[n] = tm.calc_C_mat_op_AA(ham_n, self.AA[n])
            else:
                self.C[n] = tm.calc_C_3s_mat_op_AAA(ham_n, self.AAA[n])                
    
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
                                              self.r[n + 1], self.A[n], self.AA[n])
                else:
                    self.K[n], ex = tm.calc_K_3s(self.K[n + 1], self.C[n], self.l[n - 1], 
                                              self.r[n + 2], self.A[n], self.AAA[n])

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
                                            self.r[n], self.A[n], self.AA[n - 1])
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
            Am2Am1 = self.AA[n - 2]
        else:
            lm3 = None
            Cm2 = None
            Am2Am1 = None
            
        if n <= self.N - self.ham_sites + 1:
            C = self.C[n]            
        else:
            C = None
            
        if n + 1 <= self.N - self.ham_sites + 1:
            Kp1 = self.K[n + 1]            
        else:
            Kp1 = None
            
        if n < self.N - 1:
            Ap1Ap2 = self.AA[n + 1]
            rp2 = self.r[n + 2]
        else:
            Ap1Ap2 = None
            rp2 = None
            
        if n < self.N:
            rp1 = self.r[n + 1]
            Ap1 = self.A[n + 1]
        else:
            rp1 = None
            Ap1 = None
        
        if self.ham_sites == 2:
            x = tm.calc_x(Kp1, C, Cm1, rp1,
                          lm2, Am1, self.A[n], Ap1,
                          sqrt_l, sqrt_l_inv, sqrt_r, sqrt_r_inv, Vsh)
        else:
            x = tm.calc_x_3s(Kp1, C, Cm1, Cm2, rp1, rp2, lm2, 
                             lm3, Am2Am1, Am1, self.A[n], Ap1, Ap1Ap2,
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
        
    def calc_BB_Y_2s(self, l_s, l_si, r_s, r_si, Vrh, Vlh):
        Y = sp.empty((self.N + 1), dtype=sp.ndarray)
        etaBB = sp.zeros((self.N + 1), dtype=sp.complex128)
        for n in xrange(1, self.N):
            if (not Vrh[n + 1] is None and not Vlh[n] is None):
                if self.ham_sites == 2:
                    Y[n], etaBB[n] = tm.calc_BB_Y_2s(self.C[n], Vlh[n], 
                                           Vrh[n + 1], l_s[n - 1], r_s[n + 1])
                else:
                    A_m1 = self.A[n - 1] if n - 1 > 0 else None
                    A_p2 = self.A[n + 2] if n + 2 <= self.N else None
                    l_m2 = self.l[n - 2] if n - 2 >= 0 else None
                    r_p2 = self.r[n + 2] if n + 2 <= self.N else None
                    Y[n], etaBB[n] = tm.calc_BB_Y_2s_ham_3s(A_m1, A_p2, 
                       self.C[n], self.C[n - 1], Vlh[n], 
                       Vrh[n + 1], l_m2, r_p2, l_s[n - 1],
                       l_si[n - 1], r_s[n + 1], r_si[n + 1])
                
        return Y, etaBB

    def calc_BB_2s(self, Y, Vlh, Vrh, l_si, r_si, sv_tol=1E-10, dD_max=16, D_max=0):
        dD = sp.zeros((self.N + 1), dtype=int)
        BB12 = sp.empty((self.N + 1), dtype=sp.ndarray)
        BB21 = sp.empty((self.N + 1), dtype=sp.ndarray)
        
        dD_maxes = sp.repeat([dD_max], len(self.D))
        if D_max > 0:
            dD_maxes[self.D + dD_maxes > D_max] = 0
        
        for n in xrange(1, self.N):
            if not Y[n] is None and dD_maxes[n] > 0:
                BB12[n], BB21[n + 1], dD[n] = tm.calc_BB_2s(Y[n], Vlh[n], 
                                                            Vrh[n + 1], 
                                                            l_si[n - 1], r_si[n + 1],
                                                            dD_max=dD_maxes[n], sv_tol=sv_tol)
                if BB12[n] is None:
                    log.warn("calc_BB_2s: Could not calculate BB_2s at n=%u", n)

                
        return BB12, BB21, dD
    
    def calc_B(self, n, set_eta=True, l_s_m1=None, l_si_m1=None, r_s=None, r_si=None, Vlh=None, Vrh=None):
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
            return self._calc_B_r(n, set_eta=set_eta, l_s_m1=l_s_m1, l_si_m1=l_si_m1, r_s=r_s, r_si=r_si, Vrh=Vrh)
        else:
            return self._calc_B_l(n, set_eta=set_eta, l_s_m1=l_s_m1, l_si_m1=l_si_m1, r_s=r_s, r_si=r_si, Vlh=Vlh)
    
    def _calc_B_r(self, n, set_eta=True, l_s_m1=None, l_si_m1=None, r_s=None, r_si=None, Vrh=None):
        if self.q[n] * self.D[n] - self.D[n - 1] > 0:
            if l_s_m1 is None:
                l_s_m1, l_si_m1, r_s, r_si = tm.calc_l_r_roots(self.l[n - 1], self.r[n], 
                                                           zero_tol=self.zero_tol,
                                                           sanity_checks=self.sanity_checks,
                                                           sc_data=('site', n))
            
            if Vrh is None:
                Vrh = tm.calc_Vsh(self.A[n], r_s, sanity_checks=self.sanity_checks)
            
            x = self.calc_x(n, Vrh, l_s_m1, r_s, l_si_m1, r_si)
            
            if set_eta:
                self.eta_sq[n] = sp.sqrt(m.adot(x, x))
    
            B = sp.empty_like(self.A[n])
            for s in xrange(self.q[n]):
                B[s] = m.mmul(l_si_m1, x, m.H(Vrh[s]), r_si)
            return B
        else:
            return None

    def _calc_B_l(self, n, set_eta=True, l_s_m1=None, l_si_m1=None, r_s=None, r_si=None, Vlh=None):
        if self.q[n] * self.D[n - 1] - self.D[n] > 0:
            if l_s_m1 is None:
                l_s_m1, l_si_m1, r_s, r_si = tm.calc_l_r_roots(self.l[n - 1], self.r[n], 
                                                           zero_tol=self.zero_tol,
                                                           sanity_checks=self.sanity_checks,
                                                           sc_data=('site', n))
            
            if Vlh is None:
                Vlh = tm.calc_Vsh_l(self.A[n], l_s_m1, sanity_checks=self.sanity_checks)
            
            x = self.calc_x_l(n, Vlh, l_s_m1, r_s, l_si_m1, r_si)
            
            if set_eta:
                self.eta_sq[n] = sp.sqrt(m.adot(x, x))
    
            B = sp.empty_like(self.A[n])
            for s in xrange(self.q[n]):
                B[s] = m.mmul(l_si_m1, m.H(Vlh[s]), x, r_si)
            return B
        else:
            return None

    
    def take_step(self, dtau, B=None, save_memory=False, calc_Y_2s=False, 
                  dynexp=False, dD_max=16, D_max=0, sv_tol=1E-14):   
        """Performs a complete forward-Euler step of imaginary time dtau.
        
        The operation is A[n] -= dtau * B[n] with B[n] from self.calc_B(n).
        
        If dtau is itself imaginary, real-time evolution results.
        
        Second-order corrections to the dynamics can be calculated if desired.
        If they are, the norm of the second-order contributions for each bond 
        is stored in the array self.eta_BB. For nearest-neighbour Hamiltonians, 
        this captures all errors made by projecting onto the MPS tangent plane.
                
        The second-order contributions also form the basis of the dynamical 
        expansion scheme (dynexp), which captures a configurable (dD_max)
        amount of these contributions by increasing the bond dimension.
        
        Parameters
        ----------
        dtau : complex
            The (imaginary or real) amount of imaginary time (tau) to step.
        B : sequence of ndarray
            The direction to step in. Not compatible with dynexp or save_memory.
        save_memory : bool
            Whether to save memory by avoiding storing all B[n] at once.
        calc_Y_2s : bool
            Whether to calculate the second-order contributions to the dynamics.
        dynexp : bool
            Whether to increase the bond dimension to capture more of the dynamics.
        dD_max : int
            The maximum amount by which to increase the bond dimension (for any site).
        D_max : int
            The maximum bond dimension to allow when expanding.
        sv_tol : float
            Only use singular values larger than this for dynamical expansion.
        """
        eta_sq_tot = 0
        self.eta_sq.fill(0)    
        self.etaBB_sq.fill(0)
        
        if (self.gauge_fixing == 'right' and save_memory and not calc_Y_2s and not dynexp
            and B is None):
            B = [None] * (self.N + 1)
            for n in xrange(1, self.N + self.ham_sites):
                #V is not always defined (e.g. at the right boundary vector, and possibly before)
                if n <= self.N:
                    B[n] = self.calc_B(n)
                    eta_sq_tot += self.eta_sq[n]
                
                #Only change an A after the next B no longer depends on it!
                if n >= self.ham_sites:
                    m = n - self.ham_sites + 1
                    if not B[m] is None:
                        self.A[m] += -dtau * B[m]
                        B[m] = None
             
            assert all(x is None for x in B), "take_step update incomplete!"
        else:
            if B is None or dynexp or calc_Y_2s:
                l_s = sp.empty((self.N + 1), dtype=sp.ndarray)
                l_si = sp.empty((self.N + 1), dtype=sp.ndarray)
                r_s = sp.empty((self.N + 1), dtype=sp.ndarray)
                r_si = sp.empty((self.N + 1), dtype=sp.ndarray)
                Vrh = sp.empty((self.N + 1), dtype=sp.ndarray)
                Vlh = sp.empty((self.N + 1), dtype=sp.ndarray)
                for n in xrange(1, self.N + 1):
                    l_s[n-1], l_si[n-1], r_s[n], r_si[n] = tm.calc_l_r_roots(self.l[n - 1], self.r[n], 
                                                                   zero_tol=self.zero_tol,
                                                                   sanity_checks=self.sanity_checks,
                                                                   sc_data=('site', n))
                
                if dynexp or calc_Y_2s or self.gauge_fixing == 'left':
                    for n in xrange(1, self.N + 1):
                        Vlh[n] = tm.calc_Vsh_l(self.A[n], l_s[n-1], sanity_checks=self.sanity_checks)
                if dynexp or calc_Y_2s or self.gauge_fixing == 'right':
                    for n in xrange(1, self.N + 1):
                        Vrh[n] = tm.calc_Vsh(self.A[n], r_s[n], sanity_checks=self.sanity_checks)
    
                if B is None:
                    B = [None] #There is no site zero
                    for n in xrange(1, self.N + 1):
                        Bn = self.calc_B(n, set_eta=True, l_s_m1=l_s[n-1], 
                                         l_si_m1=l_si[n-1], r_s=r_s[n], r_si=r_si[n], 
                                         Vlh=Vlh[n], Vrh=Vrh[n])
                        B.append(Bn)
                        eta_sq_tot += self.eta_sq[n]
            
            if calc_Y_2s or dynexp:
                Y, self.etaBB = self.calc_BB_Y_2s(l_s, l_si, r_s, r_si, Vrh, Vlh)
                
            if dynexp:
                BB12, BB21, dD = self.calc_BB_2s(Y, Vlh, Vrh, l_si, r_si, 
                                                 dD_max=dD_max, sv_tol=sv_tol,
                                                 D_max=D_max)
                                                 
                for n in xrange(1, self.N + 1):
                    if not B[n] is None:
                        self.A[n] += -dtau * B[n]
                        
                if sp.any(dD > 0):
                    oldA = self.A
                    oldD = self.D.copy()
                    oldeta = self.eta
                    oldetaBB = self.etaBB
                    
                    self.D += dD
                    self._init_arrays()
        
                    for n in xrange(1, self.N + 1):
                        self.A[n][:, :oldD[n - 1], :oldD[n]] = oldA[n]
                        
                        if not BB12[n] is None:
                            self.A[n][:, :oldD[n - 1], oldD[n]:] = -1.j * sp.sqrt(dtau) * BB12[n]
                        if not BB21[n] is None:
                            self.A[n][:, oldD[n - 1]:, :oldD[n]] = -1.j * sp.sqrt(dtau) * BB21[n]
                        
                    log.info("Dyn. expanded! New D: %s", self.D)
                    self.eta = oldeta
                    self.etaBB = oldetaBB
            else:
                for n in xrange(1, self.N + 1):
                    if not B[n] is None:
                        self.A[n] += -dtau * B[n]
                                    
        #self.eta = sp.sqrt(eta_sq_tot)
        
        return sp.sqrt(eta_sq_tot)
        
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
        eta_sq_tot = 0
        self.eta_sq.fill(0)

        #Take a copy of the current state
        A0 = sp.empty_like(self.A)
        for n in xrange(1, self.N + 1):
            A0[n] = self.A[n].copy()

        B_fin = [None]

        B = [None]
        for n in xrange(1, self.N + 1):
            B.append(self.calc_B(n)) #k1
            eta_sq_tot += self.eta_sq[n]
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

        return sp.sqrt(eta_sq_tot)
        
    def find_min_h_brent(self, Bs, dtau_init, tol=5E-2, skipIfLower=False, 
                         verbose=False, use_tangvec_overlap=False,
                         max_iter=20):
        As0 = cp.deepcopy(self.A)
        Cs0 = cp.deepcopy(self.C)
        Ks0 = cp.deepcopy(self.K)
        h_expect_0 = self.H_expect
        
        ls0 = cp.deepcopy(self.l)
        rs0 = cp.deepcopy(self.r)
        
        taus=[0]
        if use_tangvec_overlap:
            ress = [self.eta_sq.real.sum()]
        else:
            ress = [h_expect_0.real]
        hs = [h_expect_0.real]
        
        def f(tau, *args):
            if tau < 0:
                if use_tangvec_overlap:
                    res = tau**2 + self.eta_sq.sum().real
                else:
                    res = tau**2 + h_expect_0.real
                log.debug((tau, res, "punishing negative tau!"))
                taus.append(tau)
                ress.append(res)
                hs.append(h_expect_0.real)
                return res
            try:
                i = taus.index(tau)
                log.debug((tau, ress[i], "from stored"))
                return ress[i]
            except ValueError:
                for n in xrange(1, self.N + 1):
                    if not Bs[n] is None:
                        self.A[n] = As0[n] - tau * Bs[n]
                    
                if use_tangvec_overlap:
                    self.update(restore_CF=False)
                    Bsg = [None] * (self.N + 1)
                    for n in xrange(1, self.N + 1):
                        Bsg[n] = self.calc_B(n, set_eta=False)
                    res = 0
                    for n in xrange(1, self.N + 1):
                        if not Bs[n] is None:
                            res += abs(m.adot(self.l[n - 1], tm.eps_r_noop(self.r[n], Bsg[n], Bs[n])))
                    h_exp = self.H_expect.real
                else:
                    self.calc_l()
                    self.calc_r()
                    self.simple_renorm()
                    self.calc_C()
                    
                    h_exp = 0
                    if self.ham_sites == 2:
                        for n in xrange(1, self.N):
                            h_exp += self.expect_2s(self.ham[n], n).real
                    else:
                        for n in xrange(1, self.N - 1):
                            h_exp += self.expect_3s(self.ham[n], n).real
                    res = h_exp
                
                log.debug((tau, res, h_exp, h_exp - h_expect_0.real))
                
                taus.append(tau)
                ress.append(res)
                hs.append(h_exp)
                
                return res
        
        if skipIfLower:
            if f(dtau_init) < self.H_expect.real:
                return dtau_init
        
        brack_init = (dtau_init * 0.9, dtau_init * 1.5)
        
        attempt = 1
        while attempt < 3:
            try:
                log.debug("CG: Bracketing...")
                xa, xb, xc, fa, fb, fc, funcalls = opti.bracket(f, xa=brack_init[0], 
                                                                xb=brack_init[1], 
                                                                maxiter=5)                                                
                brack = (xa, xb, xc)
                log.debug("CG: Using bracket = " + str(brack))
                break
            except RuntimeError:
                log.debug("CG: Bracketing failed, attempt %u." % attempt)
                brack_init = (brack_init[0] * 0.1, brack_init[1] * 0.1)
                attempt += 1
        
        if attempt == 3:
            log.debug("CG: Bracketing failed. Aborting!")
            tau_opt = 0
            h_min = h_expect_0.real
        else:
            try:
                tau_opt, res_min, itr, calls = opti.brent(f, 
                                                        brack=brack, 
                                                        tol=tol,
                                                        maxiter=max_iter,
                                                        full_output=True)
    
                i = taus.index(tau_opt)
                h_min = hs[i]
            except ValueError:
                log.debug("CG: Bad bracket. Aborting!")
                tau_opt = 0
                h_min = h_expect_0.real
            
        #Must restore everything needed for take_step
        self.A = As0
        self.l = ls0
        self.r = rs0
        self.C = Cs0
        self.K = Ks0
        self.H_expect = h_expect_0
        
        return tau_opt, h_min
        
        
    def calc_B_CG(self, Bs_CG_0, eta_0, dtau_init, reset=False, verbose=False,
                  switch_threshold_eta=1E-6):
        """Calculates a tangent vector using the non-linear conjugate gradient method.
        
        Parameters:
            Bs_CG_0 : ndarray
                Tangent vector used to make the previous step. Ignored on reset.
            eta_0 : float
                Norm of the previous tangent vector.
            dtau_init : float
                Initial step-size for the line-search.
            reset : bool = False
                Whether to perform a reset, using the gradient as the next search direction.
            switch_threshold_eta : float
                Sets the state tolerance (eta) below which the gradient should
                be used to determine the energetic minimum in a given direction,
                rather of the value of the energy. The gradient method is
                more expensive, but is much more robust for small .
        """
        self.eta_sq.fill(0)
        Bs = [None] * (self.N + 1)
        for n in xrange(1, self.N + 1):
            Bs[n] = self.calc_B(n)
            
        eta = self.eta_sq.real.sum()
        
        if reset:
            beta = 0.
            log.debug("CG RESET")
            
            Bs_CG = Bs
        else:
            beta = (eta**2) / eta_0**2
        
            log.debug("BetaFR = %s", beta)
        
            beta = max(0, beta.real)
            
            Bs_CG = [None] * len(Bs)
            for n in xrange(1, self.N + 1):
                if not Bs[n] is None:
                    Bs_CG[n] = Bs[n] + beta * Bs_CG_0[n]
        
        h_expect = self.H_expect.real
        
        eta_low = eta < switch_threshold_eta #Energy differences become too small here...
        
        log.debug("CG low eta: " + str(eta_low))
        
        tau, h_min = self.find_min_h_brent(Bs_CG, dtau_init,
                                           verbose=verbose, 
                                           use_tangvec_overlap=eta_low)
        
        if tau == 0:
            log.debug("CG RESET!")
            Bs_CG = Bs
        elif not eta_low and h_min > h_expect:
            log.debug("CG RESET due to energy rise!")
            Bs_CG = Bs
            tau, h_min = self.find_min_h_brent(Bs_CG, dtau_init * 0.1, 
                                               use_tangvec_overlap=False)
        
            if h_expect < h_min:
                log.debug("CG RESET FAILED: Setting tau=0!")
                tau = 0
        
        return Bs_CG, Bs, eta, tau
        
    def vari_opt_ss_sweep(self, ncv=None):
        """Perform a DMRG-style optimizing sweep to reduce the energy.
        
        This carries out the MPS version of the one-site DMRG algorithm.
        Combined with imaginary time evolution, this can dramatically improve
        convergence speed.
        """
        assert self.canonical_form == 'right', 'vari_opt_ss_sweep only implemented for right canonical form'
    
        KL = [None] * (self.N + 1)
        KL[1] = sp.zeros((self.D[1], self.D[1]), dtype=self.typ)
        for n in xrange(1, self.N + 1):
            if n > 2:
                k = n - 1
                KL[k], ex = tm.calc_K_l(KL[k - 1], self.C[k - 1], self.l[k - 2], 
                                        self.r[k], self.A[k], self.AA[k - 1])

            lop = Vari_Opt_Single_Site_Op(self, n, KL[n - 1], sanity_checks=self.sanity_checks)
            evs, eVs = las.eigsh(lop, k=1, which='SA', v0=self.A[n].ravel(), ncv=ncv)
            
            self.A[n] = eVs[:, 0].reshape((self.q[n], self.D[n - 1], self.D[n]))
            
            #shift centre matrix right (RCF is like having a centre "matrix" at "1")
            G = tm.restore_LCF_l_seq(self.A[n - 1:n + 1], self.l[n - 1:n + 1],
                                     sanity_checks=self.sanity_checks)
            
            #This is not strictly necessary, since r[n] is not used again,
            self.r[n] = G.dot(self.r[n].dot(G.conj().T))

            if n < self.N:
                for s in xrange(self.q[n + 1]):
                    self.A[n + 1][s] = G.dot(self.A[n + 1][s])
            
            #All needed l and r should now be up to date
            #All needed KR should be valid still
            #AA and C must be updated
            if n > 1:
                self.AA[n - 1] = tm.calc_AA(self.A[n - 1], self.A[n])
                self.C[n - 1] = tm.calc_C_mat_op_AA(self.ham[n - 1], self.AA[n - 1])
            if n < self.N:
                self.AA[n] = tm.calc_AA(self.A[n], self.A[n + 1])
                self.C[n] = tm.calc_C_mat_op_AA(self.ham[n], self.AA[n])

            
        for n in xrange(self.N, 0, -1):
            if n < self.N:
                self.calc_K(n_low=n + 1, n_high=n + 1)

            lop = Vari_Opt_Single_Site_Op(self, n, KL[n - 1], sanity_checks=self.sanity_checks)
            evs, eVs = las.eigsh(lop, k=1, which='SA', v0=self.A[n].ravel(), ncv=ncv)
            
            self.A[n] = eVs[:, 0].reshape((self.q[n], self.D[n - 1], self.D[n]))
            
            #shift centre matrix left (LCF is like having a centre "matrix" at "N")
            Gi = tm.restore_RCF_r_seq(self.A[n - 1:n + 1], self.r[n - 1:n + 1],
                                      sanity_checks=self.sanity_checks)
            
            #This is not strictly necessary, since l[n - 1] is not used again
            self.l[n - 1] = Gi.conj().T.dot(self.l[n - 1].dot(Gi))

            if n > 1:
                for s in xrange(self.q[n - 1]):
                    self.A[n - 1][s] = self.A[n - 1][s].dot(Gi)
            
            #All needed l and r should now be up to date
            #All needed KL should be valid still
            #AA and C must be updated
            if n > 1:
                self.AA[n - 1] = tm.calc_AA(self.A[n - 1], self.A[n])
                self.C[n - 1] = tm.calc_C_mat_op_AA(self.ham[n - 1], self.AA[n - 1])
            if n < self.N:
                self.AA[n] = tm.calc_AA(self.A[n], self.A[n + 1])
                self.C[n] = tm.calc_C_mat_op_AA(self.ham[n], self.AA[n])
                
                
    def take_step_split(self, dtau, ncv=None):
        assert self.canonical_form == 'right', 'vari_opt_ss_sweep only implemented for right canonical form'
        assert self.ham_sites == 2, 'vari_opt_ss_sweep only implemented for nearest neighbour Hamiltonians'
        dtau *= -1
        import expokit as expokit
        def expmh(A, v, t, norm_est=1., m=20):
            xn = A.shape[0]
            vf = sp.ones((xn,), dtype=A.dtype)
            m = min(xn - 1, m)
            wsp = sp.zeros((xn * (m + 2) + 5 * (m + 2)**2 + 6 + 1,), dtype=A.dtype)
            iwsp = sp.zeros((m + 2,), dtype=int)
            iflag = sp.zeros((1,), dtype=int)
            itrace = sp.array([0])
            expokit.zhexpv(m, [t], v, vf, [0.], [norm_est], 
                           wsp, iwsp, A.matvec, itrace, iflag, n=[xn], 
                           lwsp=[len(wsp)], liwsp=[len(iwsp)])
            return vf
    
        KL = [None] * (self.N + 1)
        KL[1] = sp.zeros((self.D[1], self.D[1]), dtype=self.typ)
        for n in xrange(1, self.N + 1):
            lop = Vari_Opt_Single_Site_Op(self, n, KL[n - 1])
            #print "Befor A", n, sp.inner(self.A[n].ravel().conj(), lop.matvec(self.A[n].ravel())).real
            An = expmh(lop, self.A[n].ravel(), dtau/2., norm_est=abs(self.H_expect.real))            
            self.A[n] = An.reshape((self.q[n], self.D[n - 1], self.D[n]))
            self.l[n] = tm.eps_l_noop(self.l[n - 1], self.A[n], self.A[n])
            norm = m.adot(self.l[n], self.r[n])
            self.A[n] /= sp.sqrt(norm)
            #print "After A", n, sp.inner(self.A[n].ravel().conj(), lop.matvec(self.A[n].ravel())).real, norm.real
            
            #shift centre matrix right (RCF is like having a centre "matrix" at "1")
            G = tm.restore_LCF_l_seq(self.A[n - 1:n + 1], self.l[n - 1:n + 1],
                                     sanity_checks=self.sanity_checks) 
                                     
            if n > 1:
                self.AA[n - 1] = tm.calc_AA(self.A[n - 1], self.A[n])
                self.C[n - 1] = tm.calc_C_mat_op_AA(self.ham[n - 1], self.AA[n - 1])
                KL[n], ex = tm.calc_K_l(KL[n - 1], self.C[n - 1], self.l[n - 2], 
                                        self.r[n], self.A[n], self.AA[n - 1])
                
            if n < self.N:                    
                lop2 = Vari_Opt_SC_op(self, n, KL[n])
                #print "Befor G", n, sp.inner(G.ravel().conj(), lop2.matvec(G.ravel())).real
                G = expmh(lop2, G.ravel(), -dtau/2., norm_est=abs(self.H_expect.real))
                G = G.reshape((self.D[n], self.D[n]))
                norm = sp.trace(self.l[n].dot(G).dot(self.r[n].dot(G.conj().T)))
                G /= sp.sqrt(norm)
                #print "After G", n, sp.inner(G.ravel().conj(), lop2.matvec(G.ravel())).real, norm.real
                
                for s in xrange(self.q[n + 1]):
                    self.A[n + 1][s] = G.dot(self.A[n + 1][s])                
                
                self.AA[n] = tm.calc_AA(self.A[n], self.A[n + 1])
                self.C[n] = tm.calc_C_mat_op_AA(self.ham[n], self.AA[n])           
   
        for n in xrange(self.N, 0, -1):
            lop = Vari_Opt_Single_Site_Op(self, n, KL[n - 1], tau=1, sanity_checks=self.sanity_checks)
            #print "Before A", n, sp.inner(self.A[n].ravel().conj(), lop.matvec(self.A[n].ravel())).real
            An = expmh(lop, self.A[n].ravel(), dtau/2., norm_est=abs(self.H_expect.real))
            self.A[n] = An.reshape((self.q[n], self.D[n - 1], self.D[n]))
            self.l[n] = tm.eps_l_noop(self.l[n - 1], self.A[n], self.A[n])
            norm = m.adot(self.l[n], self.r[n])
            self.A[n] /= sp.sqrt(norm)
            #print "After A", n, sp.inner(self.A[n].ravel().conj(), lop.matvec(self.A[n].ravel())).real, norm.real
            
            #shift centre matrix left (LCF is like having a centre "matrix" at "N")
            Gi = tm.restore_RCF_r_seq(self.A[n - 1:n + 1], self.r[n - 1:n + 1],
                                      sanity_checks=self.sanity_checks)
                                      
            if n < self.N:
                self.AA[n] = tm.calc_AA(self.A[n], self.A[n + 1])
                self.C[n] = tm.calc_C_mat_op_AA(self.ham[n], self.AA[n])                                          
                self.calc_K(n_low=n, n_high=n)
            
            if n > 1:
                lop2 = Vari_Opt_SC_op(self, n - 1, KL[n - 1], tau=1, sanity_checks=self.sanity_checks)
                Gi = expmh(lop2, Gi.ravel(), -dtau/2., norm_est=abs(self.H_expect.real))
                Gi = Gi.reshape((self.D[n - 1], self.D[n - 1]))
                norm = sp.trace(self.l[n - 1].dot(Gi).dot(self.r[n - 1].dot(Gi.conj().T)))
                G /= sp.sqrt(norm)
                #print "After G", n, sp.inner(Gi.ravel().conj(), lop2.matvec(Gi.ravel())).real, norm.real

                for s in xrange(self.q[n - 1]):
                    self.A[n - 1][s] = self.A[n - 1][s].dot(Gi)
            
                self.AA[n - 1] = tm.calc_AA(self.A[n - 1], self.A[n])
                self.C[n - 1] = tm.calc_C_mat_op_AA(self.ham[n - 1], self.AA[n - 1])

        
    def expect_2s(self, op, n, AA=None):
        """Computes the expectation value of a nearest-neighbour two-site operator.
        
        The operator should be a q[n] x q[n + 1] x q[n] x q[n + 1] array 
        such that op[s, t, u, v] = <st|op|uv> or a function of the form 
        op(s, t, u, v) = <st|op|uv>.
        
        The state must be up-to-date -- see self.update()!
        
        Parameters
        ----------
        op : ndarray or callable
            The operator array or function.
        n : int
            The leftmost site number (operator acts on n, n + 1).
            
        Returns
        -------
        expval : floating point number
            The expectation value (data type may be complex)
        """
        if AA is None:
            AA = self.AA[n]
        
        if op is self.ham[n] and self.ham_sites == 2:
            res = tm.eps_r_op_2s_C12_AAA45(self.r[n + 1], self.C[n], AA)
            return m.adot(self.l[n - 1], res)
        else:
            return super(EvoMPS_MPS_Generic).expect_2s(op, n, AA=AA)

    def expect_3s(self, op, n, AAA=None):
        """Computes the expectation value of a nearest-neighbour three-site operator.

        The operator should be a q[n] x q[n + 1] x q[n + 2] x q[n] x
        q[n + 1] x q[n + 2] array such that op[s, t, u, v, w, x] =
        <stu|op|vwx> or a function of the form op(s, t, u, v, w, x) =
        <stu|op|vwx>.

        The state must be up-to-date -- see self.update()!

        Parameters
        ----------
        op : ndarray or callable
            The operator array or function.
        n : int
            The leftmost site number (operator acts on n, n + 1, n + 2).

        Returns
        -------
        expval : floating point number
            The expectation value (data type may be complex)
        """
        if AAA is None:
            if not self.AAA[n] is None:
                AAA = self.AAA[n]
            else:
                AAA = tm.calc_AAA_AA(self.AA[n], self.A[n + 2])
                
        if op is self.ham[n] and self.ham_sites == 3:
            res = tm.eps_r_op_3s_C123_AAA456(self.r[n + 2], self.C[n], AAA)
            return m.adot(self.l[n - 1], res)
        else:
            return super(EvoMPS_MPS_Generic).expect_3s(op, n, AAA=AAA)
        