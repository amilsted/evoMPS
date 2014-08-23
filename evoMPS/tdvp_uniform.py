# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 17:29:27 2011

@author: Ashley Milsted

TODO:
    - Clean up CG code: Create nice interface?
    - Split out excitations stuff?

"""
import copy as cp
import numpy as np
import scipy as sp
import scipy.linalg as la
import scipy.sparse.linalg as las
import scipy.optimize as opti
import tdvp_common as tm
import matmul as m
from mps_uniform import EvoMPS_MPS_Uniform
from mps_uniform_pinv import pinv_1mE
from mps_uniform_excite import Excite_H_Op
import logging

log = logging.getLogger(__name__)
        
class EvoMPS_TDVP_Uniform(EvoMPS_MPS_Uniform):
        
    def __init__(self, D, q, ham, ham_sites=None, L=1, dtype=None):
        """Implements the TDVP algorithm for uniform MPS.
        
        Parameters
        ----------
            D : int
                The bond-dimension
            q : int
                The single-site Hilbert space dimension
            ham : callable or ndarray
                Local Hamiltonian term (acting on two or three adjacent sites)
            ham_sites : int
                The number of sites acted on non-trivially by ham. Should be specified for callable ham.
            L : int
                The number of sites in a translation invariant block.
            dtype : numpy dtype = None
                Specifies the array type.
        """

        self.ham = ham
        """The local Hamiltonian term. Can be changed, for example, to perform
           a quench. The number of neighbouring sites acted on must be 
           specified in ham_sites."""
        
        if ham_sites is None:
            try:
                self.ham_sites = len(ham.shape) / 2
            except AttributeError: #TODO: Try to count arguments using inspect module
                self.ham_sites = 2
        else:
            self.ham_sites = ham_sites
        
        if not (self.ham_sites == 2 or self.ham_sites == 3):
            raise ValueError("Only 2 or 3 site Hamiltonian terms supported!")

        self.K_solver = las.bicgstab
        
        self.h_expect = sp.NaN
        """The energy density expectation value, available only after calling
           update() or calc_K()."""
        
        super(EvoMPS_TDVP_Uniform, self).__init__(D, q, L=L, dtype=dtype)
    
    def _init_arrays(self, D, q, L):
        super(EvoMPS_TDVP_Uniform, self)._init_arrays(D, q, L)
        
        ham_shape = []
        for i in xrange(self.ham_sites):
            ham_shape.append(q)
        C_shape = tuple(ham_shape + [D, D])        
        
        self.C = []
        self.K = []
        for k in xrange(L):
            self.C.append(np.zeros(C_shape, dtype=self.typ, order=self.odr))
            self.K.append(np.ones_like(self.A[k][0]))
            
        self.AAA = [None] * L
        
        self.Vsh = [None] * L
            
        self.K_left = [None] * L
        
        self.eta_sq = sp.empty((L), dtype=self.A[0].dtype)
        """The site contributions to the norm squared of the TDVP tangent vector 
           (projection of the exact time
           evolution onto the MPS tangent plane. Only available after calling
           take_step()."""
        self.eta_sq.fill(sp.NaN)
        
        self.eta = sp.NaN
        """The norm of the TDVP tangent vector (square root of the sum of eta_sq)."""
            
    def set_ham_array_from_function(self, ham_func):
        """Generates a Hamiltonian array from a function.
        
        Given a function ham_func(s, t, u, v) this generates an array
        ham[s, t, u, v] (example for self.ham_sites == 2). 
        Using an array instead of a function can significantly
        speed up parts of the algorithm.
        
        Parameters
        ----------
        ham_func : callable
            Local Hamiltonian term with self.ham_sites * 2 required arguments.
        """
        hv = np.vectorize(ham_func, otypes=[np.complex128])
        
        if self.ham_sites == 2:
            self.ham = np.fromfunction(hv, (self.q, self.q, self.q, self.q))
        else:
            self.ham = np.fromfunction(hv, tuple([self.q] * 6))
    
    def calc_C(self):
        """Generates the C tensor used to calculate the K and ultimately B.
        
        This is called automatically by self.update().
        
        C contains a contraction of the Hamiltonian self.ham with the parameter
        tensors over the local basis indices.
        
        This is prerequisite for calculating the tangent vector parameters B,
        which optimally approximate the exact time evolution.

        Makes use only of the nearest-neighbour Hamiltonian, and of the A's.
        
        C depends on A.
        
        """
        if callable(self.ham):
            ham = np.vectorize(self.ham, otypes=[sp.complex128])
            ham = np.fromfunction(ham, tuple(self.C[0].shape[:-2] * 2))
        else:
            ham = self.ham
        
        for k in xrange(self.L):
            if self.ham_sites == 2:
                self.C[k][:] = tm.calc_C_mat_op_AA(ham, self.AA[k])
            else:
                self.AAA[k] = tm.calc_AAA_AA(self.AA[k], self.A[(k + 2) % self.L])
                self.C[k][:] = tm.calc_C_3s_mat_op_AAA(ham, self.AAA[k])
    
    def calc_PPinv(self, x, p=0, out=None, left=False, A1=None, A2=None, rL=None, 
                   pseudo=True, brute_check=False, sc_data='', solver=None):
        """Uses an iterative method to calculate the result of applying 
        the (pseudo) inverse of (1 - exp(1.j * p) * E) to a vector |x>.
        
        Parameters
        ----------
        x : ndarray
            The matrix representation of the vector |x>.
        p : float
            Momentum in units of inverse lattice spacing.
        out : ndarray
            Appropriately-sized output matrix.
        left : bool
            Whether to act left on |x> (instead of right).
        A1 : ndarray
            Ket parameter tensor.
        A2 : ndarray
            Bra parameter tensor.
        r : ndarray
            Right eigenvector of E corresponding to the largest eigenvalue.
        pseudo : bool
            Whether to calculate the pseudo inverse (or just the inverse).
        brute_check : bool
            Whether to check the answer using dense methods (scales as D**6!).
            
        Returns
        -------
        out : ndarray
            The result of applying the inverse operator, in matrix form.
        """
        if A1 is None:
            A1 = self.A
            
        if A2 is None:
            A2 = self.A
            
        if rL is None:
            rL = self.r[-1]
        
        out = pinv_1mE(x, A1, A2, self.l[-1], rL, p=p, left=left, pseudo=pseudo, 
                       out=out, tol=self.itr_rtol, solver=solver, brute_check=brute_check,
                       sanity_checks=self.sanity_checks, sc_data=sc_data)

        return out
        
    def calc_K(self):
        """Generates the K matrix used to calculate B.
        
        This also updates the energy-density expectation value self.h_expect.
        
        This is called automatically by self.update().
        
        K contains the (non-trivial) action of the Hamiltonian on the right 
        half of the infinite chain.
        
        It directly depends on A, r, and C.
        """
        L = self.L
        
        if self.ham_sites == 2:
            Hr = tm.eps_r_op_2s_C12_AA34(self.r[1 % L], self.C[0], self.AA[0])
            for k in xrange(1, L):
                Hrk = tm.eps_r_op_2s_C12_AA34(self.r[(k + 1) % L], self.C[k], self.AA[k])
                for j in xrange(k - 1, -1, -1):
                    Hrk = tm.eps_r_noop(Hrk, self.A[j], self.A[j])
                Hr += Hrk
        else:
            Hr = tm.eps_r_op_3s_C123_AAA456(self.r[2 % L], self.C[0], self.AAA[0])
            for k in xrange(1, L):
                Hrk = tm.eps_r_op_3s_C123_AAA456(self.r[(k + 2) % L], self.C[k], self.AAA[k])
                for j in xrange(k - 1, -1, -1):
                    Hrk = tm.eps_r_noop(Hrk, self.A[j], self.A[j])
                Hr += Hrk
        
        self.h_expect = m.adot(self.l[-1], Hr)
        
        QHr = Hr - self.r[-1] * self.h_expect
        
        self.h_expect /= L
        
        self.calc_PPinv(QHr, out=self.K[0], solver=self.K_solver)
        
        if self.ham_sites == 2:
            for k in sp.arange(L - 1, 0, -1) % L:
                self.K[k], hk = tm.calc_K(self.K[(k + 1) % L], self.C[k], self.l[k - 1], self.r[(k + 1) % L],
                                          self.A[k], self.AA[k])
                self.K[k] -= self.r[(k - 1) % L] * hk
        else:
            for k in sp.arange(L - 1, 0, -1) % L:
                self.K[k], hk = tm.calc_K_3s(self.K[(k + 1) % L], self.C[k], self.l[k - 1], self.r[(k + 2) % L],
                                             self.A[k], self.AAA[k])
                self.K[k] -= self.r[(k - 1) % L] * hk
        
#        if self.sanity_checks:
#            Ex = tm.eps_r_noop(self.K[k], self.A, self.A)
#            QEQ = Ex - self.r * m.adot(self.l, self.K)
#            res = self.K - QEQ
#            if not np.allclose(res, QHr):
#                log.warning("Sanity check failed: Bad K!")
#                log.warning("Off by: %s", la.norm(res - QHr))
        
    def calc_K_l(self):
        """Generates the left K matrix.
        
        See self.calc_K().
        
        K contains the (non-trivial) action of the Hamiltonian on the left 
        half of the infinite chain.
        
        It directly depends on A, l, and C.
        
        This calculates the "bra-vector" K_l ~ <K_l| (and K_l.conj().T ~ |K_l>)
        so that <K_l|r> = trace(K_l.dot(r))
        
        Returns
        -------
        K_lefts : list of ndarrays
            The left K matrices.
        h : complex
            The energy-density expectation value.
        """
        L = self.L
        if self.ham_sites == 2:
            lH = tm.eps_l_op_2s_AA12_C34(self.l[(L - 3) % L], self.AA[(L - 2) % L], self.C[(L - 2) % L])
            for k in sp.arange(-1, L - 2) % L:
                lHk = tm.eps_l_op_2s_AA12_C34(self.l[(k - 1) % L], self.AA[k], self.C[k])
                for j in xrange((k + 2) % L, L):
                    lHk = tm.eps_l_noop(lHk, self.A[j], self.A[j])
                lH += lHk
        else:
            lH = tm.eps_l_op_3s_AAA123_C456(self.l[(L - 4) % L], self.AAA[(L - 3) % L], self.C[(L - 3) % L])
            for k in sp.arange(-2, L - 3) % L:
                lHk = tm.eps_l_op_3s_AAA123_C456(self.l[(k - 1) % L], self.AAA[k], self.C[k])
                for j in xrange((k + 3) % L, L):
                    lHk = tm.eps_l_noop(lHk, self.A[j], self.A[j])
                lH += lHk
        
        h = m.adot_noconj(lH, self.r[-1]) #=tr(lH r)
        
        lHQ = lH - self.l[-1] * h
        
        h /= L
        
        #Since A1=A2 and p=0, we get the right result without turning lHQ into a ket.
        #This is the same as...
        #self.K_left = (self.calc_PPinv(lHQ.conj().T, left=True, out=self.K_left)).conj().T
        self.K_left[-1] = self.calc_PPinv(lHQ, left=True, out=self.K_left[-1], solver=self.K_solver)
                
        if self.ham_sites == 2:
            for k in sp.arange(0, L - 1):
                self.K_left[k], hk = tm.calc_K_l(self.K[(k - 1) % L], self.C[(k - 1) % L], 
                                             self.l[(k - 2) % L], self.r[k],
                                             self.A[k], self.AA[(k - 1) % L])
                self.K_left[k] -= self.l[k] * hk
        else:
            for k in sp.arange(0, L - 1):
                self.K_left[k], hk = tm.calc_K_3s_l(self.K[(k - 1) % L], self.C[(k - 1) % L], 
                                                self.l[(k - 3) % L], self.r[k],
                                                self.A[k],
                                                self.AAA[(k - 2) % L])
                self.K_left[k] -= self.l[k] * hk
        
#        if self.sanity_checks:
#            xE = tm.eps_l_noop(self.K_left, self.A, self.A)
#            QEQ = xE - self.l * m.adot(self.r, self.K_left)
#            res = self.K_left - QEQ
#            if not np.allclose(res, lHQ):
#                log.warning("Sanity check failed: Bad K_left!")
#                log.warning("Off by: %s", la.norm(res - lHQ))
        
        return self.K_left, h
        
    def get_B_from_x(self, xk, Vshk, lkm1_sqrt_i, rk_sqrt_i, out=None):
        """Calculates a gauge-fixing B-tensor for a site with block offset k 
        given parameters x.
        
        Parameters
        ----------
        xk : ndarray
            The parameter matrix.
        Vshk : ndarray
            Parametrization tensor for site offset k.
        lkm1_sqrt_i : ndarray
            The matrix self.l[k - 1] to the power of -1/2.
        rk_sqrt_i : ndarray
            The matrix self.r[k] to the power of -1/2.
        out : ndarray
            Output tensor of appropriate shape.
        """
        if out is None:
            out = np.zeros_like(self.A[0])
            
        for s in xrange(self.q):
            out[s] = lkm1_sqrt_i.dot(xk).dot(rk_sqrt_i.dot(Vshk[s]).conj().T)
            
        return out
        
    def calc_l_r_roots(self):
        """Calculates the (inverse) square roots of self.l and self.r.
        """
        self.l_sqrt = [None] * self.L
        self.l_sqrt_i = [None] * self.L
        self.r_sqrt = [None] * self.L
        self.r_sqrt_i = [None] * self.L
        for k in xrange(self.L):
            self.l_sqrt[k], self.l_sqrt_i[k], self.r_sqrt[k], self.r_sqrt_i[k] = tm.calc_l_r_roots(self.l[k], self.r[k], zero_tol=self.zero_tol, sanity_checks=self.sanity_checks)
        
    def calc_B(self, set_eta=True):
        """Calculates a gauge-fixing tangent-vector parameter tensor capturing the projected infinitesimal time evolution of the state.
        
        A TDVP time step is defined as: A -= dtau * B
        where dtau is an infinitesimal imaginary time step.        
        
        Parameters
        ----------
        set_eta : bool
            Whether to set self.eta to the norm of the tangent vector.
        """
        self.calc_l_r_roots()
        L = self.L
        
        B = []
        Vsh = []
        for k in xrange(L):
            Vshk = tm.calc_Vsh(self.A[k], self.r_sqrt[k], sanity_checks=self.sanity_checks)
            Vsh.append(Vshk)
            
            if self.ham_sites == 2:
                x = tm.calc_x(self.K[(k + 1) % L], self.C[k], self.C[(k-1)%L], 
                              self.r[(k+1)%L], self.l[(k-2)%L], self.A[(k-1)%L], 
                              self.A[k], self.A[(k+1)%L], 
                              self.l_sqrt[(k-1)%L], self.l_sqrt_i[(k-1)%L],
                              self.r_sqrt[k], self.r_sqrt_i[k], Vshk)
            else:
                x = tm.calc_x_3s(self.K[(k + 1) % L], self.C[k], self.C[(k-1)%L], 
                                 self.C[(k-2)%L], self.r[(k+1)%L], self.r[(k+2)%L], 
                                 self.l[(k-2)%L], self.l[(k-3)%L], 
                                 self.AA[(k-2)%L], self.A[(k-1)%L], self.A[k], 
                                 self.A[(k+1)%L], self.AA[(k+1)%L], 
                                 self.l_sqrt[(k-1)%L], self.l_sqrt_i[(k-1)%L],
                                 self.r_sqrt[k], self.r_sqrt_i[k], Vshk)
            
            if set_eta:
                self.eta_sq[k] = m.adot(x, x)
            
            B.append(self.get_B_from_x(x, Vshk, self.l_sqrt_i[(k-1)%L], self.r_sqrt_i[k]))
        
            if self.sanity_checks:
                #Test gauge-fixing:
                tst = tm.eps_r_noop(self.r[k], B[k], self.A[k])
                if not np.allclose(tst, 0):
                    log.warning("Sanity check failed: Gauge-fixing violation! %s", la.norm(tst))
            
        self.Vsh = Vsh
        self.eta = sp.sqrt(self.eta_sq.real.sum())
            
        return B
        
    def calc_BB_Y_2s(self, Vlh):
        L = self.L
        Y = sp.empty((L), dtype=sp.ndarray)
        etaBB = sp.zeros((L), dtype=sp.complex128)
        if self.ham_sites == 2:
            for k in xrange(L):
                Y[k], etaBB[k] = tm.calc_BB_Y_2s(self.C[k], Vlh[k], self.Vsh[(k + 1) % L],
                                                   self.l_sqrt[k - 1], self.r_sqrt[(k + 1) % L])
        else:
            for k in xrange(L):
                Y[k], etaBB[k] = tm.calc_BB_Y_2s_ham_3s(self.A[k - 1], self.A[(k + 2) % L], 
                                       self.C[k], self.C[k - 1], Vlh[k], self.Vsh[(k + 1) % L],
                                       self.l[(k - 2) % L], self.r[(k + 2) % L],
                                       self.l_sqrt[k - 1], self.l_sqrt_i[k - 1], 
                                       self.r_sqrt[(k + 1) % L], self.r_sqrt_i[(k + 1) % L])
        
        return Y, etaBB
        
    def calc_B_2s(self, dD_max=16, sv_tol=1E-14):
        Vrh = self.Vsh
        Vlh = []
        L = self.L
        for k in xrange(L):
            Vlh.append(tm.calc_Vsh_l(self.A[k], self.l_sqrt[k - 1], sanity_checks=self.sanity_checks))
        
        Y, etaBB = self.calc_BB_Y_2s(Vlh)
        
        BB1 = [None] * L
        BB2 = [None] * L
        for k in xrange(L):
            BB1[k], BB2[(k + 1) % L], dD = tm.calc_BB_2s(Y[k], Vlh[k], Vrh[(k + 1) % L], 
                                              self.l_sqrt_i[k - 1], self.r_sqrt_i[(k + 1) % L],
                                              dD_max=dD_max, sv_tol=0) #FIXME: Make D variable...
        
        return BB1, BB2, etaBB
        
    def update(self, restore_CF=True, auto_truncate=False, restore_CF_after_trunc=True):
        """Updates secondary quantities to reflect the state parameters self.A.
        
        Must be used after taking a step or otherwise changing the 
        parameters self.A before calculating
        physical quantities or taking the next step.
        
        Also (optionally) restores canonical form by calling self.restore_CF().
        
        Parameters
        ----------
        restore_CF : bool (True)
            Whether to restore canonical form.
        auto_truncate : bool (True)
            Whether to automatically truncate the bond-dimension if
            rank-deficiency is detected. Requires restore_CF.
        restore_CF_after_trunc : bool (True)
            Whether to restore_CF after truncation.
        """
        super(EvoMPS_TDVP_Uniform, self).update(restore_CF=restore_CF,
                                                auto_truncate=auto_truncate,
                                                restore_CF_after_trunc=restore_CF_after_trunc)
        self.calc_C()
        self.calc_K()
        
    def take_step(self, dtau, B=None, dynexp=False, maxD=128, dD_max=16, 
                  sv_tol=1E-14, BB=None):
        """Performs a complete forward-Euler step of imaginary time dtau.
        
        The operation is A -= dtau * B with B from self.calc_B() by default.
        
        If dtau is itself imaginary, real-time evolution results.
        
        Parameters
        ----------
        dtau : complex
            The (imaginary or real) amount of imaginary time (tau) to step.
        B : ndarray
            A custom parameter-space tangent vector to step along.
        """
        if B is None:
            B = self.calc_B()
        
        if dynexp and self.D < maxD:
            if BB is None:
                BB = self.calc_B_2s(dD_max=dD_max, sv_tol=sv_tol)
            if not BB is None:
                BB1, BB2, etaBB = BB
                oldD = self.D
                dD = BB1[0].shape[2]
                self.expand_D(self.D + dD, refac=0, imfac=0) #FIXME: Currently expands all D
                #print BB1.shape, la.norm(BB1.ravel()), BB2.shape, la.norm(BB2.ravel())
                for k in xrange(self.L):
                    self.A[k][:, :oldD, :oldD] += -dtau * B[k]
                    self.A[k][:, :oldD, oldD:] = -1.j * sp.sqrt(dtau) * BB1[k]
                    self.A[k][:, oldD:, :oldD] = -1.j * sp.sqrt(dtau) * BB2[k]
                    self.A[k][:, oldD:, oldD:].fill(0)
                log.info("Dynamically expanded! New D: %d", self.D)
            else:
                for k in xrange(self.L):
                    self.A[k] += -dtau * B[k]
        else:
            for k in xrange(self.L):
                self.A[k] += -dtau * B[k]
            
    def take_step_RK4(self, dtau, B_i=None, dynexp=False, maxD=128, dD_max=16, 
                      sv_tol=1E-14, BB=None):
        """Take a step using the fourth-order explicit Runge-Kutta method.
        
        This requires more memory than a simple forward Euler step. 
        It is, however, far more accurate with a per-step error of
        order dtau**5.
        
        Parameters
        ----------
        dtau : complex
            The (imaginary or real) amount of imaginary time (tau) to step.
        B_i : ndarray
            B calculated using self.calc_B() (if known, to avoid calculating it again).
        """
        def update():
            self.calc_lr()
            self.calc_AA()
            self.calc_C()
            self.calc_K()            

        A0 = cp.deepcopy(self.A)
            
        if not B_i is None:
            B = B_i
        else:
            B = self.calc_B() #k1
        B_fin = B
        self.take_step(dtau / 2., B=B, dynexp=dynexp, maxD=maxD, dD_max=dD_max, sv_tol=sv_tol, BB=BB)

        dD = self.A[0].shape[1] - A0[0].shape[1]
        if dD > 0:
            [A0k.resize(self.A[0].shape) for A0k in A0] #pads with zeros
            #B part doesn't work due to different scaling with dtau :(
            assert False
        
        update()
        
        B = self.calc_B(set_eta=False) #k2                
        for k in xrange(self.L):
            self.A[k] = A0[k] - dtau/2 * B[k]
            B_fin[k] += 2 * B[k]
            
        update()
            
        B = self.calc_B(set_eta=False) #k3                
        for k in xrange(self.L):
            self.A[k] = A0[k] - dtau * B[k]
            B_fin += 2 * B[k]

        update()
        
        B = self.calc_B(set_eta=False) #k4
        for k in xrange(self.L):
            B_fin[k] += B[k]
            self.A[k] = A0[k] - dtau/6 * B_fin[k]
             

    def find_ground(use_CG=True, max_steps=None, tol=1E-6):
        pass

    def evolve_state():
        pass
            
    def _prepare_excite_op_top_triv(self, p):
        if callable(self.ham):
            self.set_ham_array_from_function(self.ham)

        self.calc_K_l()
        self.calc_l_r_roots()
        self.Vsh[0] = tm.calc_Vsh(self.A[0], self.r_sqrt[0], sanity_checks=self.sanity_checks)
        
        op = Excite_H_Op(self, self, p, sanity_checks=self.sanity_checks)

        return op        
    
    def excite_top_triv(self, p, k=6, tol=0, max_itr=None, v0=None, ncv=None,
                        sigma=None,
                        which='SM', return_eigenvectors=False):
        """Calculates approximate eigenvectors and eigenvalues of the Hamiltonian
        using tangent vectors of the current state as ansatz states.
        
        This is best used with an approximate ground state to find approximate
        excitation energies.
        
        This uses topologically trivial ansatz states. Given a ground state
        degeneracy, topologically non-trivial low-lying eigenstates 
        (such as kinks or solitons) may also exist. See self.excite_top_nontriv().
        
        Many of the parameters are passed on to scipy.sparse.linalg.eigsh().
        
        Parameters
        ----------
        p : float
            Momentum in units of inverse lattice spacing.
        k : int
            Number of eigenvalues to calculate.
        tol : float
            Tolerance (defaults to machine precision).
        max_itr : int
            Maximum number of iterations.
        v0 : ndarray
            Starting vector.
        ncv : int
            Number of Arnoldi vectors to store.
        sigma : float
            Eigenvalue shift to use.
        which : string
            Which eigenvalues to find ('SM' means the k smallest).
        return_eigenvectors : bool
            Whether to return eigenvectors as well as eigenvalues.
            
        Returns
        -------
        ev : ndarray
            List of eigenvalues.
        eV : ndarray
            Matrix of eigenvectors (if return_eigenvectors == True).
        """
        op = self._prepare_excite_op_top_triv(p)
        
        res = las.eigsh(op, which=which, k=k, v0=v0, ncv=ncv,
                         return_eigenvectors=return_eigenvectors, 
                         maxiter=max_itr, tol=tol, sigma=sigma)
                          
        return res
    
    def excite_top_triv_brute(self, p, return_eigenvectors=False):
        op = self._prepare_excite_op_top_triv(p)
        
        x = np.empty(((self.q - 1)*self.D**2), dtype=self.typ)
        
        H = np.zeros((x.shape[0], x.shape[0]), dtype=self.typ)
        
        for i in xrange(x.shape[0]):
            x.fill(0)
            x[i] = 1
            H[:, i] = op.matvec(x)

        if not np.allclose(H, H.conj().T):
            log.warning("Warning! H is not Hermitian! %s", la.norm(H - H.conj().T))
         
        return la.eigh(H, eigvals_only=not return_eigenvectors)

    def _prepare_excite_op_top_nontriv(self, donor, p):
        if callable(self.ham):
            self.set_ham_array_from_function(self.ham)
        if callable(donor.ham):
            donor.set_ham_array_from_function(donor.ham)
            
#        self.calc_lr()
#        self.restore_CF()
#        donor.calc_lr()
#        donor.restore_CF()
        
        self.phase_align(donor)
        
        self.update()
        #donor.update()

        self.calc_K_l()
        self.calc_l_r_roots()
        donor.calc_l_r_roots()
        donor.Vsh[0] = tm.calc_Vsh(donor.A[0], donor.r_sqrt[0], sanity_checks=self.sanity_checks)
        
        op = Excite_H_Op(self, donor, p, sanity_checks=self.sanity_checks)

        return op 

    def excite_top_nontriv(self, donor, p, k=6, tol=0, max_itr=None, v0=None,
                           which='SM', return_eigenvectors=False, sigma=None,
                           ncv=None):
        op = self._prepare_excite_op_top_nontriv(donor, p)
                            
        res = las.eigsh(op, sigma=sigma, which=which, k=k, v0=v0,
                            return_eigenvectors=return_eigenvectors, 
                            maxiter=max_itr, tol=tol, ncv=ncv)
        
        return res
        
    def excite_top_nontriv_brute(self, donor, p, return_eigenvectors=False):
        op = self._prepare_excite_op_top_nontriv(donor, p)
        
        x = np.empty(((self.q - 1)*self.D**2), dtype=self.typ)
        
        H = np.zeros((x.shape[0], x.shape[0]), dtype=self.typ)
        
        for i in xrange(x.shape[0]):
            x.fill(0)
            x[i] = 1
            H[:, i] = op.matvec(x)

        if not np.allclose(H, H.conj().T):
            log.warning("Warning! H is not Hermitian! %s", la.norm(H - H.conj().T))
         
        return la.eigh(H, eigvals_only=not return_eigenvectors)

    def _B_to_x(self, B):
        L = self.L
        x = [None] * self.L
        
        B_ = []
        
        for k in xrange(self.L):
            x[k] = sp.zeros((self.D, self.D * (self.q - 1)), dtype=self.typ)
            for s in xrange(self.q):
                x[k] += self.l_sqrt[(k - 1) % L].dot(B[k][s]).dot(self.r_sqrt[k].dot(self.Vsh[k][s]))
                
            B_.append(self.get_B_from_x(x[k], self.Vsh[k], self.l_sqrt_i[(k-1)%L], self.r_sqrt_i[k]))
                        
        return x, B_
        
    def _B_overlap(self, B1, B2):
        """Note: Both Bs must satisfy the GFC!
        """
        res = 0
        for k in xrange(self.L):
            res += m.adot(self.l[k - 1], tm.eps_r_noop(self.r[k], B1[k], B2[k]))
            
        return res
        
    def calc_B_CG(self, B_CG_0, BpdotBp, dtau_init, dtau_prev=0, g0_prev=0, 
                  reset=False, verbose=False,
                  switch_threshold_eta=1E-5, B_prev=None, use_PR=False):
        """Calculates a tangent vector using the non-linear conjugate gradient method.
        
        Parameters:
            B_CG_0 : ndarray
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
                more expensive, but is much more robust for small eta.
        """
        B = self.calc_B()
        BdotB = (self.eta.real**2).sum()
        eta = sp.sqrt(BdotB)
    
        if reset:
            beta = 0.
            log.debug("CG RESET")
            
            B_CG = B
        else:
            if use_PR and not B_prev is None:
                x_prev, B_prev_ = self._B_to_x(B_prev)
                BdotBp = self._B_overlap(B, B_prev_)
                print "BdotBp", BdotBp
                beta = (BdotB - BdotBp) / BpdotBp
            else:
                beta = BdotB / BpdotBp #FR
        
            log.debug("Beta = %s", beta)

            print "CG: beta =", beta
            
            beta = max(0, beta.real)
            
            B_CG = [None] * self.L 
            for k in xrange(self.L):
                B_CG[k] = B[k] + beta * B_CG_0[k]
                
            x_, B_CG = self._B_to_x(B_CG)

        
        lLb0 = self.lL_before_CF.copy()
        rLb0 = self.rL_before_CF.copy()
        
        h_expect = self.h_expect.real.copy()
        
        eta_low = eta < switch_threshold_eta #Energy differences become too small here...
        
        log.debug("CG low eta: " + str(eta_low))
        
        ls = EvoMPS_line_search(self, B_CG, B, use_tangvec_overlap=eta_low)
        g0 = ls.gs[0].real
        if g0 > 0:
            print "CG: Bad search direction! Resetting!", g0, g0_prev
            B_CG = B
            ls = EvoMPS_line_search(self, B_CG, B, use_tangvec_overlap=eta_low)
            g0 = ls.gs[0].real
#        elif g0_prev != 0 and g0 < 0:
#            dtau_prev *= abs(g0_prev / g0)
#            print "tau adjustment:", g0_prev / g0, dtau_prev
        
        if eta_low:
            tau, h_min = ls.brentq(max(dtau_prev, dtau_init * 0.01))
        else:
            tau, h_min = ls.brent(max(dtau_prev, dtau_init * 0.01))
        #tau, h_min = self.find_min_h_brent(B_CG, B, max(dtau_prev, dtau_init * 0.01),
        #                                   verbose=verbose, 
        #                                   use_tangvec_overlap=eta_low)

        if tau == 0:
            log.warning("CG RESET due to failed line search!")
            reset_retry = True
        elif not eta_low and h_min > h_expect:
            log.warning("CG RESET due to energy rise!")
            reset_retry = True
        else:
            reset_retry = False

        if reset_retry:
            B_CG = B
            self.lL_before_CF = lLb0
            self.rL_before_CF = rLb0
            #tau, h_min = self.find_min_h_brent(B_CG, B, dtau_init, 
            #                                   use_tangvec_overlap=eta_low)
            if eta_low:
                tau, h_min = ls.brentq(dtau_init)
            else:
                tau, h_min = ls.brent(dtau_init)
        
            if not eta_low and h_expect < h_min:
                log.error("CG RESET FAILED: Energy rose. Setting tau=0!")
                self.lL_before_CF = lLb0
                self.rL_before_CF = rLb0
                tau = 0
        
        return B_CG, B, BdotB, tau, g0
        
            
    def export_state(self, userdata=None):
        if userdata is None:
            userdata = self.userdata

        lL = np.asarray(self.l[-1])
        rL = np.asarray(self.r[-1])
            
        tosave = np.empty((5), dtype=np.ndarray)
        if self.L == 1:
            tosave[0] = self.A[0]
        else:
            tosave[0] = sp.array(self.A)
        tosave[1] = lL
        tosave[2] = rL
        tosave[3] = self.K[0]
        tosave[4] = np.asarray(userdata)
        
        return tosave
            
    def save_state(self, file, userdata=None):
        np.save(file, self.export_state(userdata))
        
    def import_state(self, state, expand=False, truncate=False,
                     expand_q=False, shrink_q=False, refac=0.1, imfac=0.1):
        newA = state[0]
        newlL = state[1]
        newrL = state[2]
        newK0 = state[3]
        if state.shape[0] > 4:
            self.userdata = state[4]
            
        if len(newA.shape) == 3:
            newA = [newA]
        elif len(newA.shape) == 4:
            newA = list(newA)
        
        if (len(newA) == self.L and newA[0].shape == self.A[0].shape):
            self.A = newA
            self.K[0] = newK0
            self.l[-1] = np.asarray(newlL)
            self.r[-1] = np.asarray(newrL)
            self.lL_before_CF = self.l[-1]
            self.rL_before_CF = self.r[-1]
                
            return True
        elif expand and len(newA) == self.L and (
        len(newA[0].shape) == 3) and (newA[0].shape[0] == 
        self.A[0].shape[0]) and (newA[0].shape[1] == newA[0].shape[2]) and (
        newA[0].shape[1] <= self.A[0].shape[1]):
            newD = self.D
            savedD = newA[0].shape[1]
            self._init_arrays(savedD, self.q, self.L)
            self.A = newA
            self.K[0] = newK0
            self.l[-1] = np.asarray(newlL)
            self.r[-1] = np.asarray(newrL)
            self.expand_D(newD, refac, imfac)
            self.lL_before_CF = self.l[-1]
            self.rL_before_CF = self.r[-1]
            log.warning("EXPANDED!")
        elif truncate and len(newA) == self.L and (len(newA[0].shape) == 3) \
                and (newA[0].shape[0] == self.A[0].shape[0]) \
                and (newA[0].shape[1] == newA[0].shape[2]) \
                and (newA[0].shape[1] >= self.A[0].shape[1]):
            newD = self.D
            savedD = newA[0].shape[1]
            self._init_arrays(savedD, self.q, self.L)
            self.A = newA
            self.K[0] = newK0
            self.l[-1] = np.asarray(newlL)
            self.r[-1] = np.asarray(newrL)
            self.update()  # to make absolutely sure we're in CF
            self.truncate(newD, update=True)
            log.warning("TRUNCATED!")
        elif expand_q and len(newA) == self.L and (len(newA[0].shape) == 3) and (
        newA[0].shape[0] <= self.A[0].shape[0]) and (newA[0].shape[1] == 
        newA[0].shape[2]) and (newA[0].shape[1] == self.A[0].shape[1]):
            newQ = self.q
            savedQ = newA[0].shape[0]
            self._init_arrays(self.D, savedQ, self.L)
            self.A = newA
            self.K[0] = newK0
            self.l[-1] = np.asarray(newlL)
            self.r[-1] = np.asarray(newrL)
            self.expand_q(newQ)
            self.lL_before_CF = self.l[-1]
            self.rL_before_CF = self.r[-1]
            log.warning("EXPANDED in q!")
        elif shrink_q and len(newA) == self.L and (len(newA[0].shape) == 3) and (
        newA[0].shape[0] >= self.A[0].shape[0]) and (newA[0].shape[1] == 
        newA[0].shape[2]) and (newA[0].shape[1] == self.A[0].shape[1]):
            newQ = self.q
            savedQ = newA[0].shape[0]
            self._init_arrays(self.D, savedQ)
            self.A = newA
            self.K[0] = newK0
            self.l[-1] = np.asarray(newlL)
            self.r[-1] = np.asarray(newrL)
            self.shrink_q(newQ)
            self.lL_before_CF = self.l[-1]
            self.rL_before_CF = self.r[-1]
            log.warning("SHRUNK in q!")
        else:
            return False
            
    def load_state(self, file, expand=False, truncate=False, expand_q=False,
                   shrink_q=False, refac=0.1, imfac=0.1):
        state = np.load(file)
        return self.import_state(state, expand=expand, truncate=truncate,
                                 expand_q=expand_q, shrink_q=shrink_q,
                                 refac=refac, imfac=imfac)
            
    def set_q(self, newq):
        oldK = self.K        
        super(EvoMPS_TDVP_Uniform, self).set_q(newq)        
        self.K = oldK
                    
    def expand_D(self, newD, refac=100, imfac=0):
        oldK0 = self.K[0]
        oldD = self.D

        super(EvoMPS_TDVP_Uniform, self).expand_D(newD, refac=refac, imfac=imfac)
        #self._init_arrays(newD, self.q)
        
        self.K[0][:oldD, :oldD] = oldK0
        #self.K[oldD:, :oldD].fill(0 * 1E-3 * la.norm(oldK) / oldD**2)
        #self.K[:oldD, oldD:].fill(0 * 1E-3 * la.norm(oldK) / oldD**2)
        #self.K[oldD:, oldD:].fill(0 * 1E-3 * la.norm(oldK) / oldD**2)
        val = abs(oldK0.mean())
        m.randomize_cmplx(self.K[0].ravel()[oldD**2:], a=0, b=val, aj=0, bj=0)
        
    def expect_2s(self, op, k=0):
        if op is self.ham and self.ham_sites == 2:
            res = tm.eps_r_op_2s_C12_AA34(self.r[(k + 1) % self.L], self.C[k], self.AA[k])
            return m.adot(self.l[(k - 1) % self.L], res)
        else:
            return super(EvoMPS_TDVP_Uniform, self).expect_2s(op, k=k)
            
    def expect_3s(self, op, k=0):
        if op is self.ham and self.ham_sites == 3:
            res = tm.eps_r_op_3s_C123_AAA456(self.r[(k + 2) % self.L], self.C[k], self.AAA[k])
            return m.adot(self.l[(k - 1) % self.L], res)
        else:
            return super(EvoMPS_TDVP_Uniform, self).expect_3s(op, k=k)


class line_search_wolfe(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)

class EvoMPS_line_search():
    def __init__(self, tdvp, B, Bg, use_tangvec_overlap=False, calc_gradient=True):
        self.B = B
        self.Bg0 = Bg
        self.use_tangvec_overlap = use_tangvec_overlap
        self.calc_gradient = calc_gradient
        self.penalise_neg = True
        self.in_search = False
        
        self.tdvp0 = tdvp
        self.tdvp = cp.deepcopy(tdvp)
        
        #Bring all l and r into dense form
        for k in xrange(tdvp.L):
            try:
                tdvp.l[k] = tdvp.l[k].A
            except AttributeError:
                pass
            
            try:
                tdvp.r[k] = tdvp.r[k].A
            except AttributeError:
                pass        
        
        self.taus = [0]            
        self.hs = [tdvp.h_expect.real]
        self.gs = [-2 * tdvp._B_overlap(B, Bg)]
        self.lLs = [tdvp.lL_before_CF.copy()]
        self.rLs = [tdvp.rL_before_CF.copy()]
        self.K0s = [tdvp.K[0]]
        self.Ws = [False]
        
        if use_tangvec_overlap:
            self.ress = [abs(self.gs[0].real)]
        else:
            self.ress = [self.hs[0]]
        
    def f(self, tau, *args):
        if self.penalise_neg and tau < 0:
            res = tau**2 + self.ress[0]
            log.debug((tau, res, "punishing negative tau!"))
            self.taus.append(tau)
            self.ress.append(res)
            self.hs.append(self.hs[0])
            self.gs.append(self.gs[0])
            self.lLs.append(self.lLs[0])
            self.rLs.append(self.rLs[0])
            self.K0s.append(self.K0s[0])
            self.Ws.append(False)
            return res
        try:
            i = self.taus.index(tau)
            log.debug((tau, self.ress[i], "from stored"))
            return self.ress[i]
        except ValueError:
            for k in xrange(self.tdvp.L):
                self.tdvp.A[k] = self.tdvp0.A[k] - tau * self.B[k]
            
            nearest_tau_ind = abs(sp.array(self.taus) - tau).argmin()
            self.lL_before_CF = self.lLs[nearest_tau_ind] #needn't copy these
            self.rL_before_CF = self.rLs[nearest_tau_ind]

            if self.use_tangvec_overlap or self.calc_gradient:
                self.tdvp.K[0] = self.K0s[nearest_tau_ind].copy()
                
                self.tdvp.update(restore_CF=False)
                Bg = self.tdvp.calc_B(set_eta=False)
                x_, B_ = self.tdvp._B_to_x(self.B)
                g = -2 * self.tdvp._B_overlap(Bg, B_)
                h_exp = self.tdvp.h_expect.real
                wol = self.wolfe(g, h_exp, tau)
                K0 = self.tdvp.K[0].copy()
            else:
                self.tdvp.calc_lr()
                self.tdvp.calc_AA()
                self.tdvp.calc_C()
                h_exp = 0
                if self.tdvp.ham_sites == 2:
                    for k in xrange(self.tdvp.L):
                        h_exp += self.tdvp.expect_2s(self.tdvp.ham, k).real
                else:
                    for k in xrange(self.tdvp.L):
                        h_exp += self.tdvp.expect_3s(self.tdvp.ham, k).real
                h_exp /= self.tdvp.L
                wol = False
                g = sp.NaN
                K0 = None
            
            if self.use_tangvec_overlap:
                res = g.real**2
            else:
                res = h_exp
            
            log.debug((tau, res, h_exp, h_exp - self.hs[0], wol,
                       self.tdvp.itr_l, self.tdvp.itr_r))
            print tau, g.real, wol
            
            self.taus.append(tau)
            self.hs.append(h_exp)
            self.gs.append(g)
            self.lLs.append(self.tdvp.l[-1].copy())
            self.rLs.append(self.tdvp.r[-1].copy())
            self.Ws.append(wol)
            self.ress.append(res)
            self.K0s.append(K0)

            if wol and self.in_search:
                raise line_search_wolfe(tau)
            
            return res
            
    def g(self, tau, *args):
        try:
            i = self.taus.index(tau)
            g = self.gs[i]
        except ValueError:
            self.f(tau)
            g = self.gs[-1]
            
        assert not sp.isnan(g), 'Gradient not available.' #FIXME: Force it in this case!
            
        return g.real
            
    def wolfe(self, g, h, tau, strong=False):
        #I wonder how badly this breaks if the energy is innaccurate
        w1 = h <= self.hs[0] + 1E-4 * tau * self.gs[0].real
        if strong:
            w2 = abs(g.real) <= 0.1 * abs(self.gs[0].real)
        else:
            w2 = g.real >= 0.1 * self.gs[0].real
        
        return w1 and w2
        
    def linesearch(self, dtau_init):
        tau_opt, fc, gc, phi_star, old_fval, derphi_star = opti.line_search(self.f, self.g, dtau_init, 1., c1=1E-4, c2=0.1)
        
        if tau_opt is None:
            log.error("CG: Lost precision!")
            tau_opt = 0
            h_min = self.hs[0]
        else:
            try:
                i = self.taus.index(tau_opt)
            except ValueError:
                print tau_opt
                self.f(tau_opt)
                i = -1
            h_min = self.hs[i]
            self.tdvp0.lL_before_CF = self.lLs[i]
            self.tdvp0.rL_before_CF = self.rLs[i]
        
        return tau_opt, h_min
        
    def linesearch2(self, dtau_init, tol=1E-3):
        #err = opti.check_grad(self.f, self.g, sp.array([dtau_init]))
        #print "error on grad", err
        tau_opt, fc, gc, res_min, old_tau, g = opti.linesearch.line_search_wolfe1(self.f, self.g, dtau_init, 1., c1=1E-4, c2=0.1, xtol=tol)
        
        if tau_opt is None:
            log.error("CG: Lost precision!")
            tau_opt = 0
            h_min = self.hs[0]
        else:
            try:
                i = self.taus.index(tau_opt)
            except ValueError:
                print tau_opt
                self.f(tau_opt)
                i = -1
            h_min = self.hs[i]
            self.tdvp0.lL_before_CF = self.lLs[i]
            self.tdvp0.rL_before_CF = self.rLs[i]
        
        return tau_opt, h_min
        
    def try_bracket(self, brack_init):
        attempt = 1
        while attempt < 3:
            try:
                log.debug("CG: Bracketing...")
                xa, xb, xc, fa, fb, fc, funcalls = opti.bracket(self.f, xa=brack_init[0], 
                                                                xb=brack_init[1], 
                                                                maxiter=5)
                brack = (xa, xb, xc)
                log.debug("CG: Using bracket = " + str(brack))
                break
            except RuntimeError:
                log.warning("CG: Bracketing failed, attempt %u." % attempt)
                brack_init = (brack_init[0] * 0.1, brack_init[1] * 0.1)
                attempt += 1
        
        if attempt < 3:
            return brack
        else:
            return None
        
    def brent(self, dtau_init, tol=5E-2, max_iter=20):
                  
        g = self.g(dtau_init)
        i = self.taus.index(dtau_init)
                  
        if self.wolfe(g, self.hs[i], dtau_init):
            log.debug("CG: Using initial step, since Wolfe satisfied!")
            tau_opt = dtau_init
            h_min = self.hs[i]
        else:
            brack_init = (dtau_init * 0.1, dtau_init)
            self.in_search = True
            try:
                brack = self.try_bracket(brack_init)
            except line_search_wolfe:
                log.debug("CG: Aborting bracket due to Wolfe")
                brack = brack_init #ignored

            if brack is None:
                log.error("CG: Bracketing failed. Aborting!")
                tau_opt = 0
                h_min = self.hs[0]
            else:
                if True in self.Ws:
                    print "CG: selecting from bracket"
                    taus_ = sp.array(self.taus)
                    Ws_ = sp.array(self.Ws)
                    hs_ = sp.array(self.hs)
                    i = taus_[Ws_].argmax()
                    tau_opt = taus_[Ws_][i]
                    h_min = hs_[Ws_][i]
                else:
                    try:
                        self.in_search = True
                        tau_opt, res_min, itr, calls = opti.brent(self.f, 
                                                                brack=brack, 
                                                                tol=tol,
                                                                maxiter=max_iter,
                                                                full_output=True)

                        #hopefully optimize next calc_lr
                        nearest_tau_ind = abs(np.array(self.taus) - tau_opt).argmin()
                        self.tdvp0.lL_before_CF = self.lLs[nearest_tau_ind]
                        self.tdvp0.rL_before_CF = self.rLs[nearest_tau_ind]

                        i = self.taus.index(tau_opt)
                        h_min = self.hs[i]
                    except line_search_wolfe as e:
                        print "CG: Aborting early due to Wolfe", e.value
                        tau_opt = e.value
                        i = self.taus.index(tau_opt)
                        h_min = self.hs[i]
                        self.tdvp0.lL_before_CF = self.lLs[i]
                        self.tdvp0.rL_before_CF = self.rLs[i]
                    except ValueError:
                        log.error("CG: Bad bracket. Aborting!")
                        tau_opt = 0
                        h_min = self.hs[0]
                        
        self.in_search = False
        return tau_opt, h_min
        
    def brentq(self, dtau_init, tol=1E-3, max_iter=20):
        g = self.g(dtau_init)
        i = self.taus.index(dtau_init)
                  
        if self.wolfe(g, self.hs[i], dtau_init):
            log.debug("CG: Using initial step, since Wolfe satisfied!")
            tau_opt = dtau_init
            h_min = self.hs[i]
        else:
            self.in_search = True
            
            try:
                taus = None
                if g > 0: #We already have g0 < 0 - enough to bracket
                    taus = [0, dtau_init]
                #else:
                #    taus = self.try_bracket([0, dtau_init])
                
                if taus is None:
                    for i in xrange(3):
                        taus = self.bracket_interp(dtau_init / (i + 1.))
                    
                        if not taus is None:
                            break
                    
                if taus is None:
                    return 0, self.hs[0] #fail
                
                try:
                    tau_opt = opti.brentq(self.g, taus[0], taus[-1], xtol=tol, maxiter=max_iter)
                except ValueError:
                    log.warning("CG: Failed to find a valid bracket.")
                    return 0, self.hs[0] #fail
                
                nearest_tau_ind = abs(np.array(self.taus) - tau_opt).argmin()
                self.tdvp0.lL_before_CF = self.lLs[nearest_tau_ind]
                self.tdvp0.rL_before_CF = self.rLs[nearest_tau_ind]

                i = self.taus.index(tau_opt)
                h_min = self.hs[i]
            except line_search_wolfe as e:
                print "CG: Aborting early due to Wolfe", e.value
                tau_opt = e.value
                i = self.taus.index(tau_opt)
                h_min = self.hs[i]
                self.tdvp0.lL_before_CF = self.lLs[i]
                self.tdvp0.rL_before_CF = self.rLs[i]
        
        self.in_search = False
        return tau_opt, h_min
        
    def bracket_interp(self, dtau_init, fac_red=0.9, fac_inc=1.1, max_itr=10):
        g0 = self.g(dtau_init)

        tau1 = dtau_init
        g1 = g0

        i = 0
        while g1 * g0 > 0 and i < max_itr:
            po = sp.polyfit(self.taus, self.gs, max(3, len(self.gs) - 1))
            por = sp.roots(po)
            tau1 = por.max().real
        
            if g1 < 0:
                tau1 *= fac_inc
            else:
                tau1 *= fac_red
        
            g1 = self.g(tau1)
            
            i += 1
            
        if i == max_itr:
            return None
        else:
            return sorted([dtau_init, tau1])

#
#        g2 = self.g(tau2)
#        
#        if g * g2 < 0:
#            taus = [tau2, dtau_init]
#        else:
#            po = sp.polyfit([tau2, dtau_init], [g2, g], 1)
#            tau3 = sp.roots(po)[0]
#            
#            if g > 0:
#                tau3 *= 0.5
#            else:
#                tau3 *= 1.5
#            
#            g3 = self.g(tau3)
#            print "taus:", [tau2, dtau_init, tau3], "grads:", g2, g, g3, "poly", sp.polyval(po, [dtau_init * 0.5, dtau_init, tau3])
#            taus = [tau3, dtau_init]
#            
#            if g * g3 > 0:
#                po = sp.polyfit([tau2, dtau_init, tau3], [g2, g, g3], 2)
#                por = sp.roots(po)
#                tau4 = por.max().real
#                if g > 0:
#                    tau4 *= 0.5
#                else:
#                    tau4 *= 1.5
#                g4 = self.g(tau4)
#                taus = [tau4, dtau_init]
#                print "taus:", [tau2, dtau_init, tau3, tau4], "grads:", g2, g, g3, g4, "poly", sp.polyval(po, [dtau_init * 0.5, dtau_init, tau3, tau4])
#        taus.sort()
#        
#        return taus
        
    def bracket_step(self, dtau_init, fac_red=0.1, fac_inc=2.5, max_itr=10):
        g0 = self.g(dtau_init)
        tau1 = dtau_init
        g1 = g0
        i = 0
        while g1 * g0 > 0 and i < max_itr:
            if g1 < 0:
                tau1 *= fac_inc
            else:
                tau1 *= fac_red
            
            g1 = self.g(tau1)
            
            i += 1
            
        if i == max_itr:
            return None
        else:
            return sorted([dtau_init, tau1])