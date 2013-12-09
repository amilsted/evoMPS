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
from mps_uniform_multi import EvoMPS_MPS_Uniform
from mps_uniform_pinv_multi import pinv_1mE
import logging

log = logging.getLogger(__name__)
        
class EvoMPS_TDVP_Uniform(EvoMPS_MPS_Uniform):
        
    def __init__(self, D, q, L, ham, ham_sites=None, dtype=None):
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
        
        super(EvoMPS_TDVP_Uniform, self).__init__(D, q, L, dtype=dtype)
                        
        self.eta = sp.NaN
        """The norm of the TDVP tangent vector (projection of the exact time
           evolution onto the MPS tangent plane. Only available after calling
           take_step()."""

        self.etas = sp.empty((L), dtype=dtype)
        self.etas.fill(sp.NaN)
        
        self.h_expect = sp.empty((L), dtype=dtype)
        """The energy density expectation value, available only after calling
           update() or calc_K()."""
        self.h_expect.fill(sp.NaN)
    
    def _init_arrays(self, D, q, L):
        super(EvoMPS_TDVP_Uniform, self)._init_arrays(D, q, L)
        
        ham_shape = []
        for i in xrange(self.ham_sites):
            ham_shape.append(q)
        C_shape = tuple(ham_shape + [D, D])        
        
        self.Cs = []
        self.Ks = []
        for k in xrange(L):
            self.Cs.append(np.zeros(C_shape, dtype=self.typ, order=self.odr))
            self.Ks.append(np.ones_like(self.As[k][0]))
            
        self.AAAs = [None] * L
            
        self.K_left = None
            
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
            ham = np.fromfunction(ham, tuple(self.C.shape[:-2] * 2))
        else:
            ham = self.ham
        
        for k in xrange(self.L):
            if self.ham_sites == 2:
                self.Cs[k][:] = tm.calc_C_mat_op_AA(ham, self.AAs[k])
            else:
                self.AAAs[k] = tm.calc_AAA(self.As[k], self.As[(k + 1) % self.L], 
                                           self.As[(k + 2) % self.L])
                self.Cs[k][:] = tm.calc_C_3s_mat_op_AAA(ham, self.AAAs[k])
    
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
            A1s = self.As
            
        if A2 is None:
            A2s = self.As
            
        if rL is None:
            rL = self.rs[-1]
        
        out = pinv_1mE(x, A1s, A2s, self.ls[-1], rL, p=p, left=left, pseudo=pseudo, 
                       out=out, tol=self.itr_rtol, solver=solver,
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
            Hr = tm.eps_r_op_2s_C12_AA34(self.rs[1 % L], self.Cs[0], self.AAs[0])
            for k in xrange(1, L):
                Hrk = tm.eps_r_op_2s_C12_AA34(self.rs[(k + 1) % L], self.Cs[k], self.AAs[k])
                for j in xrange(k - 1, -1, -1):
                    Hrk = tm.eps_r_noop(Hrk, self.As[j], self.As[j])
                Hr += Hrk
        else:
            Hr = tm.eps_r_op_3s_C123_AAA456(self.rs[2 % L], self.Cs[0], self.AAAs[0])
            for k in xrange(1, L):
                Hrk = tm.eps_r_op_3s_C123_AAA456(self.rs[(k + 2) % L], self.Cs[k], self.AAAs[k])
                for j in xrange(k - 1, -1, -1):
                    Hrk = tm.eps_r_noop(Hrk, self.As[j], self.As[j])
                Hr += Hrk
        
        self.h_expect = m.adot(self.ls[-1], Hr)
        
        QHr = Hr - self.rs[-1] * self.h_expect
        
        self.h_expect /= L
        
        self.calc_PPinv(QHr, out=self.Ks[0], solver=self.K_solver)
        
        if self.ham_sites == 2:
            for k in sp.arange(L - 1, 0, -1) % L:
                self.Ks[k], hk = tm.calc_K(self.Ks[(k + 1) % L], self.Cs[k], self.ls[k - 1], self.rs[(k + 1) % L],
                                           self.As[k], self.As[(k + 1) % L], sanity_checks=self.sanity_checks)
                self.Ks[k] -= self.rs[(k - 1) % L] * hk
        else:
            for k in sp.arange(L - 1, 0, -1) % L:
                self.Ks[k], hk = tm.calc_K_3s(self.Ks[(k + 1) % L], self.Cs[k], self.ls[k - 1], self.rs[(k + 2) % L],
                                          self.As[k], self.As[(k + 1) % L],
                                          self.As[(k + 2) % L], sanity_checks=self.sanity_checks)
                self.Ks[k] -= self.rs[(k - 1) % L] * hk
        
#        if self.sanity_checks:
#            Ex = tm.eps_r_noop(self.Ks[k], self.A, self.A)
#            QEQ = Ex - self.r * m.adot(self.l, self.K)
#            res = self.K - QEQ
#            if not np.allclose(res, QHr):
#                log.warning("Sanity check failed: Bad K!")
#                log.warning("Off by: %s", la.norm(res - QHr))
        
    def get_B_from_x(self, x, Vsh, l_sqrt_i, r_sqrt_i, out=None):
        """Calculates a gauge-fixing B-tensor given parameters x.
        
        Parameters
        ----------
        x : ndarray
            The parameter matrix.
        Vsh : ndarray
            Parametrization tensor.
        l_sqrt_i : ndarray
            The matrix self.l to the power of -1/2.
        r_sqrt_i : ndarray
            The matrix self.r to the power of -1/2.
        out : ndarray
            Output tensor of appropriate shape.
        """
        if out is None:
            out = np.zeros_like(self.As[0])
            
        for s in xrange(self.q):
            out[s] = l_sqrt_i.dot(x).dot(r_sqrt_i.dot(Vsh[s]).conj().T)
            
        return out
        
    def calc_l_r_roots(self):
        """Calculates the (inverse) square roots of self.l and self.r.
        """
        self.ls_sqrt = [None] * self.L
        self.ls_sqrt_i = [None] * self.L
        self.rs_sqrt = [None] * self.L
        self.rs_sqrt_i = [None] * self.L
        for k in xrange(self.L):
            self.ls_sqrt[k], self.ls_sqrt_i[k], self.rs_sqrt[k], self.rs_sqrt_i[k] = tm.calc_l_r_roots(self.ls[k], self.rs[k], zero_tol=self.zero_tol, sanity_checks=self.sanity_checks)
        
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
        
        Bs = []
        Vshs = []
        for k in xrange(L):
            Vsh = tm.calc_Vsh(self.As[k], self.rs_sqrt[k], sanity_checks=self.sanity_checks)
            Vshs.append(Vsh)
            
            if self.ham_sites == 2:
                x = tm.calc_x(self.Ks[(k + 1) % L], self.Cs[k], self.Cs[(k-1)%L], 
                              self.rs[(k+1)%L], self.ls[(k-2)%L], self.As[(k-1)%L], 
                              self.As[k], self.As[(k+1)%L], 
                              self.ls_sqrt[(k-1)%L], self.ls_sqrt_i[(k-1)%L],
                              self.rs_sqrt[k], self.rs_sqrt_i[k], Vsh)
            else:
                x = tm.calc_x_3s(self.Ks[(k + 1) % L], self.Cs[k], self.Cs[(k-1)%L], 
                                 self.Cs[(k-2)%L], self.rs[(k+1)%L], self.rs[(k+2)%L], 
                                 self.ls[(k-2)%L], self.ls[(k-3)%L], 
                                 self.As[(k-2)%L], self.As[(k-1)%L], self.As[k], 
                                 self.As[(k+1)%L], self.As[(k+2)%L], 
                                 self.ls_sqrt[(k-1)%L], self.ls_sqrt_i[(k-1)%L],
                                 self.rs_sqrt[k], self.rs_sqrt_i[k], Vsh)
            
            if set_eta:
                self.etas[k] = sp.sqrt(m.adot(x, x))
            
            Bs.append(self.get_B_from_x(x, Vsh, self.ls_sqrt_i[(k-1)%L], self.rs_sqrt_i[k]))
        
            if self.sanity_checks:
                #Test gauge-fixing:
                tst = tm.eps_r_noop(self.rs[k], Bs[k], self.As[k])
                if not np.allclose(tst, 0):
                    log.warning("Sanity check failed: Gauge-fixing violation! %s" ,la.norm(tst))
        
        if set_eta:
            self.eta = self.etas.sum()
            
        self.Vshs = Vshs
            
        return Bs
        
    def calc_BB_Y_2s(self, Vlhs):
        L = self.L
        Ys = sp.empty((L), dtype=sp.ndarray)
        etaBBs = sp.zeros((L), dtype=sp.complex128)
        if self.ham_sites == 2:
            for k in xrange(L):
                Ys[k], etaBBs[k] = tm.calc_BB_Y_2s(self.Cs[k], Vlhs[k], self.Vshs[(k + 1) % L],
                                                   self.ls_sqrt[k - 1], self.rs_sqrt[(k + 1) % L])
        else:
            for k in xrange(L):
                Ys[k], etaBBs[k] = tm.calc_BB_Y_2s_ham_3s(self.As[k - 1], self.As[(k + 2) % L], 
                                       self.Cs[k], self.Cs[k - 1], Vlhs[k], self.Vshs[(k + 1) % L],
                                       self.ls[(k - 2) % L], self.rs[(k + 2) % L],
                                       self.ls_sqrt[k - 1], self.ls_sqrt_i[k - 1], 
                                       self.rs_sqrt[(k + 1) % L], self.rs_sqrt_i[(k + 1) % L])
        
        return Ys, etaBBs
        
    def calc_B_2s(self, dD_max=16, sv_tol=1E-14):
        Vrhs = self.Vshs
        Vlhs = []
        L = self.L
        for k in xrange(L):
            Vlhs.append(tm.calc_Vsh_l(self.As[k], self.ls_sqrt[k - 1], sanity_checks=self.sanity_checks))
        
        Ys, etaBBs = self.calc_BB_Y_2s(Vlhs)
        
        BB1s = [None] * L
        BB2s = [None] * L
        for k in xrange(L):
            BB1s[k], BB2s[(k + 1) % L], dD = tm.calc_BB_2s(Ys[k], Vlhs[k], Vrhs[(k + 1) % L], 
                                              self.ls_sqrt_i[k - 1], self.rs_sqrt_i[(k + 1) % L],
                                              dD_max=dD_max, sv_tol=0) #FIXME: Make D variable...
        
        return BB1s, BB2s, etaBBs
        
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
        
    def take_step(self, dtau, Bs=None, dynexp=False, maxD=128, dD_max=16, 
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
        if Bs is None:
            Bs = self.calc_B()
        
        if dynexp and self.D < maxD:
            if BB is None:
                BB = self.calc_B_2s(dD_max=dD_max, sv_tol=sv_tol)
            if not BB is None:
                BB1s, BB2s, etaBBs = BB
                oldD = self.D
                dD = BB1s[0].shape[2]
                self.expand_D(self.D + dD, refac=0, imfac=0) #FIXME: Currently expands all D
                #print BB1.shape, la.norm(BB1.ravel()), BB2.shape, la.norm(BB2.ravel())
                for k in xrange(self.L):
                    self.As[k][:, :oldD, :oldD] += -dtau * Bs[k]
                    self.As[k][:, :oldD, oldD:] = -1.j * sp.sqrt(dtau) * BB1s[k]
                    self.As[k][:, oldD:, :oldD] = -1.j * sp.sqrt(dtau) * BB2s[k]
                    self.As[k][:, oldD:, oldD:].fill(0)
                log.info("Dynamically expanded! New D: %d", self.D)
            else:
                for k in xrange(self.L):
                    self.As[k] += -dtau * Bs[k]
        else:
            for k in xrange(self.L):
                self.As[k] += -dtau * Bs[k]
            
    def take_step_RK4(self, dtau, B_i=None):
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
        
    def calc_BHB_prereq(self, donor):
        """Calculates prerequisites for the application of the effective Hamiltonian in terms of tangent vectors.
        
        This is called (indirectly) by the self.excite.. functions.
        
        Parameters
        ----------
        donor: EvoMPS_TDVP_Uniform
            Second state (may be the same, or another ground state).
            
        Returns
        -------
        A lot of stuff.
        """
        l = self.l
        r_ = donor.r
        r__sqrt = donor.r_sqrt
        r__sqrt_i = donor.r_sqrt_i
        A = self.A
        A_ = donor.A
        AA_ = donor.AA
        
        eyed = np.eye(self.q**self.ham_sites)
        eyed = eyed.reshape(tuple([self.q] * self.ham_sites * 2))
        ham_ = self.ham - self.h_expect.real * eyed
            
        V_ = sp.transpose(donor.Vsh, axes=(0, 2, 1)).conj()
        
        Vri_ = sp.zeros_like(V_)
        try:
            for s in xrange(donor.q):
                Vri_[s] = r__sqrt_i.dot_left(V_[s])
        except AttributeError:
            for s in xrange(donor.q):
                Vri_[s] = V_[s].dot(r__sqrt_i)

        Vr_ = sp.zeros_like(V_)            
        try:
            for s in xrange(donor.q):
                Vr_[s] = r__sqrt.dot_left(V_[s])
        except AttributeError:
            for s in xrange(donor.q):
                Vr_[s] = V_[s].dot(r__sqrt)
                
        _C_AhlA = np.empty_like(self.C)
        for u in xrange(self.q):
            for s in xrange(self.q):
                _C_AhlA[u, s] = A[u].conj().T.dot(l.dot(A[s]))
        C_AhlA = sp.tensordot(ham_, _C_AhlA, ((0, 2), (0, 1)))
        
        _C_A_Vrh_ = tm.calc_AA(A_, sp.transpose(Vr_, axes=(0, 2, 1)).conj())
        C_A_Vrh_ = sp.tensordot(ham_, _C_A_Vrh_, ((3, 1), (0, 1)))
                
        C_Vri_A_conj = tm.calc_C_conj_mat_op_AA(ham_, tm.calc_AA(Vri_, A_))

        C_ = tm.calc_C_mat_op_AA(ham_, AA_)
        C_conj = tm.calc_C_conj_mat_op_AA(ham_, AA_)
        
        rhs10 = tm.eps_r_op_2s_AA12_C34(r_, AA_, C_Vri_A_conj)
        
        return ham_, C_, C_conj, V_, Vr_, Vri_, C_Vri_A_conj, C_AhlA, C_A_Vrh_, rhs10
            
    def calc_BHB(self, x, p, donor, ham_, C_, C_conj, V_, Vr_, Vri_, 
                 C_Vri_A_conj, C_AhlA, C_A_Vrh_, rhs10, M_prev=None, y_pi_prev=None, pinv_solver=None): 
        """Calculates the result of applying the effective Hamiltonian in terms
        of tangent vectors to a particular tangent vector specified by x.
        
        Note: For a good approx. ground state, H should be Hermitian pos. semi-def.
        
        Parameters
        ----------
        x : ndarray
            The tangent vector parameters according to the gauge-fixing parametrization.
        p : float
            Momentum in units of inverse lattice spacing.
        donor: EvoMPS_TDVP_Uniform
            Second state (may be the same, or another ground state).
        ...others...
            Prerequisites returned by self.calc_BHB_prereq().
        """
        if pinv_solver is None:
            pinv_solver = las.gmres
        
        A = self.A
        A_ = donor.A
        
        l = self.l
        r_ = donor.r
        
        l_sqrt = self.l_sqrt
        l_sqrt_i = self.l_sqrt_i
        
        r__sqrt = donor.r_sqrt
        r__sqrt_i = donor.r_sqrt_i
        
        K__r = donor.K
        K_l = self.K_left #this is the 'bra' vector already
        
        pseudo = donor is self
        
        B = donor.get_B_from_x(x, donor.Vsh, l_sqrt_i, r__sqrt_i)
        
        #Skip zeros due to rank-deficiency
        if la.norm(B) == 0:
            return sp.zeros_like(x), M_prev, y_pi_prev
        
        if self.sanity_checks:
            tst = tm.eps_r_noop(r_, B, A_)
            if not np.allclose(tst, 0):
                log.warning("Sanity check failed: Gauge-fixing violation! %s", la.norm(tst))

        if self.sanity_checks:
            B2 = np.zeros_like(B)
            for s in xrange(self.q):
                B2[s] = l_sqrt_i.dot(x.dot(Vri_[s]))
            if not sp.allclose(B, B2, rtol=self.itr_rtol*self.check_fac,
                               atol=self.itr_atol*self.check_fac):
                log.warning("Sanity Fail in calc_BHB! Bad Vri!")
            
        BA_ = tm.calc_AA(B, A_)
        AB = tm.calc_AA(self.A, B)
            
        y = tm.eps_l_noop(l, B, self.A)
        
        M = self.calc_PPinv(y, p=-p, left=True, A1=A_, pseudo=pseudo, sc_data='M', 
                            out=M_prev, solver=pinv_solver)
        if self.sanity_checks:
            y2 = M - sp.exp(+1.j * p) * tm.eps_l_noop(M, A_, self.A) #(1 - exp(pj) EA_A |M>)
            if not sp.allclose(y, y2, rtol=1E-10, atol=1E-12):
                norm = la.norm(y.ravel())
                if norm == 0:
                    norm = 1
                log.warning("Sanity Fail in calc_BHB! Bad M. Off by: %g", (la.norm((y - y2).ravel()) / norm))
        if pseudo:
            M = M - l * m.adot(r_, M)
        Mh = m.H(M)

        res = l_sqrt.dot(
               tm.eps_r_op_2s_AA12_C34(r_, BA_, C_Vri_A_conj) #1 OK
               + sp.exp(+1.j * p) * tm.eps_r_op_2s_AA12_C34(r_, AB, C_Vri_A_conj) #3 OK with 4
              )
        #res.fill(0)
        
        res += sp.exp(-1.j * p) * l_sqrt_i.dot(Mh.dot(rhs10)) #10
        
        exp = sp.exp
        subres = sp.zeros_like(res)
        eye = m.eyemat(C_.shape[2], dtype=self.typ)
        for s in xrange(self.q):
            #subres += C_AhlA[s, t].dot(B[s]).dot(Vr_[t].conj().T) #2 OK
            subres += tm.eps_r_noop(B[s], C_AhlA[:, s], Vr_)
            #+ exp(-1.j * p) * A[t].conj().T.dot(l.dot(B[s])).dot(C_A_Vrh_[s, t]) #4 OK with 3
            subres += exp(-1.j * p) * tm.eps_l_noop(l.dot(B[s]), A, C_A_Vrh_[:, s])
            #+ exp(-2.j * p) * A[s].conj().T.dot(Mh.dot(C_[s, t])).dot(Vr_[t].conj().T)) #12
            subres += exp(-2.j * p) * A[s].conj().T.dot(Mh).dot(tm.eps_r_noop(eye, C_[s], Vr_))
                
        res += l_sqrt_i.dot(subres)
        
        res += l_sqrt.dot(tm.eps_r_noop(K__r, B, Vri_)) #5 OK
        
        res += l_sqrt_i.dot(K_l.dot(tm.eps_r_noop(r__sqrt, B, V_))) #6
        
        res += sp.exp(-1.j * p) * l_sqrt_i.dot(Mh.dot(tm.eps_r_noop(K__r, A_, Vri_))) #8
        
        y1 = sp.exp(+1.j * p) * tm.eps_r_noop(K__r, B, A_) #7
        y2 = sp.exp(+1.j * p) * tm.eps_r_op_2s_AA12_C34(r_, BA_, C_conj) #9
        y3 = sp.exp(+2.j * p) * tm.eps_r_op_2s_AA12_C34(r_, AB, C_conj) #11
        
        y = y1 + y2 + y3
        if pseudo:
            y = y - m.adot(l, y) * r_
        y_pi = self.calc_PPinv(y, p=p, A2=A_, pseudo=pseudo, sc_data='y_pi', 
                               out=y_pi_prev, solver=pinv_solver)
        #print m.adot(l, y_pi)
        if self.sanity_checks:
            y2 = y_pi - sp.exp(+1.j * p) * tm.eps_r_noop(y_pi, self.A, A_)
            if not sp.allclose(y, y2, rtol=1E-10, atol=1E-12):
                log.warning("Sanity Fail in calc_BHB! Bad y_pi. Off by: %g", la.norm((y - y2).ravel()) / la.norm(y.ravel()))
        if pseudo:
            y_pi = y_pi - m.adot(l, y_pi) * r_
        
        res += l_sqrt.dot(tm.eps_r_noop(y_pi, self.A, Vri_))
        
        if self.sanity_checks:
            expval = m.adot(x, res) / m.adot(x, x)
            #print "expval = " + str(expval)
            if expval < 0:
                log.warning("Sanity Fail in calc_BHB! H is not pos. semi-definite (%s)", expval)
            if not abs(expval.imag) < 1E-9:
                log.warning("Sanity Fail in calc_BHB! H is not Hermitian (%s)", expval)
        
        return res, M, y_pi
    
    def calc_BHB_prereq_3s(self, donor):
        """As for self.calc_BHB_prereq(), but for Hamiltonian terms acting on three sites.
        """
        l = self.l
        r_ = donor.r
        r__sqrt = donor.r_sqrt
        r__sqrt_i = donor.r_sqrt_i
        A = self.A
        AA = self.AA
        A_ = donor.A
        AA_ = donor.AA
        AAA_ = donor.AAA
        
        eyed = np.eye(self.q**self.ham_sites)
        eyed = eyed.reshape(tuple([self.q] * self.ham_sites * 2))
        ham_ = self.ham - self.h_expect.real * eyed
        
        V_ = sp.zeros((donor.Vsh.shape[0], donor.Vsh.shape[2], donor.Vsh.shape[1]), dtype=self.typ)
        for s in xrange(donor.q):
            V_[s] = m.H(donor.Vsh[s])
        
        Vri_ = sp.zeros_like(V_)
        try:
            for s in xrange(donor.q):
                Vri_[s] = r__sqrt_i.dot_left(V_[s])
        except AttributeError:
            for s in xrange(donor.q):
                Vri_[s] = V_[s].dot(r__sqrt_i)

        Vr_ = sp.zeros_like(V_)            
        try:
            for s in xrange(donor.q):
                Vr_[s] = r__sqrt.dot_left(V_[s])
        except AttributeError:
            for s in xrange(donor.q):
                Vr_[s] = V_[s].dot(r__sqrt)
            
        C_Vri_AA_ = np.empty((self.q, self.q, self.q, Vri_.shape[1], A_.shape[2]), dtype=self.typ)
        for s in xrange(self.q):
            for t in xrange(self.q):
                for u in xrange(self.q):
                    C_Vri_AA_[s, t, u] = Vri_[s].dot(AA_[t, u])
        C_Vri_AA_ = sp.tensordot(ham_, C_Vri_AA_, ((3, 4, 5), (0, 1, 2)))
        
        C_AAA_r_Ah_Vrih = np.empty((self.q, self.q, self.q, self.q, self.q, #FIXME: could be too memory-intensive
                                    A_.shape[1], Vri_.shape[1]), 
                                   dtype=self.typ)
        for s in xrange(self.q):
            for t in xrange(self.q):
                for u in xrange(self.q):
                    for k in xrange(self.q):
                        for j in xrange(self.q):
                            C_AAA_r_Ah_Vrih[s, t, u, k, j] = AAA_[s, t, u].dot(r_.dot(A_[k].conj().T)).dot(Vri_[j].conj().T)
        C_AAA_r_Ah_Vrih = sp.tensordot(ham_, C_AAA_r_Ah_Vrih, ((3, 4, 5, 2, 1), (0, 1, 2, 3, 4)))
        
        C_AhAhlAA = np.empty((self.q, self.q, self.q, self.q,
                              A_.shape[2], A.shape[2]), dtype=self.typ)
        for t in xrange(self.q):
            for j in xrange(self.q):
                for i in xrange(self.q):
                    for s in xrange(self.q):
                        C_AhAhlAA[t, j, i, s] = AA[i, j].conj().T.dot(l.dot(AA[s, t]))
        C_AhAhlAA = sp.tensordot(ham_, C_AhAhlAA, ((4, 1, 0, 3), (0, 1, 2, 3)))
        
        C_AA_r_Ah_Vrih_ = np.empty((self.q, self.q, self.q, self.q,
                                    A_.shape[1], Vri_.shape[1]), dtype=self.typ)
        for t in xrange(self.q):
            for u in xrange(self.q):
                for k in xrange(self.q):
                    for j in xrange(self.q):
                        C_AA_r_Ah_Vrih_[t, u, k, j] = AA_[t, u].dot(r_.dot(A_[k].conj().T)).dot(Vri_[j].conj().T)
        C_AA_r_Ah_Vrih_ = sp.tensordot(ham_, C_AA_r_Ah_Vrih_, ((4, 5, 2, 1), (0, 1, 2, 3)))
        
        C_AAA_Vrh_ = np.empty((self.q, self.q, self.q, self.q,
                               A_.shape[1], Vri_.shape[1]), dtype=self.typ)
        for s in xrange(self.q):
            for t in xrange(self.q):
                for u in xrange(self.q):
                    for k in xrange(self.q):
                        C_AAA_Vrh_[s, t, u, k] = AAA_[s, t, u].dot(Vr_[k].conj().T)
        C_AAA_Vrh_ = sp.tensordot(ham_, C_AAA_Vrh_, ((3, 4, 5, 2), (0, 1, 2, 3)))
        
        C_A_r_Ah_Vrih = np.empty((self.q, self.q, self.q,
                                  A_.shape[2], Vri_.shape[1]), dtype=self.typ)
        for u in xrange(self.q):
            for k in xrange(self.q):
                for j in xrange(self.q):
                    C_A_r_Ah_Vrih[u, k, j] = A_[u].dot(r_.dot(A_[k].conj().T)).dot(Vri_[j].conj().T)
        C_A_r_Ah_Vrih = sp.tensordot(ham_, C_A_r_Ah_Vrih, ((5, 2, 1), (0, 1, 2)))
        
        C_AhlAA = np.empty((self.q, self.q, self.q,
                                  A_.shape[2], A.shape[2]), dtype=self.typ)
        for s in xrange(self.q):
            for t in xrange(self.q):
                for i in xrange(self.q):
                    C_AhlAA[s, t, i] = A[i].conj().T.dot(l.dot(AA[s, t]))
        C_AhlAA = sp.tensordot(ham_, C_AhlAA, ((3, 4, 0), (0, 1, 2)))
        
        C_AhAhlA = np.empty((self.q, self.q, self.q,
                                  A_.shape[2], A.shape[2]), dtype=self.typ)
        for j in xrange(self.q):
            for i in xrange(self.q):
                for s in xrange(self.q):
                    C_AhAhlA[j, i, s] = AA[i, j].conj().T.dot(l.dot(A[s]))
        C_AhAhlA = sp.tensordot(ham_, C_AhAhlA, ((1, 0, 3), (0, 1, 2)))
        
        C_AA_Vrh = np.empty((self.q, self.q, self.q,
                                  A_.shape[2], Vr_.shape[1]), dtype=self.typ)
        for t in xrange(self.q):
            for u in xrange(self.q):
                for k in xrange(self.q):
                    C_AA_Vrh[t, u, k] = AA_[t, u].dot(Vr_[k].conj().T)
        C_AA_Vrh = sp.tensordot(ham_, C_AA_Vrh, ((4, 5, 2), (0, 1, 2)))
        
        C_ = sp.tensordot(ham_, AAA_, ((3, 4, 5), (0, 1, 2)))
        
        rhs10 = tm.eps_r_op_3s_C123_AAA456(r_, AAA_, C_Vri_AA_)
        
        #NOTE: These C's are good as C12 or C34, but only because h is Hermitian!
        #TODO: Make this consistent with the updated 2-site case above.
        
        return V_, Vr_, Vri_, C_, C_Vri_AA_, C_AAA_r_Ah_Vrih, C_AhAhlAA, C_AA_r_Ah_Vrih_, C_AAA_Vrh_, C_A_r_Ah_Vrih, C_AhlAA, C_AhAhlA, C_AA_Vrh, rhs10,
    
    def calc_BHB_3s(self, x, p, donor, V_, Vr_, Vri_, C_, C_Vri_AA_, C_AAA_r_Ah_Vrih,
                    C_AhAhlAA, C_AA_r_Ah_Vrih_, C_AAA_Vrh_, C_A_r_Ah_Vrih, 
                    C_AhlAA, C_AhAhlA, C_AA_Vrh, rhs10,
                    M_prev=None, y_pi_prev=None, pinv_solver=None):
        """As for self.calc_BHB(), but for Hamiltonian terms acting on three sites.
        """
        if pinv_solver is None:
            pinv_solver = las.gmres        
        
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
            tst = tm.eps_r_noop(r_, B, A_)
            if not np.allclose(tst, 0):
                log.warning("Sanity check failed: Gauge-fixing violation!")

        if self.sanity_checks:
            B2 = np.zeros_like(B)
            for s in xrange(self.q):
                B2[s] = l_sqrt_i.dot(x.dot(Vri_[s]))
            if not sp.allclose(B, B2, rtol=self.itr_rtol*self.check_fac,
                               atol=self.itr_atol*self.check_fac):
                log.warning("Sanity Fail in calc_BHB! Bad Vri!")
        
        BAA_ = tm.calc_AAA(B, A_, A_)
        ABA_ = tm.calc_AAA(A, B, A_)
        AAB = tm.calc_AAA(A, A, B)
        
        y = tm.eps_l_noop(l, B, self.A)
        
        if pseudo:
            y = y - m.adot(r_, y) * l #should just = y due to gauge-fixing
        M = self.calc_PPinv(y, p=-p, left=True, A1=A_, r=r_, pseudo=pseudo, out=M_prev, solver=pinv_solver)
        #print m.adot(r, M)
        if self.sanity_checks:
            y2 = M - sp.exp(+1.j * p) * tm.eps_l_noop(M, A_, self.A)
            if not sp.allclose(y, y2):
                log.warning("Sanity Fail in calc_BHB! Bad M. Off by: %g", (la.norm((y - y2).ravel()) / la.norm(y.ravel())))
        Mh = m.H(M)
        
        res = l_sqrt.dot(
               tm.eps_r_op_3s_C123_AAA456(r_, BAA_, C_Vri_AA_) #1 1D
               + sp.exp(+1.j * p) * tm.eps_r_op_3s_C123_AAA456(r_, ABA_, C_Vri_AA_) #3
               + sp.exp(+2.j * p) * tm.eps_r_op_3s_C123_AAA456(r_, AAB, C_Vri_AA_) #3c
              )
        #res.fill(0)
        
        res += sp.exp(-1.j * p) * l_sqrt_i.dot(Mh.dot(rhs10)) #10
        
        exp = sp.exp
        subres = sp.zeros_like(res)
        for s in xrange(self.q):
            subres += exp(-2.j * p) * A[s].conj().T.dot(Mh.dot(C_AAA_r_Ah_Vrih[s])) #12
            
            for t in xrange(self.q):
                subres += (C_AhAhlAA[t, s].dot(B[s]).dot(Vr_[t].conj().T)) #2b
                subres += (exp(-1.j * p) * A[s].conj().T.dot(l.dot(B[t])).dot(C_AA_r_Ah_Vrih_[s, t])) #4
                subres += (exp(-3.j * p) * A[s].conj().T.dot(A[t].conj().T).dot(Mh).dot(C_AAA_Vrh_[t, s])) #12b
                
                for u in xrange(self.q):
                    subres += (A[s].conj().T.dot(l.dot(A[t]).dot(B[u])).dot(C_A_r_Ah_Vrih[s, t, u])) #2 -ive of that it should be....
                    subres += (exp(+1.j * p) * C_AhlAA[u, t, s].dot(B[s]).dot(r_.dot(A_[t].conj().T)).dot(Vri_[u].conj().T)) #3b
                    subres += (exp(-1.j * p) * C_AhAhlA[s, t, u].dot(B[t]).dot(A_[u]).dot(Vr_[s].conj().T)) #4b
                    subres += (exp(-2.j * p) * A[s].conj().T.dot(A[t].conj().T).dot(l.dot(B[u])).dot(C_AA_Vrh[t, s, u])) #4c
                    
        res += l_sqrt_i.dot(subres)
        
        res += l_sqrt.dot(tm.eps_r_noop(K__r, B, Vri_)) #5
        
        res += l_sqrt_i.dot(K_l.dot(tm.eps_r_noop(r__sqrt, B, V_))) #6
        
        res += sp.exp(-1.j * p) * l_sqrt_i.dot(Mh.dot(tm.eps_r_noop(K__r, A_, Vri_))) #8
        
        y1 = sp.exp(+1.j * p) * tm.eps_r_noop(K__r, B, A_) #7
        y2 = sp.exp(+1.j * p) * tm.eps_r_op_3s_C123_AAA456(r_, BAA_, C_) #9
        y3 = sp.exp(+2.j * p) * tm.eps_r_op_3s_C123_AAA456(r_, ABA_, C_) #11
        y4 = sp.exp(+3.j * p) * tm.eps_r_op_3s_C123_AAA456(r_, AAB, C_) #11b
        
        y = y1 + y2 + y3 + y4
        if pseudo:
            y = y - m.adot(l, y) * r_
        y_pi = self.calc_PPinv(y, p=p, A2=A_, r=r_, pseudo=pseudo, out=y_pi_prev, solver=pinv_solver)
        #print m.adot(l, y_pi)
        if self.sanity_checks:
            y2 = y_pi - sp.exp(+1.j * p) * tm.eps_r_noop(y_pi, self.A, A_)
            if not sp.allclose(y, y2):
                log.warning("Sanity Fail in calc_BHB! Bad x_pi. Off by: %g", (la.norm((y - y2).ravel()) / la.norm(y.ravel())))
        
        res += l_sqrt.dot(tm.eps_r_noop(y_pi, self.A, Vri_))
        
        if self.sanity_checks:
            expval = m.adot(x, res) / m.adot(x, x)
            #print "expval = " + str(expval)
            if expval < 0:
                log.warning("Sanity Fail in calc_BHB! H is not pos. semi-definite (%s)", expval)
            if not abs(expval.imag) < 1E-9:
                log.warning("Sanity Fail in calc_BHB! H is not Hermitian (%s)", expval)
        
        return res, M, y_pi        
    
    def _prepare_excite_op_top_triv(self, p):
        if callable(self.ham):
            self.set_ham_array_from_function(self.ham)

        self.calc_K_l()
        self.calc_l_r_roots()
        self.Vsh = tm.calc_Vsh(self.A, self.r_sqrt, sanity_checks=self.sanity_checks)
        
        op = Excite_H_Op(self, self, p)

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
        donor.Vsh = tm.calc_Vsh(donor.A, donor.r_sqrt, sanity_checks=self.sanity_checks)
        
        op = Excite_H_Op(self, donor, p)

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

        
    def find_min_h_brent(self, Bs, dtau_init, tol=5E-2, skipIfLower=False, 
                         verbose=False, use_tangvec_overlap=False,
                         max_iter=20):
        As0 = cp.deepcopy(self.As)
        AAs0 = cp.deepcopy(self.AAs)
        try:
            AAAs0 = cp.deepcopy(self.AAAs)
        except:
            AAAs0 = None
        Cs0 = cp.deepcopy(self.Cs)
        Ks0 = cp.deepcopy(self.Ks)
        h_expect_0 = self.h_expect.copy()
        
        ls0 = cp.deepcopy(self.ls)
        rs0 = cp.deepcopy(self.rs)
        for k in xrange(self.L):
            try:
                self.ls[k] = self.ls[k].A
            except AttributeError:
                pass
            
            try:
                self.rs[k] = self.rs[k].A
            except AttributeError:
                pass        
        
        taus=[0]
        if use_tangvec_overlap:
            ress=[self.etas.real.sum()]
        else:
            ress=[h_expect_0.real]
        hs=[h_expect_0.real]
        lLs = [self.lL_before_CF.copy()]
        rLs = [self.rL_before_CF.copy()]
        K0s = [Ks0[0]]
        
        def f(tau, *args):
            if tau < 0:
                if use_tangvec_overlap:
                    res = tau**2 + self.eta.real
                else:
                    res = tau**2 + h_expect_0.real
                log.debug((tau, res, "punishing negative tau!"))
                taus.append(tau)
                ress.append(res)
                hs.append(h_expect_0.real)
                lLs.append(ls0[-1])
                rLs.append(rs0[-1])
                K0s.append(Ks0[0])
                return res
            try:
                i = taus.index(tau)
                log.debug((tau, ress[i], "from stored"))
                return ress[i]
            except ValueError:
                for k in xrange(self.L):
                    self.As[k] = As0[k] - tau * Bs[k]
                
                if len(taus) > 0:
                    nearest_tau_ind = abs(np.array(taus) - tau).argmin()
                    self.lL_before_CF = lLs[nearest_tau_ind] #needn't copy these
                    self.rL_before_CF = rLs[nearest_tau_ind]
                    #self.l_before_CF = l0
                    #self.r_before_CF = r0
                    if use_tangvec_overlap:
                        self.Ks[0] = K0s[nearest_tau_ind].copy()

                if use_tangvec_overlap:
                    self.update(restore_CF=False)
                    Bsg = self.calc_B(set_eta=False)
                    res = 0
                    for k in xrange(self.L):
                        res += abs(m.adot(self.ls[k - 1], tm.eps_r_noop(self.rs[k], Bsg[k], Bs[k])))
                    h_exp = self.h_expect.real
                else:
                    self.calc_lr()
                    if self.ham_sites == 2:
                        self.calc_AA()
                    self.calc_C()
                    
                    h_exp = 0
                    if self.ham_sites == 2:
                        for k in xrange(self.L):
                            h_exp += self.expect_2s(self.ham, k).real
                    else:
                        for k in xrange(self.L):
                            h_exp += self.expect_3s(self.ham, k).real
                    h_exp /= self.L
                    res = h_exp
                
                log.debug((tau, res, h_exp, h_exp - h_expect_0.real, self.itr_l, self.itr_r))
                
                taus.append(tau)
                ress.append(res)
                hs.append(h_exp)
                lLs.append(self.ls[-1].copy())
                rLs.append(self.rs[-1].copy())
                if use_tangvec_overlap:
                    K0s.append(self.Ks[0].copy())
                else:
                    K0s.append(None)
                
                return res
        
        if skipIfLower:
            if f(dtau_init) < self.h_expect.real:
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
            self.lL_before_CF = ls0[-1]
            self.rL_before_CF = rs0[-1]
        else:
            try:
                tau_opt, res_min, itr, calls = opti.brent(f, 
                                                        brack=brack, 
                                                        tol=tol,
                                                        maxiter=max_iter,
                                                        full_output=True)
    
                #hopefully optimize next calc_lr
                nearest_tau_ind = abs(np.array(taus) - tau_opt).argmin()
                self.lL_before_CF = lLs[nearest_tau_ind]
                self.rL_before_CF = rLs[nearest_tau_ind]
                
                i = taus.index(tau_opt)
                h_min = hs[i]
            except ValueError:
                log.debug("CG: Bad bracket. Aborting!")
                tau_opt = 0
                h_min = h_expect_0.real
                self.lL_before_CF = ls0[-1]
                self.rL_before_CF = rs0[-1]
            
        #Must restore everything needed for take_step
        self.As = As0
        self.ls = ls0
        self.rs = rs0
        self.AAs = AAs0
        self.AAAs = AAAs0
        self.Cs = Cs0
        self.Ks = Ks0
        self.h_expect = h_expect_0
        
        return tau_opt, h_min
        
        
    def calc_B_CG(self, Bs_CG_0, eta_0, dtau_init, reset=False, verbose=False,
                  switch_threshold_eta=1E-6):
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
                more expensive, but is much more robust for small .
        """
        Bs = self.calc_B()
        eta = self.etas.real.sum()
        
        if reset:
            beta = 0.
            log.debug("CG RESET")
            
            Bs_CG = Bs
        else:
            beta = (eta**2) / eta_0**2
        
            log.debug("BetaFR = %s", beta)
        
            beta = max(0, beta.real)
            
            Bs_CG = [None] * self.L 
            for k in xrange(self.L):
                Bs_CG[k] = Bs[k] + beta * Bs_CG_0[k]

        
        lLb0 = self.lL_before_CF.copy()
        rLb0 = self.rL_before_CF.copy()
        
        h_expect = self.h_expect.real.copy()
        
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
            self.lL_before_CF = lLb0
            self.rL_before_CF = rLb0
            tau, h_min = self.find_min_h_brent(Bs_CG, dtau_init * 0.1, 
                                               use_tangvec_overlap=False)
        
            if h_expect < h_min:
                log.debug("CG RESET FAILED: Setting tau=0!")
                self.lL_before_CF = lLb0
                self.rL_before_CF = rLb0
                tau = 0
        
        return Bs_CG, Bs, eta, tau
        
            
    def export_state(self, userdata=None):
        if userdata is None:
            userdata = self.userdata

        lL = np.asarray(self.ls[-1])
        rL = np.asarray(self.rs[-1])
            
        tosave = np.empty((5), dtype=np.ndarray)
        if self.L == 1:
            tosave[0] = self.As[0]
        else:
            tosave[0] = sp.array(self.As)
        tosave[1] = lL
        tosave[2] = rL
        tosave[3] = self.Ks[0]
        tosave[4] = np.asarray(userdata)
        
        return tosave
            
    def save_state(self, file, userdata=None):
        np.save(file, self.export_state(userdata))
        
    def import_state(self, state, expand=False, truncate=False,
                     expand_q=False, shrink_q=False, refac=0.1, imfac=0.1):
        newAs = state[0]
        newlL = state[1]
        newrL = state[2]
        newK0 = state[3]
        if state.shape[0] > 4:
            self.userdata = state[4]
            
        if len(newAs.shape) == 3:
            newAs = [newAs]
        elif len(newAs.shape) == 4:
            newAs = list(newAs)
        
        if (len(newAs) == self.L and newAs[0].shape == self.As[0].shape):
            self.As = newAs
            self.Ks[0] = newK0
            self.ls[-1] = np.asarray(newlL)
            self.rs[-1] = np.asarray(newrL)
            self.lL_before_CF = self.ls[-1]
            self.rL_before_CF = self.rs[-1]
                
            return True
        elif expand and len(newAs) == self.L and (
        len(newAs[0].shape) == 3) and (newAs[0].shape[0] == 
        self.As[0].shape[0]) and (newAs[0].shape[1] == newAs[0].shape[2]) and (
        newAs[0].shape[1] <= self.As[0].shape[1]):
            newD = self.D
            savedD = newAs[0].shape[1]
            self._init_arrays(savedD, self.q, self.L)
            self.As = newAs
            self.Ks[0] = newK0
            self.ls[-1] = np.asarray(newlL)
            self.rs[-1] = np.asarray(newrL)
            self.expand_D(newD, refac, imfac)
            self.lL_before_CF = self.ls[-1]
            self.rL_before_CF = self.rs[-1]
            log.warning("EXPANDED!")
        elif truncate and len(newAs) == self.L and (len(newAs[0].shape) == 3) \
                and (newAs[0].shape[0] == self.As[0].shape[0]) \
                and (newAs[0].shape[1] == newAs[0].shape[2]) \
                and (newAs[0].shape[1] >= self.As[0].shape[1]):
            newD = self.D
            savedD = newAs[0].shape[1]
            self._init_arrays(savedD, self.q, self.L)
            self.As = newAs
            self.Ks[0] = newK0
            self.ls[-1] = np.asarray(newlL)
            self.rs[-1] = np.asarray(newrL)
            self.update()  # to make absolutely sure we're in CF
            self.truncate(newD, update=True)
            log.warning("TRUNCATED!")
        elif expand_q and len(newAs) == self.L and (len(newAs[0].shape) == 3) and (
        newAs[0].shape[0] <= self.As[0].shape[0]) and (newAs[0].shape[1] == 
        newAs[0].shape[2]) and (newAs[0].shape[1] == self.As[0].shape[1]):
            newQ = self.q
            savedQ = newAs[0].shape[0]
            self._init_arrays(self.D, savedQ, self.L)
            self.As = newAs
            self.Ks[0] = newK0
            self.ls[-1] = np.asarray(newlL)
            self.rs[-1] = np.asarray(newrL)
            self.expand_q(newQ)
            self.lL_before_CF = self.ls[-1]
            self.rL_before_CF = self.rs[-1]
            log.warning("EXPANDED in q!")
        elif shrink_q and len(newAs) == self.L and (len(newAs[0].shape) == 3) and (
        newAs[0].shape[0] >= self.As[0].shape[0]) and (newAs[0].shape[1] == 
        newAs[0].shape[2]) and (newAs[0].shape[1] == self.As[0].shape[1]):
            newQ = self.q
            savedQ = newAs[0].shape[0]
            self._init_arrays(self.D, savedQ)
            self.As = newAs
            self.Ks[0] = newK0
            self.ls[-1] = np.asarray(newlL)
            self.rs[-1] = np.asarray(newrL)
            self.shrink_q(newQ)
            self.lL_before_CF = self.ls[-1]
            self.rL_before_CF = self.rs[-1]
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
        oldK0 = self.Ks[0]
        oldD = self.D

        super(EvoMPS_TDVP_Uniform, self).expand_D(newD, refac=refac, imfac=imfac)
        #self._init_arrays(newD, self.q)
        
        self.Ks[0][:oldD, :oldD] = oldK0
        #self.K[oldD:, :oldD].fill(0 * 1E-3 * la.norm(oldK) / oldD**2)
        #self.K[:oldD, oldD:].fill(0 * 1E-3 * la.norm(oldK) / oldD**2)
        #self.K[oldD:, oldD:].fill(0 * 1E-3 * la.norm(oldK) / oldD**2)
        val = abs(oldK0.mean())
        m.randomize_cmplx(self.Ks[0].ravel()[oldD**2:], a=0, b=val, aj=0, bj=0)
        
    def expect_2s(self, op, n):
        if op is self.ham and self.ham_sites == 2:
            res = tm.eps_r_op_2s_C12_AA34(self.rs[(n + 1) % self.L], self.Cs[n], self.AAs[n])
            return m.adot(self.ls[(n - 1) % self.L], res)
        else:
            return super(EvoMPS_TDVP_Uniform, self).expect_2s(op, n)
            
    def expect_3s(self, op, n):
        if op is self.ham and self.ham_sites == 3:
            res = tm.eps_r_op_3s_C123_AAA456(self.rs[(n + 2) % self.L], self.Cs[n], self.AAAs[n])
            return m.adot(self.ls[(n - 1) % self.L], res)
        else:
            return super(EvoMPS_TDVP_Uniform, self).expect_3s(op, n)