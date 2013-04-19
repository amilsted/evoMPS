# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 17:29:27 2011

@author: Ashley Milsted

TODO:
    - Clean up CG code: Create nice interface?
    - Split out excitations stuff?

"""
import numpy as np
import scipy as sp
import scipy.linalg as la
import scipy.sparse.linalg as las
import scipy.optimize as opti
import tdvp_common as tm
import matmul as m
from mps_uniform import EvoMPS_MPS_Uniform
from mps_uniform_pinv import pinv_1mE
        
class Excite_H_Op:
    def __init__(self, tdvp, donor, p):
        """Creates an Excite_H_Op object, which is a LinearOperator.
        
        This wraps the effective Hamiltonian in terms of MPS tangent vectors
        as a LinearOperator that can be used with SciPy's sparse linear
        algebra routines.
        
        Parameters
        ----------
        tdvp : EvoMPS_TDVP_Uniform
            tdvp object providing the required operations in the matrix representation.
        donor : EvoMPS_TDVP_Uniform
            Second tdvp object (can be the same as tdvp), for example containing a different ground state.
        p : float
            Momentum in units of inverse lattice spacing.
        """
        self.donor = donor
        self.p = p
        
        self.D = tdvp.D
        self.q = tdvp.q
        
        d = (self.q - 1) * self.D**2
        self.shape = (d, d)
        
        self.dtype = np.dtype(tdvp.typ)
        
        if tdvp.ham_sites == 2:
            self.prereq = (tdvp.calc_BHB_prereq(donor))        
            self.calc_BHB = tdvp.calc_BHB
        else:
            self.prereq = (tdvp.calc_BHB_prereq_3s(donor))
            self.calc_BHB = tdvp.calc_BHB_3s
        
        self.calls = 0
        
        self.M_prev = None
        self.y_pi_prev = None
    
    def matvec(self, v):
        x = v.reshape((self.D, (self.q - 1)*self.D))
        
        self.calls += 1
        print "Calls: %u" % self.calls
        
        res, self.M_prev, self.y_pi_prev = self.calc_BHB(x, self.p, self.donor, 
                                                         *self.prereq,
                                                         M_prev=self.M_prev, 
                                                         y_pi_prev=self.y_pi_prev)
        
        return res.ravel()
        
class EvoMPS_TDVP_Uniform(EvoMPS_MPS_Uniform):
        
    def __init__(self, D, q, ham, ham_sites=None, dtype=None):
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
        if ham_sites is None:
            try:
                self.ham_sites = len(ham.shape) / 2
            except AttributeError: #TODO: Try to count arguments using inspect module
                self.ham_sites = 2
        else:
            self.ham_sites = ham_sites
        
        if not (self.ham_sites == 2 or self.ham_sites == 3):
            raise ValueError("Only 2 or 3 site Hamiltonian terms supported!")
        
        super(EvoMPS_TDVP_Uniform, self).__init__(D, q, dtype=dtype)
                        
        self.eta = 0
    
    def _init_arrays(self, D, q):
        super(EvoMPS_TDVP_Uniform, self)._init_arrays(D, q)
        
        ham_shape = []
        for i in xrange(self.ham_sites):
            ham_shape.append(q)
        C_shape = tuple(ham_shape + [D, D])        
        
        self.C = np.zeros(C_shape, dtype=self.typ, order=self.odr)
        
        self.K = np.ones_like(self.A[0])
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
        if self.ham_sites == 2:
            self.C[:] = tm.calc_C_mat_op_AA(ham, self.AA)
        else:
            self.AAA = tm.calc_AAA(self.A, self.A, self.A)
            self.C[:] = tm.calc_C_3s_mat_op_AAA(ham, self.AAA)
    
    def calc_PPinv(self, x, p=0, out=None, left=False, A1=None, A2=None, r=None, 
                   pseudo=True, brute_check=False):
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
            
        if r is None:
            r = self.r
        
        out = pinv_1mE(x, A1, A2, self.l, r, p=p, left=left, pseudo=pseudo, 
                       out=out, tol=self.itr_rtol, 
                       sanity_checks=self.sanity_checks,
                       sanity_tol=self.itr_atol * self.check_fac)

        return out
        
    def calc_K(self):
        """Generates the K matrix used to calculate B.
        
        This also updates the energy-density expectation value self.h.
        
        This is called automatically by self.update().
        
        K contains the (non-trivial) action of the Hamiltonian on the right 
        half of the infinite chain.
        
        It directly depends on A, r, and C.
        """
        if self.ham_sites == 2:
            Hr = tm.eps_r_op_2s_C12_AA34(self.r, self.C, self.AA)
        else:
            Hr = tm.eps_r_op_3s_C123_AAA456(self.r, self.C, self.AAA)
        
        self.h = m.adot(self.l, Hr)
        
        QHr = Hr - self.r * self.h
        
        self.calc_PPinv(QHr, out=self.K)
        
        if self.sanity_checks:
            Ex = tm.eps_r_noop(self.K, self.A, self.A)
            QEQ = Ex - self.r * m.adot(self.l, self.K)
            res = self.K - QEQ
            if not np.allclose(res, QHr):
                print "Sanity check failed: Bad K!"
                print "Off by: " + str(la.norm(res - QHr))
        
    def calc_K_l(self):
        """Generates the left K matrix.
        
        See self.calc_K().
        
        K contains the (non-trivial) action of the Hamiltonian on the left 
        half of the infinite chain.
        
        It directly depends on A, l, and C.
        
        Returns
        -------
        K_left : ndarray
            The left K matrix.
        h : complex
            The energy-density expectation value.
        """
        #Using C is allowed because h is Hermitian
        if self.ham_sites == 2:
            lH = tm.eps_l_op_2s_AA12_C34(self.l, self.AA, self.C)
        else:
            lH = tm.eps_l_op_3s_AAA123_C456(self.l, self.AAA, self.C)
        
        h = m.adot(self.r, lH)
        
        lHQ = lH - self.l * h
        
        self.K_left = self.calc_PPinv(lHQ, left=True, out=self.K_left)
        
        if self.sanity_checks:
            xE = tm.eps_l_noop(self.K_left, self.A, self.A)
            QEQ = xE - self.l * m.adot(self.r, self.K_left)
            res = self.K_left - QEQ
            if not np.allclose(res, lHQ):
                print "Sanity check failed: Bad K_left!"
                print "Off by: " + str(la.norm(res - lHQ))
        
        return self.K_left, h
        
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
            out = np.zeros_like(self.A)
            
        for s in xrange(self.q):
            out[s] = m.mmul(l_sqrt_i, x, m.H(Vsh[s]), r_sqrt_i)
            
        return out
        
    def calc_l_r_roots(self):
        """Calculates the (inverse) square roots of self.l and self.r.
        """
        self.l_sqrt, self.l_sqrt_i, self.r_sqrt, self.r_sqrt_i = tm.calc_l_r_roots(self.l, self.r, self.sanity_checks)
        
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
                
        self.Vsh = tm.calc_Vsh(self.A, self.r_sqrt, sanity_checks=self.sanity_checks)
        
        if self.ham_sites == 2:
            self.x = tm.calc_x(self.K, self.C, self.C, self.r, self.l, self.A, 
                               self.A, self.A, self.l_sqrt, self.l_sqrt_i,
                               self.r_sqrt, self.r_sqrt_i, self.Vsh)
        else:
            self.x = tm.calc_x_3s(self.K, self.C, self.C, self.C, self.r, self.r, 
                                  self.l, self.l, self.A, self.A, self.A, 
                                  self.A, self.A, self.l_sqrt, self.l_sqrt_i,
                                  self.r_sqrt, self.r_sqrt_i, self.Vsh)
        
        if set_eta:
            self.eta = sp.sqrt(m.adot(self.x, self.x))
        
        B = self.get_B_from_x(self.x, self.Vsh, self.l_sqrt_i, self.r_sqrt_i)
        
        if self.sanity_checks:
            #Test gauge-fixing:
            tst = tm.eps_r_noop(self.r, B, self.A)
            if not np.allclose(tst, 0):
                print "Sanity check failed: Gauge-fixing violation!"

        return B
        
    def update(self, restore_CF=True):
        """Updates secondary quantities to reflect the state parameters self.A.
        
        Must be used after taking a step or otherwise changing the 
        parameters self.A before calculating
        physical quantities or taking the next step.
        
        Also (optionally) restores canonical form by calling self.restore_CF().
        
        Parameters
        ----------
        restore_CF : bool (True)
            Whether to restore canonical form.
        """
        super(EvoMPS_TDVP_Uniform, self).update(restore_CF=restore_CF)
        self.calc_C()
        self.calc_K()
        
    def take_step(self, dtau, B=None):
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
        
        self.A += -dtau * B
            
    def take_step_RK4(self, dtau, B_i=None):
        """Take a step using the fourth-order explicit Runge-Kutta method.
        
        This requires more memory than a simple forward Euler step. 
        It is, however, far more accurate with a per-step error of
        order dtau**4.
        
        Parameters
        ----------
        dtau : complex
            The (imaginary or real) amount of imaginary time (tau) to step.
        B_i : ndarray
            B calculated using self.calc_B() (if known, to avoid calculating it again).
        """
        def update():
            self.calc_lr()
            #self.restore_CF() #this really messes things up...
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
        ham_ = self.ham - self.h.real * eyed
            
        V_ = sp.zeros((donor.Vsh.shape[0], donor.Vsh.shape[2], donor.Vsh.shape[1]), dtype=self.typ)
        for s in xrange(donor.q):
            V_[s] = m.H(donor.Vsh[s])
        
        Vri_ = sp.zeros_like(V_)
        for s in xrange(donor.q):
            Vri_[s] = r__sqrt_i.dot_left(V_[s])
            
        Vr_ = sp.zeros_like(V_)
        for s in xrange(donor.q):
            Vr_[s] = r__sqrt.dot_left(V_[s])
            
        C_AhlA = np.empty_like(self.C)
        for u in xrange(self.q):
            for s in xrange(self.q):
                C_AhlA[u, s] = m.H(A[u]).dot(l.dot(A[s]))
        C_AhlA = sp.tensordot(ham_, C_AhlA, ((2, 0), (0, 1)))
        
        C_A_Vrh_ = np.empty((self.q, self.q, A_.shape[1], Vr_.shape[1]), dtype=self.typ)
        for t in xrange(self.q):
            for v in xrange(self.q):
                C_A_Vrh_[t, v] = A_[t].dot(m.H(Vr_[v]))
        C_A_Vrh_ = sp.tensordot(ham_, C_A_Vrh_, ((1, 3), (0, 1)))
                
        C_Vri_A_ = np.empty((self.q, self.q, Vri_.shape[1], A_.shape[2]), dtype=self.typ)
        for s in xrange(self.q):
            for t in xrange(self.q):
                C_Vri_A_[s, t] = Vri_[s].dot(A_[t])
        C_Vri_A_ = sp.tensordot(ham_, C_Vri_A_, ((2, 3), (0, 1)))
        
        C = sp.tensordot(ham_, self.AA, ((2, 3), (0, 1)))

        C_ = sp.tensordot(ham_, AA_, ((2, 3), (0, 1)))
        
        rhs10 = tm.eps_r_op_2s_AA12_C34(r_, AA_, C_Vri_A_)
        
        #NOTE: These C's are good as C12 or C34, but only because h is Hermitian!
        
        return ham_, C, C_, V_, Vr_, Vri_, C_Vri_A_, C_AhlA, C_A_Vrh_, rhs10
            
    def calc_BHB(self, x, p, donor, ham_, C, C_, V_, Vr_, Vri_, 
                 C_Vri_A_, C_AhlA, C_A_Vrh_, rhs10, M_prev=None, y_pi_prev=None): 
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
                print "Sanity check failed: Gauge-fixing violation!"

        if self.sanity_checks:
            B2 = np.zeros_like(B)
            for s in xrange(self.q):
                B2[s] = l_sqrt_i.dot(x.dot(Vri_[s]))
            if not sp.allclose(B, B2, rtol=self.itr_rtol*self.check_fac,
                               atol=self.itr_atol*self.check_fac):
                print "Sanity Fail in calc_BHB! Bad Vri!"
            
        BA_ = tm.calc_AA(B, A_)
        AB = tm.calc_AA(self.A, B)
            
        y = tm.eps_l_noop(l, B, self.A)
        
        if pseudo:
            y = y - m.adot(r_, y) * l #should just = y due to gauge-fixing
        M = self.calc_PPinv(y, p=-p, left=True, A1=A_, r=r_, pseudo=pseudo, out=M_prev)
        #print m.adot(r, M)
        if self.sanity_checks:
            y2 = M - sp.exp(+1.j * p) * tm.eps_l_noop(M, A_, self.A)
            if not sp.allclose(y, y2):
                print "Sanity Fail in calc_BHB! Bad M. Off by: %g" % (la.norm((y - y2).ravel()) / la.norm(y.ravel()))
        Mh = m.H(M)

        res = l_sqrt.dot(
               tm.eps_r_op_2s_AA12_C34(r_, BA_, C_Vri_A_) #1 OK
               + sp.exp(+1.j * p) * tm.eps_r_op_2s_AA12_C34(r_, AB, C_Vri_A_) #3 OK with 4
              )
        #res.fill(0)
        
        res += sp.exp(-1.j * p) * l_sqrt_i.dot(Mh.dot(rhs10)) #10
        
        exp = sp.exp
        subres = sp.zeros_like(res)
        for s in xrange(self.q):
            for t in xrange(self.q):
                subres += (C_AhlA[s, t].dot(B[s]).dot(Vr_[t].conj().T) #2 OK
                         + exp(-1.j * p) * A[t].conj().T.dot(l.dot(B[s])).dot(C_A_Vrh_[s, t]) #4 OK with 3
                         + exp(-2.j * p) * A[s].conj().T.dot(Mh.dot(C_[s, t])).dot(Vr_[t].conj().T)) #12
        res += l_sqrt_i.dot(subres)
        
        res += l_sqrt.dot(tm.eps_r_noop(K__r, B, Vri_)) #5 OK
        
        res += l_sqrt_i.dot(m.H(K_l).dot(tm.eps_r_noop(r__sqrt, B, V_))) #6
        
        res += sp.exp(-1.j * p) * l_sqrt_i.dot(Mh.dot(tm.eps_r_noop(K__r, A_, Vri_))) #8
        
        y1 = sp.exp(+1.j * p) * tm.eps_r_noop(K__r, B, A_) #7
        y2 = sp.exp(+1.j * p) * tm.eps_r_op_2s_AA12_C34(r_, BA_, C_) #9
        y3 = sp.exp(+2.j * p) * tm.eps_r_op_2s_AA12_C34(r_, AB, C_) #11
        
        y = y1 + y2 + y3
        if pseudo:
            y = y - m.adot(l, y) * r_
        y_pi = self.calc_PPinv(y, p=p, A2=A_, r=r_, pseudo=pseudo, out=y_pi_prev)
        #print m.adot(l, y_pi)
        if self.sanity_checks:
            y2 = y_pi - sp.exp(+1.j * p) * tm.eps_r_noop(y_pi, self.A, A_)
            if not sp.allclose(y, y2):
                print "Sanity Fail in calc_BHB! Bad x_pi. Off by: %g" % (la.norm((y - y2).ravel()) / la.norm(y.ravel()))
        
        res += l_sqrt.dot(tm.eps_r_noop(y_pi, self.A, Vri_))
        
        if self.sanity_checks:
            expval = m.adot(x, res) / m.adot(x, x)
            #print "expval = " + str(expval)
            if expval < 0:
                print "Sanity Fail in calc_BHB! H is not pos. semi-definite (" + str(expval) + ")"
            if not abs(expval.imag) < 1E-9:
                print "Sanity Fail in calc_BHB! H is not Hermitian (" + str(expval) + ")"
        
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
        ham_ = self.ham - self.h.real * eyed
        
        V_ = sp.zeros((donor.Vsh.shape[0], donor.Vsh.shape[2], donor.Vsh.shape[1]), dtype=self.typ)
        for s in xrange(donor.q):
            V_[s] = m.H(donor.Vsh[s])
        
        Vri_ = sp.zeros_like(V_)
        for s in xrange(donor.q):
            Vri_[s] = r__sqrt_i.dot_left(V_[s])
            
        Vr_ = sp.zeros_like(V_)
        for s in xrange(donor.q):
            Vr_[s] = r__sqrt.dot_left(V_[s])
            
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
                                  A_.shape[2], A.shape[2]), dtype=self.typ)
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
                                  A_.shape[2], A.shape[2]), dtype=self.typ)
        for t in xrange(self.q):
            for u in xrange(self.q):
                for k in xrange(self.q):
                    C_AA_Vrh[t, u, k] = AA_[t, u].dot(Vr_[k].conj().T)
        C_AA_Vrh = sp.tensordot(ham_, C_AA_Vrh, ((4, 5, 2), (0, 1, 2)))
        
        C_ = sp.tensordot(ham_, AAA_, ((3, 4, 5), (0, 1, 2)))
        
        rhs10 = tm.eps_r_op_3s_C123_AAA456(r_, AAA_, C_Vri_AA_)
        
        #NOTE: These C's are good as C12 or C34, but only because h is Hermitian!
        
        return V_, Vr_, Vri_, C_, C_Vri_AA_, C_AAA_r_Ah_Vrih, C_AhAhlAA, C_AA_r_Ah_Vrih_, C_AAA_Vrh_, C_A_r_Ah_Vrih, C_AhlAA, C_AhAhlA, C_AA_Vrh, rhs10,
    
    def calc_BHB_3s(self, x, p, donor, V_, Vr_, Vri_, C_, C_Vri_AA_, C_AAA_r_Ah_Vrih,
                    C_AhAhlAA, C_AA_r_Ah_Vrih_, C_AAA_Vrh_, C_A_r_Ah_Vrih, 
                    C_AhlAA, C_AhAhlA, C_AA_Vrh, rhs10,
                    M_prev=None, y_pi_prev=None):
        """As for self.calc_BHB(), but for Hamiltonian terms acting on three sites.
        """
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
                print "Sanity check failed: Gauge-fixing violation!"

        if self.sanity_checks:
            B2 = np.zeros_like(B)
            for s in xrange(self.q):
                B2[s] = l_sqrt_i.dot(x.dot(Vri_[s]))
            if not sp.allclose(B, B2, rtol=self.itr_rtol*self.check_fac,
                               atol=self.itr_atol*self.check_fac):
                print "Sanity Fail in calc_BHB! Bad Vri!"
        
        BAA_ = tm.calc_AAA(B, A_, A_)
        ABA_ = tm.calc_AAA(A, B, A_)
        AAB = tm.calc_AAA(A, A, B)
        
        y = tm.eps_l_noop(l, B, self.A)
        
        if pseudo:
            y = y - m.adot(r_, y) * l #should just = y due to gauge-fixing
        M = self.calc_PPinv(y, p=-p, left=True, A1=A_, r=r_, pseudo=pseudo, out=M_prev)
        #print m.adot(r, M)
        if self.sanity_checks:
            y2 = M - sp.exp(+1.j * p) * tm.eps_l_noop(M, A_, self.A)
            if not sp.allclose(y, y2):
                print "Sanity Fail in calc_BHB! Bad M. Off by: %g" % (la.norm((y - y2).ravel()) / la.norm(y.ravel()))
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
        
        res += l_sqrt_i.dot(m.H(K_l).dot(tm.eps_r_noop(r__sqrt, B, V_))) #6
        
        res += sp.exp(-1.j * p) * l_sqrt_i.dot(Mh.dot(tm.eps_r_noop(K__r, A_, Vri_))) #8
        
        y1 = sp.exp(+1.j * p) * tm.eps_r_noop(K__r, B, A_) #7
        y2 = sp.exp(+1.j * p) * tm.eps_r_op_3s_C123_AAA456(r_, BAA_, C_) #9
        y3 = sp.exp(+2.j * p) * tm.eps_r_op_3s_C123_AAA456(r_, ABA_, C_) #11
        y4 = sp.exp(+3.j * p) * tm.eps_r_op_3s_C123_AAA456(r_, AAB, C_) #11b
        
        y = y1 + y2 + y3 + y4
        if pseudo:
            y = y - m.adot(l, y) * r_
        y_pi = self.calc_PPinv(y, p=p, A2=A_, r=r_, pseudo=pseudo, out=y_pi_prev)
        #print m.adot(l, y_pi)
        if self.sanity_checks:
            y2 = y_pi - sp.exp(+1.j * p) * tm.eps_r_noop(y_pi, self.A, A_)
            if not sp.allclose(y, y2):
                print "Sanity Fail in calc_BHB! Bad x_pi. Off by: %g" % (la.norm((y - y2).ravel()) / la.norm(y.ravel()))
        
        res += l_sqrt.dot(tm.eps_r_noop(y_pi, self.A, Vri_))
        
        if self.sanity_checks:
            expval = m.adot(x, res) / m.adot(x, x)
            #print "expval = " + str(expval)
            if expval < 0:
                print "Sanity Fail in calc_BHB! H is not pos. semi-definite (" + str(expval) + ")"
            if not abs(expval.imag) < 1E-9:
                print "Sanity Fail in calc_BHB! H is not Hermitian (" + str(expval) + ")"
        
        return res, M, y_pi        
    
    def _prepare_excite_op_top_triv(self, p):
        if callable(self.ham):
            self.set_ham_array_from_function(self.h_nn)
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

        if not np.allclose(H, m.H(H)):
            print "Warning! H is not Hermitian!"
         
        return la.eigh(H, eigvals_only=not return_eigenvectors)

    def _prepare_excite_op_top_nontriv(self, donor, p):
        if callable(self.ham):
            self.set_ham_array_from_function(self.ham)
        if callable(donor.ham):
            donor.set_ham_array_from_function(donor.ham)
            
        self.calc_lr()
        self.restore_CF()
        donor.calc_lr()
        donor.restore_CF()
        
        self.phase_align(donor)
        
        self.update()
        donor.update()

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

        if not np.allclose(H, m.H(H)):
            print "Warning! H is not Hermitian!"
         
        return la.eigh(H, eigvals_only=not return_eigenvectors)

        
    def find_min_h_brent(self, B, dtau_init, tol=5E-2, skipIfLower=False, 
                         trybracket=True):
        taus=[]
        hs=[]
        
        if len(taus) == 0:
            ls = []
            rs = []
        else:
            ls = [self.l.copy()] * len(taus)
            rs = [self.r.copy()] * len(taus)
        
        def f(tau, *args):
            if tau == 0:
                print (0, "tau=0")
                return self.h.real                
            try:
                i = taus.index(tau)
                print (tau, hs[i], hs[i] - self.h.real, "from stored")
                return hs[i]
            except ValueError:
                for s in xrange(self.q):
                    self.A[s] = A0[s] - tau * B[s]
                
                if len(taus) > 0:
                    nearest_tau_ind = abs(np.array(taus) - tau).argmin()
                    self.l = ls[nearest_tau_ind]
                    self.r = rs[nearest_tau_ind]

                self.calc_lr()
                self.calc_AA()
                self.calc_C()
                
                h = self.expect_2s(self.h_nn)
                
                print (tau, h.real, h.real - self.h.real, self.itr_l, self.itr_r)
                
                res = h.real
                
                taus.append(tau)
                hs.append(res)
                ls.append(self.l.copy())
                rs.append(self.r.copy())
                
                return res
        
        A0 = self.A.copy()                
        AA0 = self.AA.copy()
        C0 = self.C.copy()
        
        try:
            l0 = self.l
            self.l = self.l.A
        except:
            l0 = self.l.copy()
            pass
        
        try:
            r0 = self.r
            self.r = self.r.A
        except:
            r0 = self.r.copy()
            pass
        
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
        self.AA = AA0
        self.C = C0
        
        #hopefully optimize next calc_lr
        nearest_tau_ind = abs(np.array(taus) - tau_opt).argmin()
        self.l_before_CF = ls[nearest_tau_ind]
        self.r_before_CF = rs[nearest_tau_ind]
        
        return tau_opt, h_min
        
    def step_reduces_h(self, B, dtau):
        A0 = self.A.copy()
        AA0 = self.AA.copy()
        C0 = self.C.copy()
        
        try:
            l0 = self.l
            self.l = self.l.A
        except:
            l0 = self.l.copy()
            pass
        
        try:
            r0 = self.r
            self.r = self.r.A
        except:
            r0 = self.r.copy()
            pass
        
        for s in xrange(self.q):
            self.A[s] = A0[s] - dtau * B[s]
        
        self.calc_lr()
        self.calc_AA()
        self.calc_C()
        
        h = self.expect_2s(self.h_nn)
        
        #Must restore everything needed for take_step
        self.A = A0
        self.l = l0
        self.r = r0
        self.AA = AA0
        self.C = C0
        
        return h.real < self.h.real, h

    def calc_B_CG(self, B_CG_0, eta_0, dtau_init, reset=False):
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
        """
        B = self.calc_B()
        eta = self.eta
        
        if reset:
            beta = 0.
            print "RESET CG"
            
            B_CG = B
        else:
            beta = (eta**2) / eta_0**2
        
            print "BetaFR = " + str(beta)
        
            beta = max(0, beta.real)
        
            B_CG = B + beta * B_CG_0

        
        lb0 = self.l_before_CF.copy()
        rb0 = self.r_before_CF.copy()
        
        tau, h_min = self.find_min_h_brent(B_CG, dtau_init,
                                           trybracket=False)
            
        if self.h.real < h_min:
            print "RESET due to energy rise!"
            B_CG = B
            self.l_before_CF = lb0
            self.r_before_CF = rb0
            tau, h_min = self.find_min_h_brent(B_CG, dtau_init * 0.1, trybracket=False)
        
            if self.h.real < h_min:
                print "RESET FAILED: Setting tau=0!"
                self.l_before_CF = lb0
                self.r_before_CF = rb0
                tau = 0
        
        return B_CG, B, eta, tau
        
            
    def export_state(self, userdata=None):
        if userdata is None:
            userdata = self.userdata

        l = np.asarray(self.l)
        r = np.asarray(self.r)
            
        tosave = np.empty((5), dtype=np.ndarray)
        tosave[0] = self.A
        tosave[1] = l
        tosave[2] = r
        tosave[3] = self.K
        tosave[4] = np.asarray(userdata)
        
        return tosave
            
    def save_state(self, file, userdata=None):
        np.save(file, self.export_state(userdata))
        
    def import_state(self, state, expand=False, expand_q=False, shrink_q=False, refac=0.1, imfac=0.1):
        newA = state[0]
        newl = state[1]
        newr = state[2]
        newK = state[3]
        if state.shape[0] > 4:
            self.userdata = state[4]
        
        if (newA.shape == self.A.shape):
            self.A[:] = newA
            self.K[:] = newK

            self.l = np.asarray(newl)
            self.r = np.asarray(newr)
            self.l_before_CF = self.l
            self.r_before_CF = self.r
                
            return True
        elif expand and (len(newA.shape) == 3) and (newA.shape[0] == 
        self.A.shape[0]) and (newA.shape[1] == newA.shape[2]) and (newA.shape[1]
        <= self.A.shape[1]):
            newD = self.D
            savedD = newA.shape[1]
            self._init_arrays(savedD, self.q)
            self.A[:] = newA
            self.l = newl
            self.r = newr            
            self.K[:] = newK
            self.expand_D(newD, refac, imfac)
            self.l_before_CF = self.l
            self.r_before_CF = self.r
            print "EXPANDED!"
        elif expand_q and (len(newA.shape) == 3) and (newA.shape[0] <= 
        self.A.shape[0]) and (newA.shape[1] == newA.shape[2]) and (newA.shape[1]
        == self.A.shape[1]):
            newQ = self.q
            savedQ = newA.shape[0]
            self._init_arrays(self.D, savedQ)
            self.A[:] = newA
            self.l = newl
            self.r = newr
            self.K[:] = newK
            self.expand_q(newQ)
            self.l_before_CF = self.l
            self.r_before_CF = self.r
            print "EXPANDED in q!"
        elif shrink_q and (len(newA.shape) == 3) and (newA.shape[0] >= 
        self.A.shape[0]) and (newA.shape[1] == newA.shape[2]) and (newA.shape[1]
        == self.A.shape[1]):
            newQ = self.q
            savedQ = newA.shape[0]
            self._init_arrays(self.D, savedQ)
            self.A[:] = newA
            self.l = newl
            self.r = newr
            self.K[:] = newK
            self.shrink_q(newQ)
            self.l_before_CF = self.l
            self.r_before_CF = self.r
            print "SHRUNK in q!"
        else:
            return False
            
    def load_state(self, file, expand=False, expand_q=False, shrink_q=False, refac=0.1, imfac=0.1):
        state = np.load(file)
        return self.import_state(state, expand=expand, expand_q=expand_q, shrink_q=shrink_q, refac=refac, imfac=imfac)
            
    def set_q(self, newq):
        oldK = self.K        
        super(EvoMPS_TDVP_Uniform, self).set_q(newq)        
        self.K = oldK
                    
    def expand_D(self, newD, refac=100, imfac=0):
        oldK = self.K
        oldD = self.D
                
        super(EvoMPS_TDVP_Uniform, self).expand_D(newD, refac=refac, imfac=imfac)
        #self._init_arrays(newD, self.q)
                
        self.K[:oldD, :oldD] = oldK
        self.K[oldD:, :oldD].fill(la.norm(oldK) / oldD**2)
        self.K[:oldD, oldD:].fill(la.norm(oldK) / oldD**2)
        self.K[oldD:, oldD:].fill(la.norm(oldK) / oldD**2)
        
    def expect_2s(self, op):
        if op == self.h_nn:
            res = tm.eps_r_op_2s_C12_AA34(self.r, self.C, self.AA)
            return m.adot(self.l, res)
        else:
            return super(EvoMPS_TDVP_Uniform, self).expect_2s(op)