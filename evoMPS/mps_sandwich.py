# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 17:29:27 2011

@author: Ashley Milsted

"""
from __future__ import absolute_import, division, print_function

import scipy as sp
import scipy.linalg as la
from . import matmul as mm
from . import tdvp_common as tm
from .mps_gen import EvoMPS_MPS_Generic
from .mps_uniform_pinv import pinv_1mE
import copy


class EvoMPS_MPS_Sandwich(EvoMPS_MPS_Generic):

    def __init__(self, N, uni_ground):
        self.odr = 'C'
        self.typ = sp.complex128

        self.zero_tol = sp.finfo(self.typ).resolution
        """Tolerance for detecting zeros. This is used when (pseudo-) inverting 
           l and r."""

        self._sanity_checks = False

        self.N = N
        """The number of sites. Do not change after initializing."""
        
        self.N_centre = N // 2
        """The 'centre' site. This affects the gauge-fixing and canonical
           form. It is the site between the left-gauge parts and the 
           right-gauge parts."""
           
        self.D = sp.repeat(uni_ground.D, self.N + 2)
        """Vector containing the bond-dimensions. A[n] is a 
           q[n] x D[n - 1] x D[n] tensor."""
           
        self.q = sp.repeat(uni_ground.q, self.N + 2)
        """Vector containing the site Hilbert space dimensions. A[n] is a 
           q[n] x D[n - 1] x D[n] tensor."""
          
        self.uni_l = copy.deepcopy(uni_ground)
        self.uni_l.symm_gauge = True
        self.uni_l.sanity_checks = self.sanity_checks
        self.uni_l.update()
        
        if not N % self.uni_l.L == 0:
            print("Warning: Length of nonuniform window is not a multiple of the uniform block size.")

        self.uni_r = copy.deepcopy(self.uni_l)

        self.grown_left = 0
        self.grown_right = 0
        self.shrunk_left = 0
        self.shrunk_right = 0

        self._init_arrays()

        for n in range(1, self.N + 1):
            self.A[n][:] = self.uni_l.A[(n - 1) % self.uni_l.L]
        
        for n in range(self.N + 2):
            self.r[n][:] = sp.asarray(self.uni_l.r[(n - 1) % self.uni_l.L])
            self.l[n][:] = sp.asarray(self.uni_l.l[(n - 1) % self.uni_l.L])


    def _init_arrays(self):        
        #Deliberately add a None to the end to catch [-1] indexing!
        self.A = sp.empty((self.N + 3), dtype=sp.ndarray) #Elements 1..N

        self.r = sp.empty((self.N + 3), dtype=sp.ndarray) #Elements 0..N
        self.l = sp.empty((self.N + 3), dtype=sp.ndarray)

        self.l[0] = sp.zeros((self.D[0], self.D[0]), dtype=self.typ, order=self.odr)
        self.r[0] = sp.zeros((self.D[0], self.D[0]), dtype=self.typ, order=self.odr)
        self.A[0] = None
        for n in range(0, self.N + 2):
            self.r[n] = sp.zeros((self.D[n], self.D[n]), dtype=self.typ, order=self.odr)
            self.l[n] = sp.zeros((self.D[n], self.D[n]), dtype=self.typ, order=self.odr)
            if 0 < n <= self.N:
                self.A[n] = sp.zeros((self.q[n], self.D[n-1], self.D[n]), dtype=self.typ, order=self.odr)

    @property
    def sanity_checks(self):
        """Whether to perform additional (potentially costly) sanity checks."""
        return self._sanity_checks

    @sanity_checks.setter
    def sanity_checks(self, value):
        self._sanity_checks = value
        self.uni_l.sanity_checks = value
        self.uni_r.sanity_checks = value

    def correct_bond_dimension(self):
        raise NotImplementedError("correct_bond_dimension not currently implemented in sandwich case")

    def update(self, restore_CF=True, normalize=True):
        """Perform all necessary steps needed before taking the next step,
        or calculating expectation values etc., is possible.
        """
        if restore_CF:
            self.restore_CF()
        else:
            if normalize:
                self.calc_l(n_high=self.N_centre - 1)
                self.calc_r(n_low=self.N_centre - 1)
                self.simple_renorm(update_lr=True)
            else:
                self.calc_l()
                self.calc_r()


    def calc_l(self, n_low=-1, n_high=-1):
        """Updates the l matrices using the current state.
        Implements step 5 of the TDVP algorithm or, equivalently, eqn. (41).
        (arXiv:1103.0936v2 [cond-mat.str-el])
        """
        if n_low < 0:
            n_low = 1
        if n_high < 0:
            n_high = self.N

        self.l[0] = self.uni_l.l[-1]
        super(EvoMPS_MPS_Sandwich, self).calc_l(n_low=n_low, n_high=n_high)
        
        self.l[self.N + 1] = tm.eps_l_noop(self.l[self.N], self.uni_r.A[0], self.uni_r.A[0])

    def calc_r(self, n_low=-1, n_high=-1):
        """Updates the r matrices using the current state.
        Implements step 5 of the TDVP algorithm or, equivalently, eqn. (41).
        (arXiv:1103.0936v2 [cond-mat.str-el])
        """
        if n_low < 0:
            n_low = 0
        if n_high < 0:
            n_high = self.N - 1

        self.r[self.N] = self.uni_r.r[-1]
        super(EvoMPS_MPS_Sandwich, self).calc_r(n_low=n_low, n_high=n_high)

    def simple_renorm(self, update_lr=True):
        """Renormalizes the state in by multiplying one of the parameter
        tensors by a factor.
        """
        norm = mm.adot(self.l[self.N_centre - 1], self.r[self.N_centre - 1])

        if abs(1 - norm) > 1E-15:
            self.A[self.N_centre] *= 1 / sp.sqrt(norm)

        if update_lr:
            self.calc_l(n_low=self.N_centre)
            self.calc_r(n_high=self.N_centre-1)

    def restore_CF_dbg(self):
        for n in range(self.N + 2):
            print((n, self.l[n].trace().real, self.r[n].trace().real,
                   mm.adot(self.l[n], self.r[n]).real))

        norm_r = mm.adot(self.uni_r.l[-1], self.r[self.N])
        norm_l = mm.adot(self.l[0], self.uni_l.r[-1])
        print("norm of uni_r: ", norm_r)
        print("norm of uni_l: ", norm_l)

        #r_m1 = self.eps_r(0, self.r[0])
        #print mm.adot(self.l[0], r_m1).real

        norm = mm.adot(self.l[0], self.r[0]).real
        
        try:
            h = sp.empty((self.N + 1), dtype=self.typ)
            for n in range(self.N + 1):
                h[n] = self.expect_2s(self.h_nn[n], n)
            h *= 1/norm
    
            #self.uni_l.A = self.A[0] #FIXME: Not sure how to handle this yet...
            self.uni_l.l[-1] = self.l[0]
            self.uni_l.calc_AA()
            h_left = self.uni_l.expect_2s(self.uni_l.ham.copy()) / norm_l
    
            #self.uni_r.A = self.A[self.N + 1]
            self.uni_r.r[-1] = self.r[self.N]
            self.uni_r.calc_AA()
            h_right = self.uni_r.expect_2s(self.uni_r.ham.copy()) / norm_r
    
            return h, h_left, h_right
        except AttributeError:
            return sp.array([0]), 0, 0

    def _restore_CF_ONR(self):
        nc = self.N_centre

        #Want: r[n >= nc] = eye

        #First do uni_r
        uGs, uG_is = self.uni_r.restore_RCF(zero_tol=self.zero_tol, diag_l=False, ret_g=True) #FIXME: Don't always to all!
        Gi = uGs[-1] #Inverse is on the other side in the uniform code.
        self.r[self.N] = self.uni_r.r[-1]
        
        for n in range(self.N, nc, -1):
            self.r[n - 1], Gm1, Gm1_i = tm.restore_RCF_r(self.A[n], self.r[n],
                                             Gi, zero_tol=self.zero_tol,
                                             sanity_checks=self.sanity_checks)

            Gi = Gm1_i

        self.r[self.N + 1] = self.r[self.N]

        #G is now G_nc
        for s in range(self.q[nc]):
            self.A[nc][s] = self.A[nc][s].dot(Gi)

        #Now want: l[n < nc] = eye
        uGs, uG_is = self.uni_l.restore_LCF(zero_tol=self.zero_tol, diag_r=False, ret_g=True)
        Gm1 = uG_is[-1]
        self.l[0] = self.uni_l.l[-1]
        lm1 = self.l[0]
        for n in range(1, nc):
            self.l[n], G, Gi = tm.restore_LCF_l(self.A[n], lm1, Gm1,
                                                zero_tol=self.zero_tol,
                                                sanity_checks=self.sanity_checks)

            lm1 = self.l[n]
            Gm1 = G

        #Gm1 is now G_nc-1
        for s in range(self.q[nc]):
            self.A[nc][s] = Gm1.dot(self.A[nc][s])


    def _restore_CF_diag(self, dbg=False):
        nc = self.N_centre

        #Want: r[0 <= n < nc] diagonal
        Ui = sp.eye(self.D[nc], dtype=self.typ)
        for n in range(nc, 0, -1):
            self.r[n - 1], Um1, Um1_i = tm.restore_LCF_r(self.A[n], self.r[n],
                                                         Ui, sanity_checks=self.sanity_checks)
            Ui = Um1_i

        #Now U is U_0
        U = Um1
        for s in range(self.q[0]):
            self.uni_l.A[0][s] = U.dot(self.uni_l.A[0][s])
            self.uni_l.A[-1][s] = self.uni_l.A[-1][s].dot(Ui)
        self.uni_l.r[-1] = U.dot(self.uni_l.r[-1].dot(U.conj().T))

        #And now: l[nc <= n <= N] diagonal
        if dbg:
            Um1 = sp.eye(self.D[nc - 1], dtype=self.typ)
        else:
            Um1 = mm.eyemat(self.D[nc - 1], dtype=self.typ) #FIXME: This only works if l[nc - 1] is a special matrix type        
        for n in range(nc, self.N + 1):
            self.l[n], U, Ui = tm.restore_RCF_l(self.A[n], self.l[n - 1], Um1,
                                                sanity_checks=self.sanity_checks)
            Um1 = U

        #Now, Um1 = U_N
        Um1_i = Ui
        for s in range(self.q[0]):
            self.uni_r.A[0][s] = Um1.dot(self.uni_r.A[0][s])
            self.uni_r.A[-1][s] = self.uni_r.A[-1][s].dot(Um1_i)
        self.uni_r.l[-1] = Um1_i.conj().T.dot(self.uni_r.l[-1].dot(Um1_i))

    def restore_CF(self, dbg=False):
        if dbg:
            self.calc_l()
            self.calc_r()
            self.simple_renorm()
            print("BEFORE...")
            h_before, h_left_before, h_right_before = self.restore_CF_dbg()

            print((h_left_before, h_before, h_right_before))

        self.uni_l.calc_lr() #Ensures largest ev of E=1
        if dbg:
            print("uni_l calc_lr iterations: ", (self.uni_l.itr_l, self.uni_l.itr_r))

#        if self.sanity_checks:
#            if not sp.allclose(self.uni_l.l, self.l[0], atol=1E-12, rtol=1E-12):
#                print "Sanity Fail in restore_CF!: True l[0] and l[L] mismatch!", la.norm(self.l[0] - self.uni_l.l)
#
#            if not sp.allclose(self.uni_l.r, r_old, atol=1E-12, rtol=1E-12):
#                print "Sanity Fail in restore_CF!: Bad r[L]!", la.norm(r_old - self.uni_l.r)
#
#            if not sp.allclose(self.uni_l.A, self.A[0], atol=1E-12, rtol=1E-12):
#                print "Sanity Fail in restore_CF!: A[0] was scaled!", la.norm(self.A[0] - self.uni_l.A)

        self.l[0] = self.uni_l.l[-1]
        self.A[0] = None

        self.uni_r.calc_lr() #Ensures largest ev of E=1
        if dbg:
            print("uni_r calc_lr iterations: ", (self.uni_r.itr_l, self.uni_r.itr_r))
#        if self.sanity_checks:
#            if not sp.allclose(self.uni_r.A, self.A[self.N + 1], atol=1E-12, rtol=1E-12):
#                print "Sanity Fail in restore_CF!: A[R] was scaled! ", la.norm(self.A[self.N + 1] - self.uni_r.A)
#
#            if not sp.allclose(l_old, self.uni_r.l, atol=1E-12, rtol=1E-12):
#                print "Sanity Fail in restore_CF!: Bad l[R]! ", la.norm(l_old - self.uni_r.l)
#
#            if not sp.allclose(self.r[self.N], self.uni_r.r, atol=1E-12, rtol=1E-12):
#                print "Sanity Fail in restore_CF!: r[N] and r[R] mismatch!", la.norm(self.r[self.N] - self.uni_r.r)

        self.r[self.N] = self.uni_r.r[-1]
        self.A[self.N + 1] = None

        self._restore_CF_ONR()

        nc = self.N_centre
        self.r[nc - 1] = tm.eps_r_noop(self.r[nc], self.A[nc], self.A[nc])
        self.simple_renorm(update_lr=False)

        if dbg:
            self.calc_l()
            self.calc_r()
            print("AFTER ONR...")
            h_mid, h_left_mid, h_right_mid = self.restore_CF_dbg()

            print((h_left_mid, h_mid, h_right_mid))

        self._restore_CF_diag(dbg=dbg)

        #l[0] is identity, r_L = ?
        self.uni_l.l_before_CF = self.uni_l.l[-1]
        self.uni_l.r_before_CF = self.uni_l.r[-1]

        #r[N] is identity, l_R = ?
        self.uni_r.l_before_CF = self.uni_r.l[-1]
        self.uni_r.r_before_CF = self.uni_r.r[-1]

        #Set l[N + 1] as well...
        self.l[self.N + 1][:] = tm.eps_l_noop(self.l[self.N],
                                              self.uni_r.A[0],
                                              self.uni_r.A[0])

        if self.sanity_checks:
            l_n = self.l[0]
            for n in range(1, self.N + 1):
                l_n = tm.eps_l_noop(l_n, self.A[n], self.A[n])
                if not sp.allclose(l_n, self.l[n], atol=1E-12, rtol=1E-12):
                    print("Sanity Fail in restore_CF!: l_%u is bad" % n)

            r_nm1 = self.r[self.N]
            for n in reversed(range(1, self.N)):
                r_nm1 = tm.eps_r_noop(r_nm1, self.A[n], self.A[n])
                if not sp.allclose(r_nm1, self.r[n - 1], atol=1E-12, rtol=1E-12):
                    print("Sanity Fail in restore_CF!: r_%u is bad" % (n - 1))
        if dbg:
            print("AFTER...")
            h_after, h_left_after, h_right_after = self.restore_CF_dbg()

            print((h_left_after, h_after, h_right_after))

            print(h_after - h_before)

            print((h_after.sum() - h_before.sum() + h_left_after - h_left_before
                   + h_right_after - h_right_before))


    def check_RCF(self):
        raise NotImplementedError("check_RCF not implemented in sandwich case")

    def grow_left(self, m):
        """Grow the generic region to the left by m * L sites, where L is the 
        uniform block size.
        """
        oldA = self.A
        oldl_0 = self.l[0]
        oldr_N = self.r[self.N]

        oldD = self.D
        oldq = self.q
        
        m *= self.uni_l.L

        self.D = sp.zeros((self.N + m + 2), dtype=int)
        self.q = sp.zeros((self.N + m + 2), dtype=int)

        self.D[m:] = oldD
        self.q[m:] = oldq

        self.D[:m] = [self.uni_l.D] * m
        self.q[:m] = [self.uni_l.q] * m

        self.N = self.N + m
        self._init_arrays()

        for n in range(m):
            self.A[n + 1][:] = self.uni_l.A[n % self.uni_l.L]

        for n in range(m + 1, self.N + 1):
            self.A[n][:] = oldA[n - m]

        self.l[0] = oldl_0
        self.r[self.N] = oldr_N
        self.r[self.N + 1] = self.r[self.N]

        self.grown_left += m


    def grow_right(self, m):
        """Grow the generic region to the right by m * L sites, where L is the 
        uniform block size.
        """
        oldA = self.A
        oldl_0 = self.l[0]
        oldr_N = self.r[self.N]

        oldD = self.D
        oldq = self.q
        
        m *= self.uni_r.L

        self.D = sp.zeros((self.N + m + 2), dtype=int)
        self.q = sp.zeros((self.N + m + 2), dtype=int)

        self.D[:-m] = oldD
        self.q[:-m] = oldq

        self.D[-m:] = [self.uni_r.D] * m
        self.q[-m:] = [self.uni_r.q] * m

        self.N = self.N + m
        self._init_arrays()

        for n in range(1, self.N - m + 1):
            self.A[n][:] = oldA[n]

        for n in range(m):
            self.A[self.N - m + 1 + n][:] = self.uni_r.A[n % self.uni_r.L]

        self.l[0] = oldl_0
        self.r[self.N] = oldr_N
        self.r[self.N + 1] = self.r[self.N]

        self.grown_right += m

    #TODO: Add shrinking by fidelity and gauge-alignment!

    def get_A(self, n):
        if n < 1:
            return self.uni_l.A[(n - 1) % self.uni_l.L]
        elif n > self.N:
            return self.uni_r.A[(n - self.N - 1) % self.uni_r.L]
        else:
            return self.A[n]

    def get_l(self, n):
        """Gets an l[n], even if n < 0 or n > N + 1
        """
        if 0 < n <= self.N + 1:
            return self.l[n]
        elif n < 1:
            return self.uni_l.l[(n - 1) % self.uni_l.L]
        else:
            l_m1 = self.l[self.N + 1] #this is position 0 in the block
            for m in (sp.arange(self.N + 2, n + 1) - 1) % self.uni_r.L:
                l_m1 = tm.eps_l_noop(l_m1, self.uni_r.A[m], self.uni_r.A[m])
            return l_m1

    def get_r(self, n, r_np1=None):
        """Gets an r[n], even if n < 0 or n > N + 1
        """
        if 0 <= n <= self.N + 1:
            return self.r[n]
        elif n > self.N + 1:
            return self.uni_r.r[(n - 1) % self.uni_r.L]
        else:
            if r_np1 is None:
                r_m = self.r[0]
                for m in (sp.arange(0, n, -1) - 1) % self.uni_l.L:
                    r_m = tm.eps_r_noop(r_m, self.uni_l.A[m], self.uni_l.A[m])
                return r_m
            else:
                m = n % self.uni_l.L
                return tm.eps_r_noop(r_np1, self.uni_l.A[m], self.uni_l.A[m])

    def expect_1s(self, op, n):
        """Computes the expectation value of a single-site operator.

        A single-site operator is represented as a function taking three
        integer arguments (n, s, t) where n is the site number and s, t
        range from 0 to q[n] - 1 and define the requested matrix element <s|o|t>.

        Assumes that the state is normalized.

        Parameters
        ----------
        o : ndarray or callable
            The operator.
        n : int
            The site number.
        """
        A = self.get_A(n)

        if callable(op):
            op = sp.vectorize(op, otypes=[sp.complex128])
            op = sp.fromfunction(op, (A.shape[0], A.shape[0]))

        res = tm.eps_r_op_1s(self.get_r(n), A, A, op)
        return mm.adot(self.get_l(n - 1), res)

    def expect_2s(self, op, n):
        """Computes the expectation value of a nearest-neighbour two-site operator.

        The operator should be a q[n] x q[n + 1] x q[n] x q[n + 1] array
        such that op[s, t, u, v] = <st|op|uv> or a function of the form
        op(s, t, u, v) = <st|op|uv>.

        Parameters
        ----------
        o : ndarray or callable
            The operator array or function.
        n : int
            The leftmost site number (operator acts on n, n + 1).
        """
        A = self.get_A(n)
        Ap1 = self.get_A(n + 1)
        AA = tm.calc_AA(A, Ap1)

        if callable(op):
            op = sp.vectorize(op, otypes=[sp.complex128])
            op = sp.fromfunction(op, (A.shape[0], Ap1.shape[0], A.shape[0], Ap1.shape[0]))

        C = tm.calc_C_mat_op_AA(op, AA)
        res = tm.eps_r_op_2s_C12_AA34(self.get_r(n + 1), C, AA)
        return mm.adot(self.get_l(n - 1), res)

    def expect_1s_cor(self, op1, op2, n1, n2):
        """Computes the correlation of two single site operators acting on two different sites.

        See expect_1S().

        n1 must be smaller than n2.

        Assumes that the state is normalized.

        Parameters
        ----------
        op1 : function
            The first operator, acting on the first site.
        op2 : function
            The second operator, acting on the second site.
        n1 : int
            The site number of the first site.
        n2 : int
            The site number of the second site (must be > n1).
        """
        A1 = self.get_A(n1)
        A2 = self.get_A(n2)

        if callable(op1):
            op1 = sp.vectorize(op1, otypes=[sp.complex128])
            op1 = sp.fromfunction(op1, (A1.shape[0], A1.shape[0]))

        if callable(op2):
            op2 = sp.vectorize(op2, otypes=[sp.complex128])
            op2 = sp.fromfunction(op2, (A2.shape[0], A2.shape[0]))

        r_n = tm.eps_r_op_1s(self.get_r(n2), A2, A2, op2)

        for n in reversed(range(n1 + 1, n2)):
            r_n = tm.eps_r_noop(r_n, self.get_A(n), self.get_A(n))

        r_n = tm.eps_r_op_1s(r_n, A1, A1, op1)

        return mm.adot(self.get_l(n1 - 1), r_n)

    def density_2s(self, n1, n2):
        """Returns a reduced density matrix for a pair of sites.
        
        Currently only supports sites in the nonuniform window.

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
        ln1m1 = self.get_l(n1 - 1)

        for s2 in range(self.q[n2]):
            for t2 in range(self.q[n2]):
                r_n2 = mm.mmul(self.A[n2][t2], self.r[n2], mm.H(self.A[n2][s2]))

                r_n = r_n2
                for n in reversed(range(n1 + 1, n2)):
                    r_n = tm.eps_r_noop(r_n, self.A[n], self.A[n])

                for s1 in range(self.q[n1]):
                    for t1 in range(self.q[n1]):
                        r_n1 = mm.mmul(self.A[n1][t1], r_n, mm.H(self.A[n1][s1]))
                        tmp = mm.adot(ln1m1, r_n1)
                        rho[s1 * self.q[n1] + s2, t1 * self.q[n1] + t2] = tmp
        return rho

    def apply_op_1s(self, op, n):
        """Applies a one-site operator op to site n.

        Can be used to create excitations.
        """

        if not (0 < n <= self.N):
            raise ValueError("Operators can only be applied to sites 1 to N!")

        newA = sp.zeros_like(self.A[n])

        if callable(op):
            op = sp.vectorize(op, otypes=[sp.complex128])
            op = sp.fromfunction(op, (self.q[n], self.q[n]))

        for s in range(self.q[n]):
            for t in range(self.q[n]):
                newA[s] += self.A[n][t] * op[s, t]

        self.A[n] = newA

    def overlap(self, other, sanity_checks=False):
        if not self.N == other.N:
            print("States must have same number of non-uniform sites!")
            return

        if not sp.all(self.D == other.D):
            print("States must have same bond-dimensions!")
            return

        if not sp.all(self.q == other.q):
            print("States must have same Hilbert-space dimensions!")
            return

        dL, phiL, gLl = self.uni_l.fidelity_per_site(other.uni_l, full_output=True, left=True)

        dR, phiR, gRr = self.uni_r.fidelity_per_site(other.uni_r, full_output=True, left=False)

        gLl = gLl.reshape(self.uni_l.D, self.uni_l.D)
        gRr = gRr.reshape(self.uni_r.D, self.uni_r.D)

        gr = mm.H(la.inv(gRr).dot(sp.asarray(self.uni_r.r[-1])))
        gri = mm.H(la.inv(sp.asarray(self.uni_r.r[-1])).dot(gRr))

#        if sanity_checks: #FIXME: Not gonna work for L > 1....
#            AR = other.uni_r.A.copy()
#            for s in xrange(AR.shape[0]):
#                AR[s] = gr.dot(AR[s]).dot(gri)
#
#            print la.norm(AR - self.uni_r.A)

        r = gr.dot(sp.asarray(other.uni_r.r)).dot(mm.H(gr))
        fac = la.norm(sp.asarray(self.uni_r.r)) / la.norm(r)
        gr *= sp.sqrt(fac)
        gri /= sp.sqrt(fac)

        if sanity_checks:
            r *= fac
            print(la.norm(r - self.uni_r.r[-1]))

#            AN = other.A[self.N].copy()
#            for s in xrange(AN.shape[0]):
#                AN[s] = AN[s].dot(gri)
#            r = tm.eps_r_noop(self.uni_r.r, AN, AN)
#            print la.norm(r - other.r[self.N - 1])

        gl = la.inv(sp.asarray(self.uni_l.l[-1])).dot(gLl)
        gli = la.inv(gLl).dot(sp.asarray(self.uni_l.l[-1]))

        l = mm.H(gli).dot(sp.asarray(other.uni_l.l[-1])).dot(gli)
        fac = la.norm(sp.asarray(self.uni_l.l[-1])) / la.norm(l)

        gli *= sp.sqrt(fac)
        gl /= sp.sqrt(fac)

        if sanity_checks:
            l *= fac
            print(la.norm(l - self.uni_l.l[-1]))

            l = mm.H(gli).dot(sp.asarray(other.uni_l.l[-1])).dot(gli)
            print(la.norm(l - self.uni_l.l[-1]))

        #print (dL, dR, phiL, phiR)

        if not abs(dL - 1) < 1E-12:
            print("Left bulk states do not match!")
            return 0

        if not abs(dR - 1) < 1E-12:
            print("Right bulk states do not match!")
            return 0

        AN = other.A[self.N].copy()
        for s in range(AN.shape[0]):
            AN[s] = AN[s].dot(gri)

        A1 = other.A[1].copy()
        for s in range(A1.shape[0]):
            A1[s] = gl.dot(A1[s])

        r = tm.eps_r_noop(self.uni_r.r[-1], self.A[self.N], AN)
        for n in range(self.N - 1, 1, -1):
            r = tm.eps_r_noop(r, self.A[n], other.A[n])
        r = tm.eps_r_noop(r, self.A[1], A1)

        return mm.adot(self.uni_l.l[-1], r)

    def overlap_tangent(self, B, p, top_triv=True):
        """Inner product of the state with an MPS tangent vector.
        Assumes the uniform left and right parts of the tangent vector terms
        are defined by the *same tensors* that define the left and right bulk
        parts of the sandwich state.
        This will generically require computing excitations anew using the left
        and right bulk tensors.

        It is also assumed that `B` is "right gauge-fixing" with respect to
        its bulk tensors.
        """
        if not (self.uni_l.L == 1 and self.uni_r.L == 1):
            raise ValueError("Bulk unit cell size must currently be 1.")
        rR = self.uni_r.r[0]
        AR = self.uni_r.A[0]
        check_gf = tm.eps_r_noop(rR, AR, B)
        print("check GF:", la.norm(check_gf))

        lL = self.uni_l.l[0]
        AL = self.uni_l.A[0]

        pseudo = top_triv

        # FIXME: rR is not correct here since, even if top_triv is true,
        #        AL and AR need not be equal.
        vBL = pinv_1mE(
            tm.eps_l_noop(lL, AL, B), [AL], [AR], lL, rR, p=p, 
            left=True, pseudo=pseudo)

        rs = [rR]
        for j in range(self.N, 0, -1):
            rs.insert(0, tm.eps_r_noop(rs[0], self.A[j], AR))

        res = mm.adot(vBL, rs[0]) * sp.exp(-1.j * p)
        print(res)

        l = lL
        for j in range(1, self.N + 1):
            res_j = sp.exp(1.j * p * j) * mm.adot(
                l, tm.eps_r_noop(rs[j], self.A[j], B))
            print(res_j)
            res += res_j
            l = tm.eps_l_noop(l, self.A[j], AL)

        return res

    def save_state(self, file):
        raise NotImplementedError("save_state not implemented in sandwich case")

    def load_state(self, file):
        raise NotImplementedError("load_state not implemented in sandwich case")
