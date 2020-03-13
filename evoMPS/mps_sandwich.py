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
from .mps_uniform import EvoMPS_MPS_Uniform
from .mps_uniform_pinv import pinv_1mE
import copy

def sandwich_from_tensors(As_L, As_C, As_R):
    """Creates a sandwich state from a collection of MPS tensors.
    `As_L` is the left uniform bulk unit cell.
    `As_C` is the central nonuniform window.
    `As_R` is the right uniform bulk unit cell.
    """
    N = len(As_C)

    qL = As_L[0].shape[0]
    DL = As_L[0].shape[1]
    dtype = As_L[0].dtype
    sw = EvoMPS_MPS_Sandwich(
        N,
        EvoMPS_MPS_Uniform(
            DL, qL, L=len(As_L), dtype=dtype, do_update=False)
    )

    for n in range(1, N + 1):
        sw.A[n] = As_C[n - 1]

    Ds = sp.array(
        [DL] +
        [sw.A[n].shape[2] for n in range(1, N + 1)] +
        [As_R[0].shape[1]]
    )
    sw.D = Ds

    sw.uni_l.A = As_L
    sw.uni_r.A = As_R
    sw.uni_l.update(restore_CF=False)
    sw.uni_r.update(restore_CF=False)

    sw.update(restore_CF=False, normalize=False)

    return sw


class EvoMPS_MPS_Sandwich(EvoMPS_MPS_Generic):

    def __init__(self, N, uni_ground, uni_right=None, update_bulks=True):
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
        self.uni_l.sanity_checks = self.sanity_checks
        if update_bulks:
            self.uni_l.symm_gauge = True
            self.uni_l.update()
        
        if not N % self.uni_l.L == 0:
            print("Warning: Length of nonuniform window is not a multiple of the uniform block size.")

        if uni_right is not None:
            self.uni_r = copy.deepcopy(uni_right)
            self.uni_r.sanity_checks = self.sanity_checks
            if update_bulks:
                self.uni_r.symm_gauge = True
                self.uni_r.update()
        else:
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

    @classmethod
    def from_tensors(cls, As_L, As_C, As_R):
        """Creates a sandwich state from a collection of MPS tensors.
        `As_L` is the left uniform bulk unit cell.
        `As_C` is the central nonuniform window.
        `As_R` is the right uniform bulk unit cell.
        """
        N = len(As_C)

        qL, DL = As_L[0].shape[0:2]
        qR, DR = As_R[0].shape[0:2]
        dtype = As_L[0].dtype
        sw = cls(
            N,
            EvoMPS_MPS_Uniform(
                DL, qL, L=len(As_L), dtype=dtype, do_update=False),
            uni_right=EvoMPS_MPS_Uniform(
                DR, qR, L=len(As_R), dtype=dtype, do_update=False),
            update_bulks=False
        )

        for n in range(1, N + 1):
            sw.A[n] = As_C[n - 1]

        Ds = sp.array(
            [DL] +
            [sw.A[n].shape[2] for n in range(1, N + 1)] +
            [As_R[0].shape[1]]
        )
        sw.D = Ds

        sw.uni_l.A = As_L
        sw.uni_r.A = As_R
        sw.uni_l.update(restore_CF=False)
        sw.uni_r.update(restore_CF=False)

        sw.update(restore_CF=False, normalize=False)

        return sw

    @classmethod
    def from_file(cls, file_name):
        toload = sp.load(file_name, allow_pickle=True)
        tensors = toload[0]

        return cls.from_tensors(tensors[0], tensors[1:-2], tensors[-2])

        # self.uni_l.A = tensors[0]
        # self.uni_l.l[-1] = toload[1]
        # self.uni_l.r[-1] = toload[2]
        # self.uni_l.l_before_CF = self.uni_l.l[-1]
        # self.uni_l.r_before_CF = self.uni_l.r[-1]

        # self.uni_r.A = tensors[-2]
        # self.uni_r.l[-1] = toload[5]
        # self.uni_r.r[-1] = toload[4]
        # self.uni_r.l_before_CF = self.uni_r.l[-1]
        # self.uni_r.r_before_CF = self.uni_r.r[-1]

        # self.grown_left = toload[7][0, 0]
        # self.grown_right = toload[7][0, 1]
        # self.shrunk_left = toload[7][1, 0]
        # self.shrunk_right = toload[7][1, 1]

    def save_state(self, file_name, userdata=None):
        tosave = sp.empty((9), dtype=sp.ndarray)

        Asave = self.A.copy()
        Asave[0] = self.uni_l.A
        Asave[self.N + 1] = self.uni_r.A

        tosave[0] = Asave
        tosave[1] = self.l[0]
        tosave[2] = self.uni_l.r[-1]
        tosave[3] = None
        tosave[4] = self.r[self.N]
        tosave[5] = self.uni_r.l[-1]
        tosave[6] = None
        tosave[7] = sp.array([[self.grown_left, self.grown_right],
                             [self.shrunk_left, self.shrunk_right]])
        tosave[8] = userdata

        sp.save(file_name, tosave)

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

    def update(self, restore_CF=True, normalize=True, auto_truncate=False):
        """Perform all necessary steps needed before taking the next step,
        or calculating expectation values etc., is possible.
        """
        if auto_truncate:
            raise NotImplementedError()

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

    def get_D(self, n):
        if n < 0:
            return self.uni_l.D
        elif n > self.N:
            return self.uni_r.D
        else:
            return self.D[n]

    def get_q(self, n):
        if n < 1:
            return self.uni_l.q
        elif n > self.N:
            return self.uni_r.q
        else:
            return self.q[n]

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

    def overlap(self, other, sanity_checks=False):
        if not self.N == other.N:
            print("States must have same number of non-uniform sites!")
            return

        #if not sp.all(self.D == other.D):
        #    print("States must have same bond-dimensions!")
        #    return

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

        r = mm.mmul(gr, other.uni_r.r[-1], mm.H(gr))
        fac = la.norm(sp.asarray(self.uni_r.r[-1])) / la.norm(r)
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

    def overlap_tangent(self, B, p, top_triv=True, return_contributions=False):
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
        # rs[0] is now the right half including site 1, so the r matrix needed
        # for computations involving site 0.

        ol_left_bulk = mm.adot(vBL, rs[0]) * sp.exp(-1.j * p)

        l = lL
        ols_window = []
        for j in range(1, self.N + 1):
            res_j = sp.exp(1.j * p * j) * mm.adot(
                l, tm.eps_r_noop(rs[j], self.A[j], B))
            ols_window.append(res_j)
            l = tm.eps_l_noop(l, self.A[j], AL)

        if return_contributions:
            return ol_left_bulk, sp.array(ols_window)
        return ol_left_bulk + sum(ols_window)

    def load_state(self, file_name, autogrow=False, do_update=True):
        toload = sp.load(file_name, allow_pickle=True)

        if toload.shape[0] != 9:
            print("Error loading state: Bad data!")
            return

        if autogrow and toload[0].shape[0] != self.A.shape[0]:
            newN = toload[0].shape[0] - 3
            print("Changing N to: %u" % newN)
            self.grow_left(newN - self.N)

        if toload[0].shape != self.A.shape:
            print("Cannot load state: Dimension mismatch!")
            return

        self.A = toload[0]
        self.l[0] = toload[1]
        self.uni_l.r[-1] = toload[2]
        self.r[self.N] = toload[4]
        self.r[self.N + 1] = self.r[self.N]
        self.uni_r.l[-1] = toload[5]

        self.grown_left = toload[7][0, 0]
        self.grown_right = toload[7][0, 1]
        self.shrunk_left = toload[7][1, 0]
        self.shrunk_right = toload[7][1, 1]
        
        self.uni_l.A = self.A[0]
        self.uni_l.l[-1] = self.l[0]
        self.uni_l.l_before_CF = self.uni_l.l[-1]
        self.uni_l.r_before_CF = self.uni_l.r[-1]
        
        self.uni_r.A = self.A[self.N + 1]
        self.uni_r.r[-1] = self.r[self.N]
        self.uni_r.l_before_CF = self.uni_r.l[-1]
        self.uni_r.r_before_CF = self.uni_r.r[-1]

        try:
            if len(self.uni_l.A.shape) == 3:
                self.uni_l.A = [self.uni_l.A]
        except AttributeError:
            pass

        try:
            if len(self.uni_r.A.shape) == 3:
                self.uni_r.A = [self.uni_r.A]
        except AttributeError:
            pass

        self.A[0] = None
        self.A[self.N + 1] = None

        print("loaded.")
        
        if do_update:
            self.update()

        return toload[8]