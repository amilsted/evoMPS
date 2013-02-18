# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 17:29:27 2011

@author: Ashley Milsted

To subclass, or not to subclass?

 Option 1:
     Sandwich contains gen and uni objects that store the actual state.
     Wrappers like getA() bridge the gaps.
     
 Option 2:
     Sandwich is a subclass of generic MPS.
     Need to override stuff.
     
 Option 3:
     Sandwich neither uses nor inherits from gen MPS.
     Duplication of code...
     
 Problem: What to do for TDVP?
     If we inherit from gen MPS, can we still inherit from gen TDVP?
     ...Possibly

"""
import scipy as sp
import scipy.linalg as la
import matmul as mm
import tdvp_common as tm
from mps_gen import EvoMPS_MPS_Generic
import copy


class EvoMPS_MPS_Sandwich(EvoMPS_MPS_Generic):

    def __init__(self, N, uni_ground):
        self.odr = 'C'
        self.typ = sp.complex128
        
        self._sanity_checks = True        
        
        self.eps = sp.finfo(self.typ).eps
        
        self.N = N
        self.D = sp.repeat(uni_ground.D, self.N + 2)
        self.q = sp.repeat(uni_ground.q, self.N + 2)
        
        self.uni_l = copy.deepcopy(uni_ground)
        self.uni_l.symm_gauge = False
        self.uni_l.sanity_checks = self.sanity_checks

        self.uni_r = copy.deepcopy(uni_ground)
        self.uni_r.sanity_checks = self.sanity_checks
        self.uni_r.symm_gauge = False
        
        self.grown_left = 0
        self.grown_right = 0
        self.shrunk_left = 0
        self.shrunk_right = 0

        self.eps = sp.finfo(self.typ).eps

        self._init_arrays()

        for n in xrange(self.N + 2):
            self.A[n][:] = self.uni_l.A

        self.r[self.N] = self.uni_r.r
        self.r[self.N + 1] = self.r[self.N]
        self.l[0] = self.uni_l.l
        

    def _init_arrays(self):
        #Deliberately add a None to the end to catch [-1] indexing!
        self.A = sp.empty((self.N + 3), dtype=sp.ndarray) #Elements 1..N

        self.r = sp.empty((self.N + 3), dtype=sp.ndarray) #Elements 0..N
        self.l = sp.empty((self.N + 3), dtype=sp.ndarray)

        self.l[0] = sp.zeros((self.D[0], self.D[0]), dtype=self.typ, order=self.odr)
        self.r[0] = sp.zeros((self.D[0], self.D[0]), dtype=self.typ, order=self.odr)
        self.A[0] = sp.empty((self.q[0], self.D[0], self.D[0]), dtype=self.typ, order=self.odr)
        for n in xrange(1, self.N + 2):
            self.r[n] = sp.zeros((self.D[n], self.D[n]), dtype=self.typ, order=self.odr)
            self.l[n] = sp.zeros((self.D[n], self.D[n]), dtype=self.typ, order=self.odr)
            self.A[n] = sp.empty((self.q[n], self.D[n-1], self.D[n]), dtype=self.typ, order=self.odr)
    
    @property
    def sanity_checks(self):
        return self._sanity_checks
        
    @sanity_checks.setter
    def sanity_checks(self, value):
        self._sanity_checks = value
        self.uni_l.sanity_checks = value
        self.uni_r.sanity_checks = value
    
    def correct_bond_dimension(self):
        raise NotImplementedError("correct_bond_dimension not implemented in sandwich case")

    def update(self, restore_rcf=True):
        """Perform all necessary steps needed before taking the next step,
        or calculating expectation values etc., is possible.
        
        Return the excess energy.
        """
        if restore_rcf:
            self.restore_RCF()
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
            n_high = self.N + 1
            
        super(EvoMPS_MPS_Sandwich, self).calc_l(n_low=n_low, n_high=n_high)

    def calc_r(self, n_low=-1, n_high=-1):
        """Updates the r matrices using the current state.
        Implements step 5 of the TDVP algorithm or, equivalently, eqn. (41).
        (arXiv:1103.0936v2 [cond-mat.str-el])
        """
        if n_low < 0:
            n_low = 0
        if n_high < 0:
            n_high = self.N - 1
        
        super(EvoMPS_MPS_Sandwich, self).calc_r(n_low=n_low, n_high=n_high)
        
    def simple_renorm(self, update_r=True):
        raise NotImplementedError("simple_renorm not implemented in sandwich case")

    def restore_RCF_dbg(self):
        for n in xrange(self.N + 2):
            print (n, self.l[n].trace().real, self.r[n].trace().real,
                   mm.adot(self.l[n], self.r[n]).real)

        norm_r = mm.adot(self.uni_r.l, self.r[self.N])
        norm_l = mm.adot(self.l[0], self.uni_l.r)
        print norm_r
        print norm_l

        #r_m1 = self.eps_r(0, self.r[0])
        #print mm.adot(self.l[0], r_m1).real

        norm = mm.adot(self.l[0], self.r[0]).real

        h = sp.empty((self.N + 1), dtype=self.typ)
        for n in xrange(self.N + 1):
            h[n] = self.expect_2s(self.h_nn, n)
        h *= 1/norm

        self.uni_l.A = self.A[0].copy()
        self.uni_l.l = self.l[0].copy()
        self.uni_l.calc_AA()
        h_left = self.uni_l.expect_2s(self.uni_l.h_nn) / norm_l

        self.uni_r.A = self.A[self.N + 1].copy()
        self.uni_r.r = self.r[self.N].copy()
        self.uni_r.calc_AA()
        h_right = self.uni_l.expect_2s(self.uni_r.h_nn) / norm_r

        return h, h_left, h_right

    def restore_RCF_r(self):
        G_n_i = None
        for n in reversed(xrange(1, self.N + 2)):
            self.r[n - 1], G_n_i, G_n = tm.restore_RCF_r(self.A[n], self.r[n], 
                                             G_n_i, sanity_checks=self.sanity_checks)

        #self.r[self.N + 1] = self.r[self.N]

        #Now G_n_i contains g_0_i
        for s in xrange(self.q[0]): #Note: This does not change the scale of A[0]
            self.A[0][s] = mm.mmul(G_n, self.A[0][s], G_n_i)

        self.uni_l.r = mm.mmul(G_n, self.uni_l.r, mm.H(G_n))
        self.l[0] = mm.mmul(mm.H(G_n_i), self.l[0], G_n_i)

    def restore_RCF_l(self):
        G_nm1 = None
        l_nm1 = self.l[0]
        for n in xrange(self.N + 1):
            self.l[n], G_nm1, G_nm1_i = tm.restore_RCF_l(self.A[n], l_nm1, 
                                      G_nm1, sanity_checks=self.sanity_checks)
            
            if n == 0:
                self.uni_l.r = mm.mmul(G_nm1, self.uni_l.r, G_nm1_i) #since r is not eye

            l_nm1 = self.l[n]

        #Now G_nm1 = G_N
        for s in xrange(self.q[self.N + 1]):
            self.A[self.N + 1][s] = mm.mmul(G_nm1, self.A[self.N + 1][s], G_nm1_i)

        ##This should not be necessary if G_N is really unitary
        #self.r[self.N] = mm.mmul(G_nm1, self.r[self.N], mm.H(G_nm1))
        #self.r[self.N + 1] = self.r[self.N]
        self.uni_r.l = mm.mmul(mm.H(G_nm1_i), self.uni_r.l, G_nm1_i)
        
        self.S_hc = sp.zeros((self.N), dtype=sp.complex128)
        for n in xrange(1, self.N + 1):
            self.S_hc[n-1] = -sp.sum(self.l[n].diag * sp.log2(self.l[n].diag))

    def restore_RCF(self, dbg=False):
        if dbg:
            self.calc_l()
            self.calc_r()
            print "BEFORE..."
            h_before, h_left_before, h_right_before = self.restore_RCF_dbg()

            print (h_left_before, h_before, h_right_before)

        self.restore_RCF_r()

        if dbg:
            self.calc_l()
            print "MIDDLE..."
            h_mid, h_left_mid, h_right_mid = self.restore_RCF_dbg()

            print (h_left_mid, h_mid, h_right_mid)

        fac = 1 / self.l[0].trace().real
        if dbg:
            print "Scale l[0]: %g" % fac
        self.l[0] *= fac
        self.uni_l.r *= 1/fac

        self.restore_RCF_l()

        if dbg:
            print "Uni left:"
        self.uni_l.A = self.A[0]
        self.uni_l.l = self.l[0]
        self.uni_l.calc_lr() #Ensures largest ev of E=1
        self.l[0] = self.uni_l.l #No longer diagonal!
        self.A[0] = self.uni_l.A
        if self.sanity_checks:
            if not sp.allclose(self.l[0], sp.diag(sp.diag(self.l[0])), atol=1E-12, rtol=1E-12):
                print "Sanity Fail in restore_RCF!: True l[0] not diagonal!"
                print la.norm(self.l[0] - sp.diag(sp.diag(self.l[0])))
        self.l[0] = mm.simple_diag_matrix(sp.diag(self.l[0]))

        fac = 1 / sp.trace(self.l[0]).real
        if dbg:
            print "Scale l[0]: %g" % fac
        self.l[0] *= fac
        self.uni_l.r *= 1/fac

        self.uni_l.l = self.l[0]

        if dbg:
            print "Uni right:"
        self.uni_r.A = self.A[self.N + 1]
        self.uni_r.r = self.r[self.N]
        self.uni_r.calc_lr() #Ensures largest ev of E=1
        self.r[self.N] = self.uni_r.r
        self.A[self.N + 1] = self.uni_r.A
        if self.sanity_checks:
            if not sp.allclose(self.r[self.N], sp.eye(self.D[self.N]), atol=1E-12, rtol=1E-12):
                print "Sanity Fail in restore_RCF!: True r[N] not eye!"
        self.r[self.N] = mm.eyemat(self.D[self.N], dtype=self.typ)
        self.uni_r.r = self.r[self.N]
        self.r[self.N + 1] = self.r[self.N]

        self.l[self.N + 1][:] = tm.eps_l_noop(self.l[self.N], 
                                              self.A[self.N + 1], 
                                              self.A[self.N + 1])

        if self.sanity_checks:
            l_n = self.l[0]
            for n in xrange(0, self.N + 1):
                l_n = tm.eps_l_noop(l_n, self.A[n], self.A[n])
                if not sp.allclose(l_n, self.l[n], atol=1E-12, rtol=1E-12):
                    print "Sanity Fail in restore_RCF!: l_%u is bad" % n

            r_nm1 = self.r[self.N + 1]
            for n in reversed(xrange(1, self.N + 2)):
                r_nm1 = tm.eps_r_noop(r_nm1, self.A[n], self.A[n])
                if not sp.allclose(r_nm1, self.r[n - 1], atol=1E-12, rtol=1E-12):
                    print "Sanity Fail in restore_RCF!: r_%u is bad" % (n - 1)
        if dbg:
            print "AFTER..."
            h_after, h_left_after, h_right_after = self.restore_RCF_dbg()

            print (h_left_after, h_after, h_right_after)

            print h_after - h_before

            print (h_after.sum() - h_before.sum() + h_left_after - h_left_before
                   + h_right_after - h_right_before)
    
    def restore_RCF_bulk_only(self):
        self.uni_l.calc_lr()
        g, gi = self.uni_l.restore_CF(ret_g=True)

        m = self.uni_l.A.mean()
        print sp.sqrt(sp.conj(m) / m)
        self.uni_l.A *= sp.sqrt(sp.conj(m) / m)
        
        self.uni_l.calc_lr()
        
        self.A[0] = self.uni_l.A
        for s in xrange(self.A[1].shape[0]):
            self.A[1][s] = gi.dot(self.A[1][s])
        self.l[0] = self.uni_l.l
        
        self.uni_r.calc_lr()
        g, gi = self.uni_r.restore_CF(ret_g=True)
        
        m = self.uni_r.A.mean()
        print sp.sqrt(sp.conj(m) / m)
        self.uni_r.A *= sp.sqrt(sp.conj(m) / m)
        
        self.uni_r.calc_lr()
        
        self.A[self.N + 1] = self.uni_r.A
        for s in xrange(self.A[1].shape[0]):
            self.A[self.N][s] = self.A[self.N][s].dot(g)
        self.r[self.N] = self.uni_r.r
        self.r[self.N + 1] = self.r[self.N]
        
        self.calc_r()
        
        norm = mm.adot(self.l[0], self.r[0])
        
        self.A[1] *= 1 / sp.sqrt(norm)
        
        self.calc_l()
        self.calc_r()
        
    def check_RCF(self):
        raise NotImplementedError("check_RCF not implemented in sandwich case")
    
    def grow_left(self, m):
        """Grow the generic region to the left by m sites.
        """
        oldA = self.A
        oldl_0 = self.l[0]
        oldr_N = self.r[self.N]

        self.N = self.N + m
        self._init_arrays()

        for n in xrange(m + 1):
            self.A[n][:] = oldA[0]

        for n in xrange(m + 1, self.N + 2):
            self.A[n][:] = oldA[n - m]

        self.l[0] = oldl_0
        self.r[self.N] = oldr_N
        self.r[self.N + 1] = self.r[self.N]
        
        self.grown_left += m


    def grow_right(self, m):
        """Grow the generic region to the right by m sites.
        """
        oldA = self.A
        oldl_0 = self.l[0]
        oldr_N = self.r[self.N]

        self.N = self.N + m
        self._init_arrays()

        for n in xrange(self.N - m + 1):
            self.A[n][:] = oldA[n]

        for n in xrange(self.N - m + 1, self.N + 2):
            self.A[n][:] = oldA[self.N - m + 1]

        self.l[0] = oldl_0
        self.r[self.N] = oldr_N
        self.r[self.N + 1] = self.r[self.N]
        
        self.grown_right += m
        
    
    def shrink_left(self, m):
        """Experimental shrinking of the state.
        Generic-section matrices are simply replaced
        with copies of the uniform matrices. Usually a bad idea!
        """
        oldA = self.A
        oldl_0 = self.l[0]
        oldr_N = self.r[self.N]

        self.N = self.N - m
        self._init_arrays()

        for n in xrange(self.N + 2):
            self.A[n][:] = oldA[n + m]
            
        self.A[0][:] = oldA[0]

        self.l[0] = oldl_0
        self.r[self.N] = oldr_N
        self.r[self.N + 1] = self.r[self.N]
        
        self.shrunk_left += m
        
    def shrink_right(self, m):
        """See shrink_left().
        """
        oldA = self.A
        oldl_0 = self.l[0]
        oldr_N = self.r[self.N]

        self.N = self.N - m
        self._init_arrays()

        for n in xrange(self.N + 1):
            self.A[n][:] = oldA[n]
            
        for n in xrange(self.N + 1, self.N + 2):
            self.A[n][:] = oldA[n + m]

        self.l[0] = oldl_0
        self.r[self.N] = oldr_N
        self.r[self.N + 1] = self.r[self.N]
        
        self.shrunk_right += m
    
    def get_A(self, n):
        if n < 0:
            return self.A[0]
        elif n > self.N + 1:
            return self.A[self.N + 1]
        else:
            return self.A[n]
    
    def get_l(self, n):
        """Gets an l[n], even if n < 0 or n > N + 1
        """
        if 0 <= n <= self.N + 1:
            return self.l[n]
        elif n < 0:
            return self.l[0]
        else:
            l_m = self.l[self.N + 1]
            for m in xrange(self.N + 2, n + 1):
                l_m = tm.eps_l_noop(l_m, self.A[self.N + 1], self.A[self.N + 1])
            return l_m

    def get_r(self, n, r_np1=None):
        """Gets an r[n], even if n < 0 or n > N + 1
        """
        if 0 <= n <= self.N + 1:
            return self.r[n]
        elif n > self.N + 1:
            return self.r[self.N + 1]
        else:
            if r_np1 is None:
                r_m = self.r[0]
                for m in reversed(xrange(n, 0)):
                    r_m = tm.eps_r_noop(r_m, self.A[0], self.A[0])
                return r_m
            else:
                return tm.eps_r_noop(r_np1, self.A[0], self.A[0])

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
        o1 : function
            The first operator, acting on the first site.
        o2 : function
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

        for n in reversed(xrange(n1 + 1, n2)):
            r_n = tm.eps_r_noop(r_n, self.get_A(n), self.get_A(n))

        r_n = tm.eps_r_op_1s(r_n, A1, A1, op1)

        return mm.adot(self.get_l(n1 - 1), r_n)

    def density_2s(self, n1, n2):
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
        ln1m1 = self.get_l(n1 - 1)

        for s2 in xrange(self.q[n2]):
            for t2 in xrange(self.q[n2]):
                r_n2 = mm.mmul(self.A[n2][t2], self.r[n2], mm.H(self.A[n2][s2]))

                r_n = r_n2
                for n in reversed(xrange(n1 + 1, n2)):
                    r_n = tm.eps_r_noop(r_n, self.A[n], self.A[n])

                for s1 in xrange(self.q[n1]):
                    for t1 in xrange(self.q[n1]):
                        r_n1 = mm.mmul(self.A[n1][t1], r_n, mm.H(self.A[n1][s1]))
                        tmp = mm.adot(ln1m1, r_n1)
                        rho[s1 * self.q[n1] + s2, t1 * self.q[n1] + t2] = tmp
        return rho

    def apply_op_1s(self, op, n):
        """Applies a one-site operator o to site n.
        
        Can be used to create excitations.
        """
        
        if not (0 < n <= self.N):
            raise ValueError("Operators can only be applied to sites 1 to N!")
        
        newA = sp.zeros_like(self.A[n])
        
        if callable(op):
            op = sp.vectorize(op, otypes=[sp.complex128])
            op = sp.fromfunction(op, (self.q[n], self.q[n]))

        for s in xrange(self.q[n]):
            for t in xrange(self.q[n]):
                newA[s] += self.A[n][t] * op[s, t]

        self.A[n] = newA
        
    def overlap(self, other, sanity_checks=False):
        if not self.N == other.N:
            print "States must have same number of non-uniform sites!"
            return
            
        if not sp.all(self.D == other.D):
            print "States must have same bond-dimensions!"
            return
            
        if not sp.all(self.q == other.q):
            print "States must have same Hilbert-space dimensions!"
            return
        
        dL, phiL, gLl = self.uni_l.fidelity_per_site(other.u_gnd_l, full_output=True, left=True)
        
        dR, phiR, gRr = self.uni_r.fidelity_per_site(other.u_gnd_r, full_output=True, left=False)
        
        gr = mm.H(la.inv(gRr).dot(sp.asarray(self.uni_r.r)))
        gri = mm.H(la.inv(sp.asarray(self.uni_r.r)).dot(gRr))
        
        if sanity_checks:
            AR = other.u_gnd_r.A.copy()        
            for s in xrange(AR.shape[0]):
                AR[s] = gr.dot(AR[s]).dot(gri)
                
            print la.norm(AR - self.uni_r.A)
        
        r = gr.dot(sp.asarray(other.u_gnd_r.r)).dot(mm.H(gr))
        fac = la.norm(sp.asarray(self.uni_r.r)) / la.norm(r)        
        gr *= sp.sqrt(fac)
        gri /= sp.sqrt(fac)
        
        if sanity_checks:
            r *= fac
            print la.norm(r - self.uni_r.r)

            AN = other.A[self.N].copy()
            for s in xrange(AN.shape[0]):
                AN[s] = AN[s].dot(gri)
            r = tm.eps_r_noop(self.uni_r.r, AN, AN)
            print la.norm(r - other.r[self.N - 1])

        gl = la.inv(sp.asarray(self.uni_l.l)).dot(gLl)
        gli = la.inv(gLl).dot(sp.asarray(self.uni_l.l))
        
        l = mm.H(gli).dot(sp.asarray(other.u_gnd_l.l)).dot(gli)
        fac = la.norm(sp.asarray(self.uni_l.l)) / la.norm(l)

        gli *= sp.sqrt(fac)
        gl /= sp.sqrt(fac)
        
        if sanity_checks:
            l *= fac
            print la.norm(l - self.uni_l.l)
        
            l = mm.H(gli).dot(sp.asarray(other.u_gnd_l.l)).dot(gli)
            print la.norm(l - self.uni_l.l)
        
        #print (dL, dR, phiL, phiR)

        if not abs(dL - 1) < 1E-12:
            print "Left bulk states do not match!"
            return 0
            
        if not abs(dR - 1) < 1E-12:
            print "Right bulk states do not match!"
            return 0
                    
        AN = other.A[self.N].copy()
        for s in xrange(AN.shape[0]):
            AN[s] = AN[s].dot(gri)
            
        A1 = other.A[1].copy()
        for s in xrange(A1.shape[0]):
            A1[s] = gl.dot(A1[s])
        
        r = tm.eps_r_noop(self.uni_r.r, self.A[self.N], AN)
        for n in xrange(self.N - 1, 1, -1):
            r = tm.eps_r_noop(r, self.A[n], other.A[n])
        r = tm.eps_r_noop(r, self.A[1], A1)
    
        return mm.adot(self.uni_l.l, r)

    def save_state(self, file):
        raise NotImplementedError("save_state not implemented in sandwich case")

        
    def load_state(self, file):
        raise NotImplementedError("load_state not implemented in sandwich case")
