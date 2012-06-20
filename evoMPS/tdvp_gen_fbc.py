# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 17:29:27 2011

@author: Ashley Milsted

Issues:
    - Getting stuck: Especially transverse Ising. More noise -> higher "residual energy"
      - Tends to involve a problem at the left boundary
      - Expanding to the left allows the 'excitation' to dissipate
      - Always on the left! Due to choice of canonical form? But how?
        - Is the Restore_RCF GT pushing info to the left after all?
        - Or is this a result of the right gauge-fixing choice?

"""
import scipy as sp
import scipy.linalg as la
import nullspace as ns
import matmul as mm
import tdvp_uniform as uni

def go(sfbc, tau, steps, dbg=False, force_calc_lr=False, RK4=False,
       op=None, op_every=5, autogrow=False):
    """A simple integration loop for testing"""
    h_prev = 0
    sfbc.restore_RCF(dbg=dbg)
    if force_calc_lr:
        sfbc.calc_l()
        sfbc.calc_r()

    data = []

    for i in xrange(steps):
        if RK4:
            h, eta = sfbc.take_step_RK4(tau)
        else:
            h, eta = sfbc.take_step(tau)

        etas = sfbc.eta[1:].copy()
        
        #Basic dynamic expansion:
        if autogrow and etas[0] > sfbc.eta_uni * 10:
            sfbc.Grow_left(2)
            for row in data:
                row.insert(0, 0)
                row.insert(0, 0)

        if autogrow and etas[-1] > sfbc.eta_uni * 10:
            sfbc.Grow_right(2)
            for row in data:
                row.append(0)
                row.append(0)

        sfbc.restore_RCF(dbg=dbg)
        if force_calc_lr:
            sfbc.calc_l()
            sfbc.calc_r()

        norm_uni = mm.adot(sfbc.u_gnd_r.l, sfbc.u_gnd_r.r).real
        h_uni = sfbc.u_gnd_r.h.real / norm_uni
        print "\t".join(map(str, (eta.real, h.real - h_uni, (h - h_prev).real)))
        print etas.real
#        if i > 0 and (h - h_prev).real > 0:
#            break
        h_prev = h

        if (not op is None) and (i % op_every == 0):
            op_range = range(-10, sfbc.N + 10)
            data.append(map(lambda n: sfbc.expect_1S(op, n).real, op_range))

    return sp.array(data)

class EvoMPS_TDVP_GenFBC:
    odr = 'C'
    typ = sp.complex128

    #Numsites
    N = 0

    D = None
    q = None

    h_nn = None
    h_ext = None

    eps = 0

    sanity_checks = False

    u_gnd_l = None
    u_gnd_r = None

    def trim_D(self):
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
        self.D = sp.repeat(self.u_gnd_l.D, self.N + 2)
        self.q = sp.repeat(self.u_gnd_l.q, self.N + 2)

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
        self.D[0] = self.u_gnd_l.D
        self.D[self.N + 1] = self.u_gnd_l.D

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

    def __init__(self, numsites, uni_ground):
        self.u_gnd_l = uni.evoMPS_TDVP_Uniform(uni_ground.D, uni_ground.q)
        self.u_gnd_l.sanity_checks = self.sanity_checks
        self.u_gnd_l.h_nn = uni_ground.h_nn
        self.u_gnd_l.h_nn_cptr = uni_ground.h_nn_cptr
        self.u_gnd_l.A = uni_ground.A.copy()
        self.u_gnd_l.l = uni_ground.l.copy()
        self.u_gnd_l.r = uni_ground.r.copy()

        self.u_gnd_l.symm_gauge = False
        self.u_gnd_l.Update()
        self.u_gnd_l.Calc_lr()
        self.u_gnd_l.Calc_B()
        self.eta_uni = self.u_gnd_l.eta

        self.u_gnd_r = uni.evoMPS_TDVP_Uniform(uni_ground.D, uni_ground.q)
        self.u_gnd_r.sanity_checks = self.sanity_checks
        self.u_gnd_r.symm_gauge = False
        self.u_gnd_r.h_nn = uni_ground.h_nn
        self.u_gnd_r.h_nn_cptr = uni_ground.h_nn_cptr
        self.u_gnd_r.A = self.u_gnd_l.A.copy()
        self.u_gnd_r.l = self.u_gnd_l.l.copy()
        self.u_gnd_r.r = self.u_gnd_l.r.copy()

        self.h_nn = self.wrap_h

        self.eps = sp.finfo(self.typ).eps

        self.N = numsites

        self._init_arrays()

        for n in xrange(self.N + 2):
            self.A[n][:] = self.u_gnd_l.A

        self.r[self.N] = self.u_gnd_r.r
        self.r[self.N + 1] = self.r[self.N]
        self.l[0] = self.u_gnd_l.l

    def randomize(self):
        """Set A's randomly, trying to keep the norm reasonable.

        We need to ensure that the M matrix in eps_r() is positive definite. How?
        Does it really have to be?
        """
        for n in xrange(1, self.N + 1):
            self.A[n].real = (sp.rand(self.D[n - 1], self.D[n]) - 0.5) / sp.sqrt(self.q[n]) #/ sp.sqrt(self.N) #/ sp.sqrt(self.D[n])
            self.A[n].imag = 0#(sp.rand(self.D[n - 1], self.D[n]) - 0.5) / sp.sqrt(self.q[n])

    def wrap_h(self, n, s, t, u, v):
        return self.u_gnd_l.h_nn(s, t, u, v)

    def calc_C(self, n_low=-1, n_high=-1):
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
            for u in xrange(self.q[n]):
                for v in xrange(self.q[n + 1]):
                    AA = mm.mmul(self.A[n][u], self.A[n + 1][v]) #only do this once for each
                    for s in xrange(self.q[n]):
                        for t in xrange(self.q[n + 1]):
                            h_nn_stuv = self.h_nn(n, s, t, u, v)
                            if h_nn_stuv != 0:
                                self.C[n][s, t] += h_nn_stuv * AA

    def calc_K(self, n_low=-1, n_high=-1):
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

            for s in xrange(self.q[n]):
                for t in xrange(self.q[n+1]):
                    self.K[n] += mm.mmul(self.C[n][s, t], self.r[n + 1],
                                           mm.H(self.A[n + 1][t]),
                                           mm.H(self.A[n][s]))

                self.K[n] += mm.mmul(self.A[n][s], self.K[n + 1],
                                       mm.H(self.A[n][s]))

    def calc_K_luni(self, C, h_l):
        K_np1 = self.K[0]
        K_n = sp.empty_like(K_np1)
        r_np1 = None

        print (mm.adot(self.l[0], K_np1) / (self.N + 1)).real

        for n in reversed(xrange(-10000, 0)):
            K_n.fill(0)

            if not r_np1 is None and sp.allclose(r_np1/la.norm(r_np1),
                                                 self.u_gnd_l.r/la.norm(self.u_gnd_l.r),
                                                 atol=1E-10, rtol=1E-10):
                break
            else:
                r_np1 = self.get_r(n + 1, r_np1)

            for s in xrange(self.q[0]):
                for t in xrange(self.q[0]):
                    K_n += mm.mmul(C[s, t], r_np1, mm.H(self.A[0][t]),
                                     mm.H(self.A[0][s]))

                K_n += mm.mmul(self.A[0][s], K_np1, mm.H(self.A[0][s]))

            h = (mm.adot(self.l[0], K_n) - h_l * abs(n)) / (self.N + 1.0)
            #Note: This becomes ever more inaccurate due to numerical error!!

            h_is_stable = sp.allclose(mm.adot(self.l[0], K_n), mm.adot(self.l[0], K_np1) + h_l,
                                      atol=1E-14, rtol=1E-14)

            if n < -1:
                print (n, h.real, mm.adot(self.l[0], K_n).real,
                       (mm.adot(self.l[0], K_np1) + h_l).real, h_is_stable,
                       la.norm(K_n/la.norm(K_n) - K_np1/la.norm(K_np1)),
                       la.norm(r_np1/la.norm(r_np1) - self.u_gnd_l.r/la.norm(self.u_gnd_l.r)))
                if h_is_stable:
                    break

            K_np1 = K_n.copy()

        return h

    def calc_Vsh(self, n, sqrt_r):
        """Generates mm.H(V[n][s]) for a given n, used for generating B[n][s]

        This is described on p. 14 of arXiv:1103.0936v2 [cond-mat.str-el] for left
        gauge fixing. Here, we are using right gauge fixing.

        Array slicing and reshaping is used to manipulate the indices as necessary.

        Each V[n] directly depends only on A[n] and r[n].

        We return the conjugate mm.H(V) because we use it in more places than V.
        """
        R = sp.zeros((self.D[n], self.q[n], self.D[n-1]), dtype=self.typ, order='C')

        for s in xrange(self.q[n]):
            R[:,s,:] = mm.mmul(sqrt_r, mm.H(self.A[n][s]))

        R = R.reshape((self.q[n] * self.D[n], self.D[n-1]))
        V = mm.H(ns.nullspace(mm.H(R)))

        if self.sanity_checks:
            if not sp.allclose(mm.mmul(V, R), 0):
                print "Sanity Fail in calc_Vsh!: VR_%u != 0" % (n)
            if not sp.allclose(mm.mmul(V, mm.H(V)), sp.eye(V.shape[0])):
                print "Sanity Fail in calc_Vsh!: V H(V)_%u != eye" % (n)
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
                M += mm.mmul(mm.H(Vsh[s]), sqrt_r, mm.H(self.A[n][s]))
            if not sp.allclose(M, 0):
                print "Sanity Fail in calc_Vsh!: Bad Vsh_%u" % (n)

        return Vsh

    def calc_opt_x(self, n, Vsh, sqrt_l, sqrt_r, sqrt_l_inv, sqrt_r_inv):
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

        x_part.fill(0)
        for s in xrange(self.q[n]):
            x_subpart.fill(0)

            if n < self.N + 1:
                x_subsubpart.fill(0)
                for t in xrange(self.q[n + 1]):
                    x_subsubpart += mm.mmul(self.C[n][s,t], self.r[n + 1], mm.H(self.A[n + 1][t])) #~1st line

                x_subsubpart += mm.mmul(self.A[n][s], self.K[n + 1]) #~3rd line

                x_subpart += mm.mmul(x_subsubpart, sqrt_r_inv)

            if not self.h_ext is None:
                x_subsubpart.fill(0)
                for t in xrange(self.q[n]):                         #Extra term to take care of h_ext..
                    x_subsubpart += self.h_ext(n, s, t) * self.A[n][t] #it may be more effecient to squeeze this into the nn term...
                x_subpart += mm.mmul(x_subsubpart, sqrt_r)

            x_part += mm.mmul(x_subpart, Vsh[s])

        x += mm.mmul(sqrt_l, x_part)

        if n > 0:
            l_nm2 = self.get_l(n - 2)
            x_part.fill(0)
            for s in xrange(self.q[n]):     #~2nd line
                x_subsubpart.fill(0)
                for t in xrange(self.q[n + 1]):
                    x_subsubpart += mm.mmul(mm.H(self.A[n - 1][t]), l_nm2, self.C[n - 1][t, s])
                x_part += mm.mmul(x_subsubpart, sqrt_r, Vsh[s])
            x += mm.mmul(sqrt_l_inv, x_part)

        return x

    def calc_B(self, n):
        """Generates the B[n] tangent vector corresponding to physical evolution of the state.

        In other words, this returns B[n][x*] (equiv. eqn. (47) of
        arXiv:1103.0936v2 [cond-mat.str-el])
        with x* the parameter matrices satisfying the Euler-Lagrange equations
        as closely as possible.
        """
        if self.q[n] * self.D[n] - self.D[n - 1] > 0:
            l_sqrt, r_sqrt, l_sqrt_inv, r_sqrt_inv = self.calc_l_r_roots(n)

            Vsh = self.calc_Vsh(n, r_sqrt)

            x = self.calc_opt_x(n, Vsh, l_sqrt, r_sqrt, l_sqrt_inv, r_sqrt_inv)

            self.eta[n] = sp.sqrt(mm.adot(x, x))

            B = sp.empty_like(self.A[n])
            for s in xrange(self.q[n]):
                B[s] = mm.mmul(l_sqrt_inv, x, mm.H(Vsh[s]), r_sqrt_inv)

            if self.sanity_checks:
                M = sp.zeros_like(self.r[n - 1])
                for s in xrange(self.q[n]):
                    M += mm.mmul(B[s], self.r[n], mm.H(self.A[n][s]))

                if not sp.allclose(M, 0):
                    print "Sanity Fail in calc_B!: B_%u does not satisfy GFC!" % n

            #print "eta_%u = %g" % (n, eta.real)

            return B
        else:
            return None, 0

    def calc_l_r_roots(self, n):
        """Returns the matrix square roots (and inverses) needed to calculate B.

        Hermiticity of l[n] and r[n] is used to speed this up.
        If an exception occurs here, it is probably because these matrices
        are no longer Hermitian (enough).
        """
        try:
            l_sqrt = self.l[n - 1].sqrt()
        except AttributeError:
            l_sqrt, evd = mm.sqrtmh(self.l[n - 1], ret_evd=True)

        try:
            l_sqrt_inv = l_sqrt.inv()
        except AttributeError:
            l_sqrt_inv = mm.invmh(l_sqrt, evd=evd)

        try:
            r_sqrt = self.r[n].sqrt()
        except AttributeError:
            r_sqrt, evd =  mm.sqrtmh(self.r[n], ret_evd=True)

        try:
            r_sqrt_inv = r_sqrt.inv()
        except AttributeError:
            r_sqrt_inv = mm.invmh(r_sqrt, evd=evd)

        if self.sanity_checks:
            if not sp.allclose(mm.mmul(l_sqrt, l_sqrt), self.l[n - 1]):
                print "Sanity Fail in calc_l_r_roots: Bad l_sqrt_%u" % (n - 1)
            if not sp.allclose(mm.mmul(r_sqrt, r_sqrt), self.r[n]):
                print "Sanity Fail in calc_l_r_roots: Bad r_sqrt_%u" % (n)
            if not sp.allclose(mm.mmul(l_sqrt, l_sqrt_inv), sp.eye(l_sqrt.shape[0])):
                print "Sanity Fail in calc_l_r_roots: Bad l_sqrt_inv_%u" % (n - 1)
            if not sp.allclose(mm.mmul(r_sqrt, r_sqrt_inv), sp.eye(r_sqrt.shape[0])):
                print "Sanity Fail in calc_l_r_roots: Bad r_sqrt_inv_%u" % (n)

        return l_sqrt, r_sqrt, l_sqrt_inv, r_sqrt_inv

    def update(self):
        self.calc_C()

        self.u_gnd_r.Calc_AA()
        self.u_gnd_r.Calc_C()
        self.u_gnd_r.Calc_K()
        self.K[self.N + 1][:] = self.u_gnd_r.K

        self.calc_K()

        self.u_gnd_l.Calc_AA()
        self.u_gnd_l.Calc_C()
        K_left, h_left_uni = self.u_gnd_l.Calc_K_left()

        h = (mm.adot(K_left, self.r[0]) + mm.adot(self.l[0], self.K[0])) / (self.N + 1)

        return h

    def take_step(self, dtau): #simple, forward Euler integration
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
        h = self.update()

        eta_tot = 0

        B_prev = None
        for n in xrange(1, self.N + 2):
            if n <= self.N:
                B = self.calc_B(n)
                eta_tot += self.eta[n]

            #V is not always defined
            if n > 1 and not B_prev is None:
                self.A[n - 1] += -dtau * B_prev

            B_prev = B

        return h, eta_tot


    def take_step_RK4(self, dtau):
        """Take a step using the fourth-order explicit Runge-Kutta method.

        This requires more memory than a simple forward Euler step, and also
        more than a backward Euler step. It is, however, far more accurate
        and stable than forward Euler, and much faster than the backward
        Euler method, since there is no need to iteratively solve an implicit
        equation.
        """
        #self.Restore_RCF()

        h = self.update()

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
                B_fin[n] = B

            if not B_prev is None:
                self.A[n - 1] = A0[n - 1] - dtau/2 * B_prev

            B_prev = B

        #self.Restore_RCF()
        #self.Update()
        self.calc_l()
        self.calc_r()
        self.calc_C()
        self.calc_K()

        B_prev = None
        for n in xrange(1, self.N + 2):
            if n <= self.N:
                B = self.calc_B(n) #k2

            if not B_prev is None:
                self.A[n - 1] = A0[n - 1] - dtau/2 * B_prev
                B_fin[n - 1] += 2 * B_prev

            B_prev = B

        #self.Restore_RCF()
        #self.Update()
        self.calc_l()
        self.calc_r()
        self.calc_C()
        self.calc_K()

        B_prev = None
        for n in xrange(1, self.N + 2):
            if n <= self.N:
                B = self.calc_B(n) #k3

            if not B_prev is None:
                self.A[n - 1] = A0[n - 1] - dtau * B_prev
                B_fin[n - 1] += 2 * B_prev

            B_prev = B

        #self.Restore_RCF()
        #self.Update()
        self.calc_l()
        self.calc_r()
        self.calc_C()
        self.calc_K()

        for n in xrange(1, self.N + 1):
            B = self.calc_B(n) #k4
            if not B is None:
                B_fin[n] += B

        for n in xrange(1, self.N + 1):
            if not B_fin[n] is None:
                self.A[n] = A0[n] - dtau /6 * B_fin[n]

        return h, eta_tot

    def add_noise(self, fac, n_i=-1, n_f=-1):
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


    def calc_l(self, start=-1, finish=-1):
        """Updates the l matrices using the current state.
        Implements step 5 of the TDVP algorithm or, equivalently, eqn. (41).
        (arXiv:1103.0936v2 [cond-mat.str-el])
        """
        if start < 0:
            start = 1
        if finish < 0:
            finish = self.N + 1
        for n in xrange(start, finish + 1):
            try:
                self.l[n] = self.l[n].A
            except AttributeError:
                pass
            self.l[n].fill(0)

            for s in xrange(self.q[n]):
                self.l[n] += mm.mmul(mm.H(self.A[n][s]), self.l[n - 1], self.A[n][s])

    def calc_r(self, n_low=-1, n_high=-1):
        """Updates the r matrices using the current state.
        Implements step 5 of the TDVP algorithm or, equivalently, eqn. (41).
        (arXiv:1103.0936v2 [cond-mat.str-el])
        """
        if n_low < 0:
            n_low = 0
        if n_high < 0:
            n_high = self.N - 1
        for n in reversed(xrange(n_low, n_high + 1)):
            self.r[n] = sp.asarray(self.r[n])
            self.eps_r(self.r[n], n + 1, self.r[n + 1], None)

    def eps_r(self, res, n, x, o):
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

        if o is None:
            for s in xrange(self.q[n]):
                res += mm.mmul(self.A[n][s], x, mm.H(self.A[n][s]))
        else:
            for s in xrange(self.q[n]):
                for t in xrange(self.q[n]):
                    o_st = o(n, s, t)
                    if o_st != 0.:
                        tmp = mm.mmul(self.A[n][t], x, mm.H(self.A[n][s]))
                        tmp *= o_st
                        res += tmp
        return res

    def eps_r_2s(self, n, x, op, A1=None, A2=None, A3=None, A4=None):
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

        for u in xrange(self.q[n]):
            for v in xrange(self.q[m]):
                AAuvH = mm.mmul(A3[u], A4[v])
                AAuvH = mm.H(AAuvH, out=AAuvH)
                subres = sp.zeros_like(A1[0])
                for s in xrange(self.q[n]):
                    for t in xrange(self.q[m]):
                        opval = op(n, u, v, s, t)
                        if opval != 0:
                            subres += opval * sp.dot(A1[s], A2[t])
                res += mm.mmul(subres, x, AAuvH)

        return res

    def eps_l(self, res, n, x):
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

        for s in xrange(self.q[n]):
            res += mm.mmul(mm.H(self.A[n][s]), x, self.A[n][s])
        return res

    def restore_ONR_n(self, n, G_n_i):
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
            GGh_n_i = mm.mmul(G_n_i, self.r[n], mm.H(G_n_i))

        M = self.eps_r(None, n, GGh_n_i, None)

        #The following should be more efficient than eigh():
        try:
            tu = la.cholesky(M) #Assumes M is pos. def.. It should raise LinAlgError if not.
            G_nm1 = mm.H(mm.invtr(tu)) #G is now lower-triangular
            G_nm1_i = mm.H(tu)
        except sp.linalg.LinAlgError:
            print "Restore_ON_R_%u: Falling back to eigh()!" % n
            e,Gh = la.eigh(M)
            G_nm1 = mm.H(mm.mmul(Gh, sp.diag(1/sp.sqrt(e) + 0.j)))
            G_nm1_i = la.inv(G_nm1)

        if G_n_i is None:
            G_n_i = G_nm1_i

        if self.sanity_checks:
            if not sp.allclose(sp.dot(G_nm1, G_nm1_i), sp.eye(G_nm1.shape[0]), atol=1E-13, rtol=1E-13):
                print "Sanity Fail in restore_ONR_%u!: Bad GT at n=%u" % (n, n)

        for s in xrange(self.q[n]):
            self.A[n][s] = mm.mmul(G_nm1, self.A[n][s], G_n_i)

        return G_nm1_i, G_nm1


    def restore_RCF_dbg(self):
        for n in xrange(self.N + 2):
            print (n, self.l[n].trace().real, self.r[n].trace().real,
                   mm.adot(self.l[n], self.r[n]).real)

        norm_r = mm.adot(self.u_gnd_r.l, self.r[self.N])
        norm_l = mm.adot(self.l[0], self.u_gnd_l.r)
        print norm_r
        print norm_l

        #r_m1 = self.eps_r(None, 0, self.r[0], None)
        #print mm.adot(self.l[0], r_m1).real

        norm = mm.adot(self.l[0], self.r[0]).real

        h = sp.empty((self.N + 1), dtype=self.typ)
        for n in xrange(self.N + 1):
            h[n] = self.expect_2s(self.h_nn, n)
        h *= 1/norm

        self.u_gnd_l.A = self.A[0].copy()
        self.u_gnd_l.l = self.l[0].copy()
        self.u_gnd_l.Calc_AA()
        h_left = self.u_gnd_l.Expect_2S(self.u_gnd_l.h_nn) / norm_l

        self.u_gnd_r.A = self.A[self.N + 1].copy()
        self.u_gnd_r.r = self.r[self.N].copy()
        self.u_gnd_r.Calc_AA()
        h_right = self.u_gnd_l.Expect_2S(self.u_gnd_r.h_nn) / norm_r

        return h, h_left, h_right

    def restore_RCF_r(self):
        G_n_i = None
        for n in reversed(xrange(1, self.N + 2)):
            G_n_i, G_n = self.restore_ONR_n(n, G_n_i)

            self.r[n - 1] = mm.eyemat(self.D[n - 1], self.typ)

            if self.sanity_checks: #and not diag_l:
                r_n = mm.eyemat(self.D[n], self.typ)

                r_nm1 = self.eps_r(None, n, r_n, None)
                if not sp.allclose(r_nm1, self.r[n - 1].A, atol=1E-13, rtol=1E-13):
                    print "Sanity Fail in restore_RCF_r!: r_%u is bad" % (n - 1)
                    print la.norm(r_nm1 - self.r[n - 1])

        #self.r[self.N + 1] = self.r[self.N]

        #Now G_n_i contains g_0_i
        for s in xrange(self.q[0]): #Note: This does not change the scale of A[0]
            self.A[0][s] = mm.mmul(G_n, self.A[0][s], G_n_i)

        self.u_gnd_l.r = mm.mmul(G_n, self.u_gnd_l.r, mm.H(G_n))
        self.l[0] = mm.mmul(mm.H(G_n_i), self.l[0], G_n_i)

    def restore_RCF_l(self):
        G_nm1 = None
        l_nm1 = self.l[0]
        for n in xrange(self.N + 1):
            if n == 0:
                x = l_nm1
            else:
                x = mm.mmul(mm.H(G_nm1), l_nm1, G_nm1)
            M = self.eps_l(None, n, x)
            ev, EV = la.eigh(M)

            self.l[n] = mm.simple_diag_matrix(ev, dtype=self.typ)
            G_n_i = EV

            if n == 0:
                G_nm1 = mm.H(EV) #for left uniform case
                l_nm1 = self.l[n] #for sanity check
                self.u_gnd_l.r = mm.mmul(G_nm1, self.u_gnd_l.r, G_n_i) #since r is not eye

            for s in xrange(self.q[n]):
                self.A[n][s] = mm.mmul(G_nm1, self.A[n][s], G_n_i)

            if self.sanity_checks:
                l = self.eps_l(None, n, l_nm1)
                if not sp.allclose(l, self.l[n], atol=1E-12, rtol=1E-12):
                    print "Sanity Fail in restore_RCF_l!: l_%u is bad" % n
                    print la.norm(l - self.l[n])

            G_nm1 = mm.H(EV)
            l_nm1 = self.l[n]

            if self.sanity_checks:
                if not sp.allclose(sp.dot(G_nm1, G_n_i), sp.eye(G_n_i.shape[0]),
                                   atol=1E-12, rtol=1E-12):
                    print "Sanity Fail in restore_RCF_l!: Bad GT for l_%u" % n

        #Now G_nm1 = G_N
        G_nm1_i = mm.H(G_nm1)
        for s in xrange(self.q[self.N + 1]):
            self.A[self.N + 1][s] = mm.mmul(G_nm1, self.A[self.N + 1][s], G_nm1_i)

        ##This should not be necessary if G_N is really unitary
        #self.r[self.N] = mm.mmul(G_nm1, self.r[self.N], mm.H(G_nm1))
        #self.r[self.N + 1] = self.r[self.N]
        self.u_gnd_r.l[:] = mm.mmul(mm.H(G_nm1_i), self.u_gnd_r.l, G_nm1_i)

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
        self.u_gnd_l.r *= 1/fac

        self.restore_RCF_l()

        if dbg:
            print "Uni left:"
        self.u_gnd_l.A = self.A[0]
        self.u_gnd_l.l = self.l[0]
        self.u_gnd_l.Calc_lr() #Ensures largest ev of E=1
        self.l[0] = self.u_gnd_l.l #No longer diagonal!
        self.A[0] = self.u_gnd_l.A
        if self.sanity_checks:
            if not sp.allclose(self.l[0], sp.diag(sp.diag(self.l[0])), atol=1E-12, rtol=1E-12):
                print "Sanity Fail in restore_RCF!: True l[0] not diagonal!"
        self.l[0] = mm.simple_diag_matrix(sp.diag(self.l[0]))

        fac = 1 / sp.trace(self.l[0]).real
        if dbg:
            print "Scale l[0]: %g" % fac
        self.l[0] *= fac
        self.u_gnd_l.r *= 1/fac

        self.u_gnd_l.l = self.l[0]

        if dbg:
            print "Uni right:"
        self.u_gnd_r.A = self.A[self.N + 1]
        self.u_gnd_r.r = self.r[self.N]
        self.u_gnd_r.Calc_lr() #Ensures largest ev of E=1
        self.r[self.N] = self.u_gnd_r.r
        self.A[self.N + 1] = self.u_gnd_r.A
        if self.sanity_checks:
            if not sp.allclose(self.r[self.N], sp.eye(self.D[self.N]), atol=1E-12, rtol=1E-12):
                print "Sanity Fail in restore_RCF!: True r[N] not eye!"
        self.r[self.N] = mm.eyemat(self.D[self.N], dtype=self.typ)
        self.u_gnd_r.r = self.r[self.N]
        self.r[self.N + 1] = self.r[self.N]

        self.l[self.N + 1][:] = self.eps_l(None, self.N + 1, self.l[self.N])

        if self.sanity_checks:
            l_n = self.l[0]
            for n in xrange(0, self.N + 1):
                l_n = self.eps_l(None, n, l_n)
                if not sp.allclose(l_n, self.l[n], atol=1E-12, rtol=1E-12):
                    print "Sanity Fail in restore_RCF!: l_%u is bad" % n

            r_nm1 = self.r[self.N + 1]
            for n in reversed(xrange(1, self.N + 2)):
                r_nm1 = self.eps_r(None, n, r_nm1, None)
                if not sp.allclose(r_nm1, self.r[n - 1], atol=1E-12, rtol=1E-12):
                    print "Sanity Fail in restore_RCF!: r_%u is bad" % (n - 1)
        if dbg:
            print "AFTER..."
            h_after, h_left_after, h_right_after = self.restore_RCF_dbg()

            print (h_left_after, h_after, h_right_after)

            print h_after - h_before

            print (h_after.sum() - h_before.sum() + h_left_after - h_left_before
                   + h_right_after - h_right_before)

    def check_RCF(self):
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

    def grow_left(self, m):
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

    def grow_right(self, m):
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

    def get_l(self, n):
        if 0 <= n <= self.N + 1:
            return self.l[n]
        elif n < 0:
            return self.l[0]
        else:
            l_m = self.l[self.N + 1]
            for m in xrange(self.N + 2, n + 1):
                l_m = self.eps_l(None, self.N + 1, l_m)
            return l_m

    def get_r(self, n, r_np1=None):
        if 0 <= n <= self.N + 1:
            return self.r[n]
        elif n > self.N + 1:
            return self.r[self.N + 1]
        else:
            if r_np1 is None:
                r_m = self.r[0]
                for m in reversed(xrange(n, 0)):
                    r_m = self.eps_r(None, 0, r_m, None)
                return r_m
            else:
                return self.eps_r(None, 0, r_np1, None)

    def expect_1s(self, o, n):
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
        res = self.eps_r(None, n, self.get_r(n), o)
        return mm.adot(self.get_l(n - 1), res)

    def expect_2s(self, o, n):
        res = self.eps_r_2s(n, self.get_r(n + 1), o)
        return mm.adot(self.get_l(n - 1), res)

    def expect_1s_Cor(self, o1, o2, n1, n2):
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
        r_n = self.eps_r(None, n2, self.r[n2], o2)

        for n in reversed(xrange(n1 + 1, n2)):
            r_n = self.eps_r(None, n, r_n, None)

        r_n = self.eps_r(None, n1, r_n, o1)

        res = mm.mmul(self.l[n1 - 1], r_n)
        return res.trace()

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

        for s2 in xrange(self.q[n2]):
            for t2 in xrange(self.q[n2]):
                r_n2 = mm.mmul(self.A[n2][t2], self.r[n2], mm.H(self.A[n2][s2]))

                r_n = r_n2
                for n in reversed(xrange(n1 + 1, n2)):
                    r_n = self.eps_r(None, n, r_n, None)

                for s1 in xrange(self.q[n1]):
                    for t1 in xrange(self.q[n1]):
                        r_n1 = mm.mmul(self.A[n1][t1], r_n, mm.H(self.A[n1][s1]))
                        tmp = mm.mmul(self.l[n1 - 1], r_n1)
                        rho[s1 * self.q[n1] + s2, t1 * self.q[n1] + t2] = tmp.trace()
        return rho

    def apply_op_1s(self, o, n):
        newA = sp.zeros_like(self.A[n])

        for s in xrange(self.q[n]):
            for t in xrange(self.q[n]):
                newA[s] += self.A[n][t] * o(n, s, t)

        self.A[n] = newA

    def save_state(self, file_name):
        sp.save(file_name, self.A)

    def load_state(self, file_name):
        self.A = sp.load(file_name)
