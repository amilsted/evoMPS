# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 17:29:27 2011

@author: Ashley Milsted

"""
import scipy as sp
import scipy.linalg as la
import matmul as mm
import tdvp_uniform as uni
import tdvp_common as tm

def go(sim, tau, steps, force_calc_lr=False, RK4=False,
       autogrow=False, autogrow_amount=2, autogrow_max_N=1000,
       op=None, op_every=5, prev_op_data=None, op_save_as=None,
       en_save_as=None,
       entropy_save_as=None,
       append_saved=True,
       save_every=10, save_as=None, counter_start=0,
       csv_file=None,
       tol=0,
       print_eta_n=False):
    """A simple integration loop for testing"""
    h_prev = 0

    if not prev_op_data is None:
        data = prev_op_data
    else:
        data = []
        
    endata = []
    if (not en_save_as is None):
        if append_saved:
            try:
                endata = sp.genfromtxt(en_save_as).tolist()
            except:
                print "No previous  en-data, or error loading!"
                pass
            enf = open(en_save_as, "a")
        else:
            enf = open(en_save_as, "w")
            
    Sdata = []
    if (not entropy_save_as is None):
        if append_saved:
            try:
                Sdata = sp.genfromtxt(entropy_save_as).tolist()
            except:
                print "No previous  entropy-data, or error loading!"
                pass
            Sf = open(entropy_save_as, "a")
        else:
            Sf = open(entropy_save_as, "w")
        
    if not op_save_as is None:
        if append_saved:
            try:
                data = sp.genfromtxt(op_save_as).tolist()
            except:
                print "No previous  op-data, or error loading!"
                pass
            opf = open(op_save_as, "a")
        else:
            opf = open(op_save_as, "w")
        
    if not csv_file is None:
        if append_saved:
            csvf = open(csv_file, "a")
        else:
            csvf = open(csv_file, "w")
        
    header = "\t".join(["Step", "eta", "E_nonuniform", "E - E_prev", "grown_left", 
                        "grown_right"])
    print header
    print
    if not csv_file is None:
        csvf.write(header + "\n")

    for i in xrange(counter_start, steps):
        rewrite_opf = False
        if i > counter_start:
            if RK4:
                eta = sim.take_step_RK4(tau)
            else:
                eta = sim.take_step(tau)

            etas = sim.eta[1:].copy()
        
            #Basic dynamic expansion:
            if autogrow and sim.N < autogrow_max_N:
                if etas[0] > sim.eta_uni * 10:
                    rewrite_opf = True
                    print "Growing left by: %u" % autogrow_amount
                    sim.grow_left(autogrow_amount)
                    for j in range(autogrow_amount):
                        for row in data:                        
                            row.insert(0, 0)
                        for row in endata:
                            row.insert(0, 0)
                        for row in Sdata:
                            row.insert(0, 0)
    
                if etas[-1] > sim.eta_uni * 10:
                    rewrite_opf = True
                    print "Growing right by: %u" % autogrow_amount
                    sim.grow_right(autogrow_amount)
                    for j in range(autogrow_amount):
                        for row in data:
                            row.append(0)
                        for row in endata:
                            row.append(0)
                        for row in Sdata:
                            row.append(0)

        else:            
            eta = 0
            etas = sp.zeros(1)
            
        h = sim.update() #now we are measuring the stepped state
            
        if not save_as is None and ((i % save_every == 0)
                                    or i == steps - 1):
            sim.save_state(save_as + "_%u" % i)

        if i % 20 == 19:
            print header
            
        line = "\t".join(map(str, (i, eta.real, h.real, (h - h_prev).real, 
                                   sim.grown_left, sim.grown_right)))
        print line
        if print_eta_n:
            print "eta_n:"
            print etas.real
        
        if not csv_file is None:
            csvf.write(line + "\n")
            csvf.flush()

        h_prev = h

        if (not op is None) and (i % op_every == 0):
            op_range = range(-10, sim.N + 10)
            row = map(lambda n: sim.expect_1s(op, n).real, op_range)
            data.append(row)
            if not op_save_as is None:
                if rewrite_opf:
                    opf.close()
                    opf = open(op_save_as, "w")
                    for row in data:
                        opf.write("\t".join(map(str, row)) + "\n")
                    opf.flush()
                else:
                    opf.write("\t".join(map(str, row)) + "\n")
                    opf.flush()
                    
        if (not en_save_as is None):
            row = sim.h_expect.real.tolist()
            endata.append(row)
            if rewrite_opf:
                enf.close()
                enf = open(en_save_as, "w")
                for row in endata:
                    enf.write("\t".join(map(str, row)) + "\n")
                enf.flush()
            else:
                enf.write("\t".join(map(str, row)) + "\n")
                enf.flush()
                
        if (not entropy_save_as is None):
            row = sim.S_hc.real.tolist()
            Sdata.append(row)
            if rewrite_opf:
                Sf.close()
                Sf = open(entropy_save_as, "w")
                for row in Sdata:
                    Sf.write("\t".join(map(str, row)) + "\n")
                Sf.flush()
            else:
                Sf.write("\t".join(map(str, row)) + "\n")
                Sf.flush()
            
        if i > counter_start and eta.real < tol:
            print "Tolerance reached!"
            break
            
    if not op_save_as is None:
        opf.close()
        
    if (not en_save_as is None):
        enf.close()
        
    if (not entropy_save_as is None):
        Sf.close()
        
    if not csv_file is None:
        csvf.close()

    return data, endata, Sdata

class EvoMPS_TDVP_Sandwich:
    odr = 'C'
    typ = sp.complex128

    #Numsites
    N = 0

    D = None
    q = None

    h_nn = None

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
        self.u_gnd_l = uni.EvoMPS_TDVP_Uniform(uni_ground.D, uni_ground.q)
        self.u_gnd_l.sanity_checks = self.sanity_checks
        self.u_gnd_l.h_nn = uni_ground.h_nn
        self.u_gnd_l.h_nn_cptr = uni_ground.h_nn_cptr
        self.u_gnd_l.A = uni_ground.A.copy()
        self.u_gnd_l.l = uni_ground.l.copy()
        self.u_gnd_l.r = uni_ground.r.copy()

        self.u_gnd_l.symm_gauge = False
        self.u_gnd_l.update()
        self.u_gnd_l.calc_lr()
        self.u_gnd_l.calc_B()
        self.eta_uni = self.u_gnd_l.eta
        self.u_gnd_l_kmr = la.norm(self.u_gnd_l.r / la.norm(self.u_gnd_l.r) - 
                                   self.u_gnd_l.K / la.norm(self.u_gnd_l.K))

        self.u_gnd_r = uni.EvoMPS_TDVP_Uniform(uni_ground.D, uni_ground.q)
        self.u_gnd_r.sanity_checks = self.sanity_checks
        self.u_gnd_r.symm_gauge = False
        self.u_gnd_r.h_nn = uni_ground.h_nn
        self.u_gnd_r.h_nn_cptr = uni_ground.h_nn_cptr
        self.u_gnd_r.A = self.u_gnd_l.A.copy()
        self.u_gnd_r.l = self.u_gnd_l.l.copy()
        self.u_gnd_r.r = self.u_gnd_l.r.copy()
        
        self.grown_left = 0
        self.grown_right = 0
        self.shrunk_left = 0
        self.shrunk_right = 0

        self.h_nn = self.wrap_h
        self.h_nn_mat = None

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
        
    def gen_h_matrix(self):
        """Generates a matrix form for h_nn, which can speed up parts of the
        algorithm by avoiding excess loops and python calls.
        """
        self.h_nn_mat = sp.zeros((self.N + 1, self.q.max(), self.q.max(), 
                                  self.q.max(), self.q.max()), dtype=sp.complex128)
        for n in xrange(self.N + 1):
            for u in xrange(self.q[n]):
                for v in xrange(self.q[n + 1]):
                    for s in xrange(self.q[n]):
                        for t in xrange(self.q[n + 1]):
                            self.h_nn_mat[n, s, t, u, v] = self.h_nn(n, s, t, u, v)

    def calc_C(self, n_low=-1, n_high=-1):
        """Generates the C matrices used to calculate the K's and ultimately the B's

        These are to be used on one side of the super-operator when applying the
        nearest-neighbour Hamiltonian, similarly to C in eqn. (44) of
        arXiv:1103.0936v2 [cond-mat.str-el], except being for the non-norm-preserving case.

        Makes use only of the nearest-neighbour hamiltonian, and of the A's.

        C[n] depends on A[n] and A[n + 1].
        
        This calculation can be significantly faster if a matrix form for h_nn
        is available. See gen_h_matrix().

        """
        if self.h_nn is None:
            return 0

        if n_low < 1:
            n_low = 0
        if n_high < 1:
            n_high = self.N + 1
        
        if self.h_nn_mat is None:
            for n in xrange(n_low, n_high):
                self.C[n] = tm.calc_C_func_op(self.h_nn, self.A[n], self.A[n + 1])
        else:
            for n in xrange(n_low, n_high):
                AA = tm.calc_AA(self.A[n], self.A[n + 1])
                        
                if n == 0: #FIXME: Temp. hack
                    self.AA0 = AA
                elif n == 1:
                    self.AA1 = AA
                
                self.C[n][:] = tm.calc_C_mat_op_AA(self.h_nn_mat[n], AA)

    def calc_K(self):
        """Generates the right K matrices used to calculate the B's

        K[n] is recursively defined. It depends on C[m] and A[m] for all m >= n.

        It directly depends on A[n], A[n + 1], r[n], r[n + 1], C[n] and K[n + 1].

        This is equivalent to K on p. 14 of arXiv:1103.0936v2 [cond-mat.str-el], except
        that it is for the non-norm-preserving case.

        K[1] is, assuming a normalized state, the expectation value H of Ĥ.
        
        Return the excess energy.
        """
        n_low = 0
        n_high = self.N + 1
   
        self.h_expect = sp.zeros((self.N + 1), dtype=self.typ)
        
        self.u_gnd_r.calc_AA()
        self.u_gnd_r.calc_C()
        self.u_gnd_r.calc_K()
        self.K[self.N + 1][:] = self.u_gnd_r.K

        for n in reversed(xrange(n_low, n_high)): #FIXME: lm1???
            self.K[n], he = tm.calc_K(self.K[n + 1], self.C[n], self.get_l(n),
                                      self.r[n + 1], self.A[n], self.A[n + 1],
                                      sanity_checks=self.sanity_checks)
                
            self.h_expect[n] = he
            
        self.u_gnd_l.calc_AA()
        self.u_gnd_l.calc_C()
        K_left, h_left_uni = self.u_gnd_l.calc_K_l()

        h = (mm.adot(K_left, self.r[0]) + mm.adot(self.l[0], self.K[0]) 
             - (self.N + 1) * self.u_gnd_r.h)
             
        return h

    def calc_x(self, n, Vsh, sqrt_l, sqrt_r, sqrt_l_inv, sqrt_r_inv):
        """Calculate the parameter matrix x* giving the desired B.

        This is equivalent to eqn. (49) of arXiv:1103.0936v2 [cond-mat.str-el] except
        that, here, norm-preservation is not enforced, such that the optimal
        parameter matrices x*_n (for the parametrization of B) are given by the
        derivative w.r.t. x_n of <Phi[B, A]|Ĥ|Psi[A]>, rather than
        <Phi[B, A]|Ĥ - H|Psi[A]> (with H = <Psi|Ĥ|Psi>).

        Direct dependencies:
            - A[n - 1], A[n], A[n + 1]
            - r[n], r[n + 1], l[n - 2], l[n - 1]
            - C[n], C[n - 1]
            - K[n + 1]
            - V[n]
        """
        if n > 0:
            lm2 = self.get_l(n - 2)
        else:
            lm2 = None
            
        if n < self.N + 1:
            C = self.C[n]
        else:
            C = None
            
        x = tm.calc_x(self.K[n + 1], C, self.C[n - 1], self.r[n + 1],
                      lm2, self.A[n - 1], self.A[n], self.A[n + 1],
                      sqrt_l, sqrt_l_inv, sqrt_r, sqrt_r_inv, Vsh)

        return x
        
    def calc_B1(self):
        """Calculate the optimal B1 given right gauge-fixing on B2..N and
        no gauge-fixing on B1.
        
        We use the non-norm-preserving K's, since the norm-preservation
        is not needed elsewhere. It is cleaner to subtract the relevant
        norm-changing terms from the K's here than to generate all K's
        with norm-preservation.
        """
        B1 = sp.empty_like(self.A[1])
        
        try:
            r1_i = self.r[1].inv()
        except AttributeError:
            r1_i = mm.invmh(self.r[1])
            
        try:
            l0_i = self.l[0].inv()
        except AttributeError:
            l0_i = mm.invmh(self.l[0])
        
        A0 = self.A[0]
        A1 = self.A[1]
        A2 = self.A[2]
        r1 = self.r[1]
        r2 = self.r[2]
        l0 = self.l[0]
        
        KLh = mm.H(self.u_gnd_l.K_left - l0 * mm.adot(self.u_gnd_l.K_left, self.r[0]))
        K2 = self.K[2] - r1 * mm.adot(self.l[1], self.K[2])
        
        C1 = self.C[1] - self.h_expect[1] * self.AA1
        C0 = self.C[0] - self.h_expect[0] * self.AA0
        
        for s in xrange(self.q[1]):
            try:
                B1[s] = A1[s].dot(r1_i.dot_left(K2))
            except AttributeError:
                B1[s] = A1[s].dot(K2.dot(r1_i))
            
            for t in xrange(self.q[2]):
                try:
                    B1[s] += C1[s, t].dot(r2.dot(r1_i.dot_left(mm.H(A2[t]))))
                except AttributeError:
                    B1[s] += C1[s, t].dot(r2.dot(mm.H(A2[t]).dot(r1_i)))                    
                
            B1sbit = KLh.dot(A1[s])
                            
            for t in xrange(self.q[0]):
                B1sbit += mm.H(A0[t]).dot(l0.dot(C0[t,s]))
                
            B1[s] += l0_i.dot(B1sbit)
           
        rb = tm.eps_r_noop(r1, B1, B1)
        eta = sp.sqrt(mm.adot(l0, rb))
                
        return B1, eta

    def calc_B(self, n, set_eta=True):
        """Generates the B[n] tangent vector corresponding to physical evolution of the state.

        In other words, this returns B[n][x*] (equiv. eqn. (47) of
        arXiv:1103.0936v2 [cond-mat.str-el])
        with x* the parameter matrices satisfying the Euler-Lagrange equations
        as closely as possible.
        
        In the case of B1, use the general B1 generated in calc_B1().
        """
        if self.q[n] * self.D[n] - self.D[n - 1] > 0:
            if n == 1:
                B, eta1 = self.calc_B1()
                if set_eta:
                    self.eta[1] = eta1
            else:
                l_sqrt, r_sqrt, l_sqrt_inv, r_sqrt_inv = self.calc_l_r_roots(n)
    
                Vsh = tm.calc_Vsh(self.A[n], r_sqrt, sanity_checks=self.sanity_checks)
    
                x = self.calc_x(n, Vsh, l_sqrt, r_sqrt, l_sqrt_inv, r_sqrt_inv)
                
                if set_eta:
                    self.eta[n] = sp.sqrt(mm.adot(x, x))
    
                B = sp.empty_like(self.A[n])
                for s in xrange(self.q[n]):
                    B[s] = mm.mmul(l_sqrt_inv, x, mm.H(Vsh[s]), r_sqrt_inv)

            if self.sanity_checks:
                M = tm.eps_r_noop(self.r[n], B, self.A[n])
                if not sp.allclose(M, 0):
                    print "Sanity Fail in calc_B!: B_%u does not satisfy GFC!" % n

            return B
        else:
            return None, 0

    def calc_l_r_roots(self, n):
        """Returns the matrix square roots (and inverses) needed to calculate B.

        Hermiticity of l[n] and r[n] is used to speed this up.
        If an exception occurs here, it is probably because these matrices
        are no longer Hermitian (enough).
        
        If l[n] or r[n] are diagonal or the identity, further optimizations are
        used.
        """
        l_sqrt, l_sqrt_i, r_sqrt, r_sqrt_i = tm.calc_l_r_roots(self.l[n - 1], 
                                                               self.r[n], 
                                                            self.sanity_checks)

        return l_sqrt, r_sqrt, l_sqrt_i, r_sqrt_i

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
        
        self.calc_C()
        h = self.calc_K()

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

        return eta_tot


    def take_step_RK4(self, dtau):
        """Take a step using the fourth-order explicit Runge-Kutta method.

        This requires more memory than a simple forward Euler step, and also
        more than a backward Euler step. It is, however, far more accurate
        and stable than forward Euler.
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
            self.l[n] = tm.eps_l_noop(self.l[n - 1], self.A[n], self.A[n])

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
            self.r[n] = tm.eps_r_noop(self.r[n + 1], self.A[n + 1], self.A[n + 1])

    def restore_RCF_dbg(self):
        for n in xrange(self.N + 2):
            print (n, self.l[n].trace().real, self.r[n].trace().real,
                   mm.adot(self.l[n], self.r[n]).real)

        norm_r = mm.adot(self.u_gnd_r.l, self.r[self.N])
        norm_l = mm.adot(self.l[0], self.u_gnd_l.r)
        print norm_r
        print norm_l

        #r_m1 = self.eps_r(0, self.r[0])
        #print mm.adot(self.l[0], r_m1).real

        norm = mm.adot(self.l[0], self.r[0]).real

        h = sp.empty((self.N + 1), dtype=self.typ)
        for n in xrange(self.N + 1):
            h[n] = self.expect_2s(self.h_nn, n)
        h *= 1/norm

        self.u_gnd_l.A = self.A[0].copy()
        self.u_gnd_l.l = self.l[0].copy()
        self.u_gnd_l.calc_AA()
        h_left = self.u_gnd_l.expect_2s(self.u_gnd_l.h_nn) / norm_l

        self.u_gnd_r.A = self.A[self.N + 1].copy()
        self.u_gnd_r.r = self.r[self.N].copy()
        self.u_gnd_r.calc_AA()
        h_right = self.u_gnd_l.expect_2s(self.u_gnd_r.h_nn) / norm_r

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

        self.u_gnd_l.r = mm.mmul(G_n, self.u_gnd_l.r, mm.H(G_n))
        self.l[0] = mm.mmul(mm.H(G_n_i), self.l[0], G_n_i)

    def restore_RCF_l(self):
        G_nm1 = None
        l_nm1 = self.l[0]
        for n in xrange(self.N + 1):
            self.l[n], G_nm1, G_nm1_i = tm.restore_RCF_l(self.A[n], l_nm1, 
                                      G_nm1, sanity_checks=self.sanity_checks)
            
            if n == 0:
                self.u_gnd_l.r = mm.mmul(G_nm1, self.u_gnd_l.r, G_nm1_i) #since r is not eye

            l_nm1 = self.l[n]

        #Now G_nm1 = G_N
        for s in xrange(self.q[self.N + 1]):
            self.A[self.N + 1][s] = mm.mmul(G_nm1, self.A[self.N + 1][s], G_nm1_i)

        ##This should not be necessary if G_N is really unitary
        #self.r[self.N] = mm.mmul(G_nm1, self.r[self.N], mm.H(G_nm1))
        #self.r[self.N + 1] = self.r[self.N]
        self.u_gnd_r.l = mm.mmul(mm.H(G_nm1_i), self.u_gnd_r.l, G_nm1_i)
        
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
        self.u_gnd_l.r *= 1/fac

        self.restore_RCF_l()

        if dbg:
            print "Uni left:"
        self.u_gnd_l.A = self.A[0]
        self.u_gnd_l.l = self.l[0]
        self.u_gnd_l.calc_lr() #Ensures largest ev of E=1
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
        self.u_gnd_r.calc_lr() #Ensures largest ev of E=1
        self.r[self.N] = self.u_gnd_r.r
        self.A[self.N + 1] = self.u_gnd_r.A
        if self.sanity_checks:
            if not sp.allclose(self.r[self.N], sp.eye(self.D[self.N]), atol=1E-12, rtol=1E-12):
                print "Sanity Fail in restore_RCF!: True r[N] not eye!"
        self.r[self.N] = mm.eyemat(self.D[self.N], dtype=self.typ)
        self.u_gnd_r.r = self.r[self.N]
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
        self.u_gnd_l.calc_lr()
        g, gi = self.u_gnd_l.restore_CF(ret_g=True)

        m = self.u_gnd_l.A.mean()
        print sp.sqrt(sp.conj(m) / m)
        self.u_gnd_l.A *= sp.sqrt(sp.conj(m) / m)
        
        self.u_gnd_l.calc_lr()
        
        self.A[0] = self.u_gnd_l.A
        for s in xrange(self.A[1].shape[0]):
            self.A[1][s] = gi.dot(self.A[1][s])
        self.l[0] = self.u_gnd_l.l
        
        self.u_gnd_r.calc_lr()
        g, gi = self.u_gnd_r.restore_CF(ret_g=True)
        
        m = self.u_gnd_r.A.mean()
        print sp.sqrt(sp.conj(m) / m)
        self.u_gnd_r.A *= sp.sqrt(sp.conj(m) / m)
        
        self.u_gnd_r.calc_lr()
        
        self.A[self.N + 1] = self.u_gnd_r.A
        for s in xrange(self.A[1].shape[0]):
            self.A[self.N][s] = self.A[self.N][s].dot(g)
        self.r[self.N] = self.u_gnd_r.r
        self.r[self.N + 1] = self.r[self.N]
        
        self.calc_r()
        
        norm = mm.adot(self.l[0], self.r[0])
        
        self.A[1] *= 1 / sp.sqrt(norm)
        
        self.calc_l()
        self.calc_r()
        
        #TODO: Phase!
    
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
        
        self.gen_h_matrix()

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
        
        self.gen_h_matrix()
    
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
        
        self.gen_h_matrix()
        
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
        
        self.gen_h_matrix()
    
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
        A = self.get_A(n)
        op = lambda s, t: o(n, s, t)
        opv = sp.vectorize(op, otypes=[sp.complex128])
        opm = sp.fromfunction(opv, (A.shape[0], A.shape[0]))
        res = tm.eps_r_op_1s(self.get_r(n), A, A, opm)
        return mm.adot(self.get_l(n - 1), res)

    def expect_2s(self, o, n):
        A = self.get_A(n)
        Ap1 = self.get_A(n + 1)
        AA = tm.calc_AA(A, Ap1)
        op = lambda s,t,u,v: o(n, s, t, u, v)
        res = tm.eps_r_op_2s_AA(self.get_r(n + 1), AA, AA, op)
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
        r_n = tm.eps_r_op_1s(self.r[n2], self.A[n2], self.A[n2], o2)

        for n in reversed(xrange(n1 + 1, n2)):
            r_n = tm.eps_r_noop(r_n, self.A[n], self.A[n])

        r_n = tm.eps_r_op_1s(r_n, self.A[n1], self.A[n1], o1)

        return mm.adot(self.l[n1 - 1], r_n)

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
                    r_n = tm.eps_r_noop(r_n, self.A[n], self.A[n])

                for s1 in xrange(self.q[n1]):
                    for t1 in xrange(self.q[n1]):
                        r_n1 = mm.mmul(self.A[n1][t1], r_n, mm.H(self.A[n1][s1]))
                        tmp = mm.adot(self.l[n1 - 1], r_n1)
                        rho[s1 * self.q[n1] + s2, t1 * self.q[n1] + t2] = tmp
        return rho

    def apply_op_1s(self, o, n):
        """Applies a one-site operator o to site n.
        
        Can be used to create excitations.
        """
        newA = sp.zeros_like(self.A[n])

        for s in xrange(self.q[n]):
            for t in xrange(self.q[n]):
                newA[s] += self.A[n][t] * o(n, s, t)

        self.A[n] = newA
        
    def overlap(self, other, sanity_checks=False):
        dL, phiL, gLl = self.u_gnd_l.fidelity_per_site(other.u_gnd_l, full_output=True, left=True)
        
        dR, phiR, gRr = self.u_gnd_r.fidelity_per_site(other.u_gnd_r, full_output=True, left=False)
        
        gr = mm.H(la.inv(gRr).dot(sp.asarray(self.u_gnd_r.r)))
        gri = mm.H(la.inv(sp.asarray(self.u_gnd_r.r)).dot(gRr))
        
        if sanity_checks:
            AR = other.u_gnd_r.A.copy()        
            for s in xrange(AR.shape[0]):
                AR[s] = gr.dot(AR[s]).dot(gri)
                
            print la.norm(AR - self.u_gnd_r.A)
        
        r = gr.dot(sp.asarray(other.u_gnd_r.r)).dot(mm.H(gr))
        fac = la.norm(sp.asarray(self.u_gnd_r.r)) / la.norm(r)        
        gr *= sp.sqrt(fac)
        gri /= sp.sqrt(fac)
        
        if sanity_checks:
            r *= fac
            print la.norm(r - self.u_gnd_r.r)

            AN = other.A[self.N].copy()
            for s in xrange(AN.shape[0]):
                AN[s] = AN[s].dot(gri)
            r = tm.eps_r_noop(self.u_gnd_r.r, AN, AN)
            print la.norm(r - other.r[self.N - 1])

        gl = la.inv(sp.asarray(self.u_gnd_l.l)).dot(gLl)
        gli = la.inv(gLl).dot(sp.asarray(self.u_gnd_l.l))
        
        l = mm.H(gli).dot(sp.asarray(other.u_gnd_l.l)).dot(gli)
        fac = la.norm(sp.asarray(self.u_gnd_l.l)) / la.norm(l)

        gli *= sp.sqrt(fac)
        gl /= sp.sqrt(fac)
        
        if sanity_checks:
            l *= fac
            print la.norm(l - self.u_gnd_l.l)
        
            l = mm.H(gli).dot(sp.asarray(other.u_gnd_l.l)).dot(gli)
            print la.norm(l - self.u_gnd_l.l)
        
        print (dL, dR, phiL, phiR)
        
        if not self.N == other.N:
            print "States must have same number of non-uniform sites!"
            return
            
        if not sp.all(self.D == other.D):
            print "States must have same bond-dimensions!"
            return
            
        if not sp.all(self.q == other.q):
            print "States must have same Hilbert-space dimensions!"
            return

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
        
        r = tm.eps_r_noop(self.u_gnd_r.r, self.A[self.N], AN)
        for n in xrange(self.N - 1, 1, -1):
            r = tm.eps_r_noop(r, self.A[n], other.A[n])
        r = tm.eps_r_noop(r, self.A[1], A1)
    
        return mm.adot(self.u_gnd_l.l, r)
        

    def save_state(self, file_name, userdata=None):
        tosave = sp.empty((9), dtype=sp.ndarray)
        
        tosave[0] = self.A
        tosave[1] = self.l[0]
        tosave[2] = self.u_gnd_l.r
        tosave[3] = self.u_gnd_l.K_left
        tosave[4] = self.r[self.N]
        tosave[5] = self.u_gnd_r.l
        tosave[6] = self.u_gnd_r.K
        tosave[7] = sp.array([[self.grown_left, self.grown_right], 
                             [self.shrunk_left, self.shrunk_right]])
        tosave[8] = userdata
        
        sp.save(file_name, tosave)

    def load_state(self, file_name, autogrow=False):
        toload = sp.load(file_name)
        
        try:
            if toload.shape[0] != 9:
                print "Error loading state: Bad data!"
                return
                
            if autogrow and toload[0].shape[0] != self.A.shape[0]:
                newN = toload[0].shape[0] - 3
                print "Changing N to: %u" % newN
                self.grow_left(newN - self.N)
                
            if toload[0].shape != self.A.shape:
                print "Cannot load state: Dimension mismatch!"
                return
            
            self.A = toload[0]
            self.l[0] = toload[1]
            self.u_gnd_l.r = toload[2]
            self.u_gnd_l.K_left = toload[3]
            self.r[self.N] = toload[4]
            self.r[self.N + 1] = self.r[self.N]
            self.u_gnd_r.l = toload[5]
            self.u_gnd_r.K = toload[6]
            
            self.grown_left = toload[7][0, 0]
            self.grown_right = toload[7][0, 1]
            self.shrunk_left = toload[7][1, 0]
            self.shrunk_right = toload[7][1, 1]
            
            return toload[8]
            
        except AttributeError:
            print "Error loading state: Bad data!"
            return