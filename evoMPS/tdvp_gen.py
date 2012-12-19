# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 17:29:27 2011

@author: Ashley Milsted

TODO:
    - Implement evaluation of the error due to restriction to bond dim.
    - Add an algorithm for expanding the bond dimension.
    - Adaptive step size.
    - Find a way to randomize the starting state.

"""
import scipy as sp
import scipy.linalg as la
import matmul as m
import tdvp_merged as tm

class EvoMPS_TDVP_Generic:
    odr = 'C'
    typ = sp.complex128
    
    #Numsites
    N = 0
    
    D = None
    q = None
    
    h_nn = None
    h_ext = None
    
    eps = 0
    
    sanity_checks = True
    
    def setup_A(self):
        """Initializes the state to full rank with norm 1.
        """
        for n in xrange(1, self.N + 1):
            self.A[n].fill(0)
            
            f = sp.sqrt(1. / self.q[n])
            
            if self.D[n-1] == self.D[n]:
                for s in xrange(self.q[n]):
                    sp.fill_diagonal(self.A[n][s], f)
            else:
                x = 0
                y = 0
                s = 0
                
                if self.D[n] > self.D[n - 1]:
                    f = 1.
                
                for i in xrange(max((self.D[n], self.D[n - 1]))):
                    self.A[n][s, x, y] = f
                    x += 1
                    y += 1
                    if x >= self.A[n][s].shape[0]:
                        x = 0
                        s += 1
                    elif y >= self.A[n][s].shape[1]:
                        y = 0
                        s += 1
    
    def randomize(self):
        """Set A's randomly, trying to keep the norm reasonable.
        
        We need to ensure that the M matrix in eps_r() is positive definite. How?
        Does it really have to be?
        """
        for n in xrange(1, self.N + 1):
            self.A[n].real = (sp.rand(self.D[n - 1], self.D[n]) - 0.5) / sp.sqrt(self.q[n]) #/ sp.sqrt(self.N) #/ sp.sqrt(self.D[n])
            self.A[n].imag = (sp.rand(self.D[n - 1], self.D[n]) - 0.5) / sp.sqrt(self.q[n]) #/ sp.sqrt(self.N) #/ sp.sqrt(self.D[n])
                
        self.restore_RCF()
            
    def __init__(self, numsites, D, q):
        """Creates a new TDVP_MPS object.
        
        The TDVP_MPS class implements the time-dependent variational principle 
        for matrix product states for systems with open boundary conditions and
        a hamiltonian consisting of a nearest-neighbour interaction term and a 
        single-site term (external field).
        
        Bond dimensions will be adjusted where they are too high to be useful.
        FIXME: Add reference.
        
        Parameters
        ----------
        numsites : int
            The number of lattice sites.
        D : ndarray
            A 1-d array, length numsites, of integers indicating the desired bond dimensions.
        q : ndarray
            A 1-d array, also length numsites, of integers indicating the 
            dimension of the hilbert space for each site.
    
        Returns
        -------
        sqrt_A : ndarray
            An array of the same shape and type as A containing the matrix square root of A.        
        """
        self.eps = sp.finfo(self.typ).eps
        
        self.N = numsites
        self.D = sp.array(D)
        self.q = sp.array(q)
        
        #Make indicies correspond to the thesis
        self.K = sp.empty((self.N + 1), dtype=sp.ndarray) #Elements 1..N
        self.C = sp.empty((self.N), dtype=sp.ndarray) #Elements 1..N-1
        self.A = sp.empty((self.N + 1), dtype=sp.ndarray) #Elements 1..N
        
        self.r = sp.empty((self.N + 1), dtype=sp.ndarray) #Elements 0..N
        self.l = sp.empty((self.N + 1), dtype=sp.ndarray)        
        
        if (self.D.ndim != 1) or (self.q.ndim != 1):
            raise NameError('D and q must be 1-dimensional!')
            
        #TODO: Check for integer type.
        
        #Don't do anything pointless
        self.D[0] = 1
        self.D[self.N] = 1

        qacc = 1
        for n in reversed(xrange(self.N)):
            if qacc < self.D.max(): #Avoid overflow!
                qacc *= self.q[n + 1]

            if self.D[n] > qacc:
                self.D[n] = qacc
                
        qacc = 1
        for n in xrange(1, self.N + 1):
            if qacc < self.D.max(): #Avoid overflow!
                qacc *= q[n - 1]

            if self.D[n] > qacc:
                self.D[n] = qacc
        
        self.r[0] = sp.zeros((self.D[0], self.D[0]), dtype=self.typ, order=self.odr)  
        self.l[0] = sp.eye(self.D[0], self.D[0], dtype=self.typ).copy(order=self.odr) #Already set the 0th element (not a dummy)    
    
        for n in xrange(1, self.N + 1):
            self.K[n] = sp.zeros((self.D[n-1], self.D[n-1]), dtype=self.typ, order=self.odr)    
            self.r[n] = sp.zeros((self.D[n], self.D[n]), dtype=self.typ, order=self.odr)
            self.l[n] = sp.zeros((self.D[n], self.D[n]), dtype=self.typ, order=self.odr)
            self.A[n] = sp.empty((self.q[n], self.D[n-1], self.D[n]), dtype=self.typ, order=self.odr)
            if n < self.N:
                self.C[n] = sp.empty((self.q[n], self.q[n+1], self.D[n-1], self.D[n+1]), dtype=self.typ, order=self.odr)
        sp.fill_diagonal(self.r[self.N], 1.)
        self.setup_A()
        
        self.eta = sp.zeros((self.N + 1), dtype=self.typ)
    
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
            n_low = 1
        if n_high < 1:
            n_high = self.N
        
        for n in xrange(n_low, n_high):
            h_nn_n = lambda s, t, u, v: self.h_nn(n, s, t, u, v) 
            self.C[n] = tm.calc_C_func_op(h_nn_n, self.A[n], self.A[n + 1])
    
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
            n_low = 1
        if n_high < 1:
            n_high = self.N + 1
            
        for n in reversed(xrange(n_low, n_high)):
            if n < self.N:
                self.K[n], ex = tm.calc_K(self.K[n + 1], self.C[n], self.l[n - 1], 
                                          self.r[n + 1], self.A[n], self.A[n + 1], 
                                          sanity_checks=self.sanity_checks)
            else:
                self.K[n].fill(0)
            
            if not self.h_ext is None:
                for s in xrange(self.q[n]):
                    for t in xrange(self.q[n]):
                        h_ext_st = self.h_ext(n, s, t)
                        if h_ext_st != 0:
                            self.K[n] += h_ext_st * self.A[n][t].dot(self.r[n].dot(m.H(self.A[n][s])))
    
    def update(self):
        self.calc_l()
        self.calc_r()
        self.restore_RCF()
        self.calc_C()
        self.calc_K()
        
    def calc_x(self, n, Vsh, sqrt_l, sqrt_r, sqrt_l_inv, sqrt_r_inv):
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
        if n > 1:
            lm2 = self.l[n - 2]
        else:
            lm2 = None
            
        if n < self.N:
            C = self.C[n]
        else:
            C = None
            
        x = tm.calc_x(self.K[n + 1], C, self.C[n - 1], self.r[n + 1],
                      lm2, self.A[n - 1], self.A[n], self.A[n + 1],
                      sqrt_l, sqrt_l_inv, sqrt_r, sqrt_r_inv, Vsh)
                              
        #Extra term to take care of h_ext..
        if not self.h_ext is None:
            x_subpart = sp.zeros_like(self.A[n][0])
            x_subsubpart = sp.empty_like(self.A[n][0])
            
            for s in xrange(self.q[n]):
                x_subsubpart.fill(0)
                for t in xrange(self.q[n]):
                    x_subsubpart += self.h_ext(n, s, t) * self.A[n][t] #it may be more effecient to squeeze this into the nn term...
                x_subpart += x_subsubpart.dot(sqrt_r.dot(Vsh[s]))
            x += sqrt_l.dot(x_subpart)
                
        return x
        
    def calc_B(self, n, set_eta=True):
        """Generates the B[n] tangent vector corresponding to physical evolution of the state.
        
        In other words, this returns B[n][x*] (equiv. eqn. (47) of 
        arXiv:1103.0936v2 [cond-mat.str-el]) 
        with x* the parameter matrices satisfying the Euler-Lagrange equations
        as closely as possible.
        """
        if self.q[n] * self.D[n] - self.D[n - 1] > 0:
            l_sqrt, r_sqrt, l_sqrt_inv, r_sqrt_inv = self.calc_l_r_roots(n)
            
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
        
    def calc_l_r_roots(self, n):
        """Returns the matrix square roots (and inverses) needed to calculate B.
        
        Hermiticity of l[n] and r[n] is used to speed this up.
        If an exception occurs here, it is probably because these matrices
        are not longer Hermitian (enough).
        """
        l_sqrt, l_sqrt_i, r_sqrt, r_sqrt_i = tm.calc_l_r_roots(self.l[n - 1], 
                                                               self.r[n], 
                                                            self.sanity_checks)

        
        return l_sqrt, r_sqrt, l_sqrt_i, r_sqrt_i
    
    def take_step(self, dtau): #simple, forward Euler integration     
        """Performs a complete forward-Euler step of imaginary time dtau.
        
        If dtau is itself imaginary, real-time evolution results.
        
        Here, the A's are updated as the sites are visited. Since we want all
        tangent vectors to be generated using the old state, we must delay
        updating each A[n] until we are *two* steps away (due to the direct
        dependency on A[n - 1] in calc_x).
        
        The dependencies on l, r, C and K are not a problem because we store
        all these matrices separately and do not update them at all during take_step().
        
        Parameters
        ----------
        dtau : complex
            The (imaginary or real) amount of imaginary time (tau) to step.
        """
        eta_tot = 0
        
        B_prev = None
        for n in xrange(1, self.N + 2):
            #V is not always defined (e.g. at the right boundary vector, and possibly before)
            if n <= self.N:
                B = self.calc_B(n)
                eta_tot += self.eta[n]
            
            if n > 1 and not B_prev is None:
                self.A[n - 1] += -dtau * B_prev
                
            B_prev = B
            
        return eta_tot

    def take_step_implicit(self, dtau, midpoint=True):
        """A backward (implicit) integration step.
        
        Based on p. 8-10 of arXiv:1103.0936v2 [cond-mat.str-el].
        
        NOTE: Not currently working as well as expected. Iterative solution of 
              implicit equation stops converging at some point...
        
        This is made trickier by the gauge freedom. We solve the implicit equation iteratively, aligning the tangent
        vectors along the gauge orbits for each step to obtain the physically relevant difference dA. The gauge-alignment is done
        by solving a matrix equation.
        
        The iteration seems to be difficult. At the moment, iteration over the whole chain is combined with iteration over a single
        site (a single A[n]). Iteration over the chain is needed because the evolution of a single site depends on the state of the
        whole chain. The algorithm runs through the chain from N to 1, iterating a few times over each site (depending on how many
        chain iterations we have done, since iterating over a single site becomes more fruitful as the rest of the chain stabilizes).
        
        In running_update mode, which is likely the most sensible choice, the current midpoint guess is updated after visiting each
        site. I.e. The trial backwards step at site n is based on the updates that were made to site n + 1, n + 2 during the current chain
        iteration.
        
        Parameters
        ----------
        dtau : complex
            The (imaginary or real) amount of imaginary time (tau) to step.
        midpoint : bool
            Whether to use approximately time-symmetric midpoint integration,
            or just a backward-Euler step.
        """        
        #---------------------------
        #Hard-coded params:
        debug = True
        dbg_bstep = False
        safe_mode = True
        
        tol = sp.finfo(sp.complex128).eps * 3
        max_iter = 10
        itr_switch_mode = 10
        #---------------------------
        
        if midpoint:
            dtau = dtau / 2
        
        self.restore_RCF()

        #Take a copy of the current state
        A0 = sp.empty_like(self.A)
        for n in xrange(1, self.N + 1):
            A0[n] = self.A[n].copy()
        
        #Take initial forward-Euler step
        self.take_step(dtau)     

        itr = 0
        delta = 1
        delta_prev = 0                
        final_check = False

        while delta > tol * (self.N - 1) and itr < max_iter or final_check:
            print "OUTER"
            running_update = itr < itr_switch_mode
            
            A_np1 = A0[self.N]            
            
            #Prepare for next calculation of B from the new A
            self.restore_RCF() #updates l and r
            
            if running_update:
                self.calc_C() #we really do need all of these, since B directly uses C[n-1]
                self.calc_K()            
            
            g0_n = sp.eye(self.D[self.N - 1], dtype=self.typ)       #g0_n is the gauge transform matrix needed to solve the implicit equation
            
            #Loop through the chain, optimizing the individual A's
            delta = 0
            for n in reversed(xrange(1, self.N)): #We start at N - 1, since the right vector can't be altered here.
                print "SWEEP"
                if not running_update: #save new A[n + 1] and replace with old version for building B
                    A_np1_new = self.A[n + 1].copy()
                    self.A[n + 1] = A_np1  
                    A_np1 = self.A[n].copy()
                    max_itr_n = 1 #wait until the next run-through of the chain to change A[n] again
                else:
                    max_itr_n = itr + 1 #do more iterations here as the outer loop progresses
                
                delta_n = 1
                itr_n = 0
                while True:
                    print "INNER"
                    #Find transformation to gauge-align A0 with the backwards-obtained A.. is this enough?
                    M = m.mmul(A0[n][0], g0_n, self.r[n], m.H(self.A[n][0]))
                    for s in xrange(1, self.q[n]):
                        M += m.mmul(A0[n][s], g0_n, self.r[n], m.H(self.A[n][s]))
                    
                    g0_nm1 = la.solve(self.r[n - 1], M, sym_pos=True, overwrite_b=True)
                    
                    if not (delta_n > tol and itr_n < max_itr_n):
                        break
                    
                    B = self.calc_B(n)
                    
                    if B is None:
                        delta_n = 0
                        fnorm = 0
                        break
                    
                    g0_nm1_inv = la.inv(g0_nm1) #sadly, we need the inverse too...    
                    r_dA = sp.zeros_like(self.r[n - 1])
                    dA = sp.empty_like(self.A[n])
                    sqsum = 0
                    for s in xrange(self.q[n]):
                        dA[s] = m.mmul(g0_nm1_inv, A0[n][s], g0_n)
                        dA[s] -= self.A[n][s] 
                        dA[s] -= dtau * B[s]
                        if not final_check:
                            self.A[n][s] += dA[s]
    
                    for s in xrange(self.q[n]):
                        r_dA += m.mmul(dA[s], self.r[n], m.H(dA[s]))
                        sqsum += sum(dA[s]**2)
                    
                    fnorm = sp.sqrt(sqsum)
                    
                    delta_n = sp.sqrt(sp.trace(m.mmul(self.l[n - 1], r_dA)))
                    
                    if running_update: #Since we want to use the current A[n] and A[n + 1], we need this:
                        if safe_mode:
                            self.restore_RCF()
                            self.calc_C()
                            self.calc_K()
                        else:
                            self.restore_RCF(start=n) #will also renormalize
                            self.calc_C(n_low=n-1, n_high=n)
                            self.calc_K(n_low=n, n_high=n+1)
                                        
                    itr_n += 1
                    
                    if final_check:
                        break
                    
                if not running_update: #save new A[n + 1] and replace with old version for building B
                    self.A[n + 1] = A_np1_new
                
                if debug:
                    print "delta_%d: %g, (%d iterations)" % (n, delta_n.real, itr_n) + ", fnorm = " + str(fnorm)
                delta += delta_n
                
                if safe_mode:
                    self.calc_r()
                else:
                    self.calc_r(n - 2, n - 1) #We only need these for the next step.
                
                g0_n = g0_nm1                
                            
            itr += 1
            if debug:
                print "delta: %g  delta delta: %g (%d iterations)" % (delta.real, (delta - delta_prev).real, itr)
            delta_prev = delta
            
            if debug:
                if final_check:
                    break
                elif delta <= tol * (self.N - 1) or itr >= max_iter:
                    print "Final check to get final delta:"
                    final_check = True
        
        #Test backward step!        
        if dbg_bstep:
            Anew = sp.empty_like(self.A)
            for n in xrange(1, self.N + 1):
                Anew[n] = self.A[n].copy()
            
#            self.calc_l()
#            self.simple_renorm()
#            self.restore_RCF()
            self.calc_C()
            self.calc_K()        
            self.take_step(-dtau)
            self.restore_RCF()
            
            delta2 = 0            
            for n in reversed(xrange(1, self.N + 1)):
                #print n
                dA = A0[n] - self.A[n]
                #Surely this dA should also preserve the gauge choice, since both A's are in ON_R...                
                #print dA/A0[n]
                r_dA = sp.zeros_like(self.r[n - 1])
                sqsum = 0
                for s in xrange(self.q[n]):
                    r_dA += m.mmul(dA[s], self.r[n], m.H(dA[s]))
                    sqsum += sum(dA[s]**2)
                delta_n = sp.sqrt(sp.trace(m.mmul(self.l[n - 1], r_dA)))                
                delta2 += delta_n
                if debug:
                    print "A[%d] OK?: " % n + str(sp.allclose(dA, 0)) + ", delta = " + str(delta_n) + ", fnorm = " + str(sp.sqrt(sqsum))
                #print delta_n
            if debug:
                print "Total delta: " + str(delta2)
                
            for n in xrange(1, self.N + 1):
                self.A[n] = Anew[n]
        else:
            delta2 = 0
            
        if midpoint:
            #Take a final step from the midpoint
            #self.restore_RCF() #updates l and r            
            self.calc_l()
            self.simple_renorm()
            self.calc_C()
            self.calc_K()
            self.take_step(dtau)
            
        return itr, delta, delta2
        
    def take_step_RK4(self, dtau):
        """Take a step using the fourth-order explicit Runge-Kutta method.
        
        This requires more memory than a simple forward Euler step, and also
        more than a backward Euler step. It is, however, far more accurate
        and stable than forward Euler, and much faster than the backward
        Euler method, since there is no need to iteratively solve an implicit
        equation.
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
            
    def add_noise(self, fac):
        """Adds some random noise of a given order to the state matrices A
        This can be used to determine the influence of numerical innaccuracies
        on quantities such as observables.
        """
        for n in xrange(1, self.N + 1):
            for s in xrange(self.q[n]):
                self.A[n][s].real += (sp.rand(self.D[n - 1], self.D[n]) - 0.5) * 2 * fac
                self.A[n][s].imag += (sp.rand(self.D[n - 1], self.D[n]) - 0.5) * 2 * fac
                
    
    def calc_l(self, start=-1, finish=-1):
        """Updates the l matrices using the current state.
        Implements step 5 of the TDVP algorithm or, equivalently, eqn. (41).
        (arXiv:1103.0936v2 [cond-mat.str-el])
        """
        if start < 0:
            start = 1
        if finish < 0:
            finish = self.N
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
    
    def simple_renorm(self, update_r=True):
        """Renormalize the state by altering A[N] by a factor.
        
        We change A[N] only, which is a column vector because D[N] = 1, using a factor
        equivalent to an almost-gauge transformation where all G's are the identity, except
        G[N], which represents the factor. Almost means G[0] =/= G[N] (the norm is allowed to change).
        
        Requires that l is up to date. 
        
        Note that this generally breaks ON_R, because this changes r[N - 1] by the same factor.
        
        By default, this also updates the r matrices to reflect the change in A[N].
        
        Parameters
        ----------
        calc_r : bool
            Whether to call calc_r() after normalization (defaults to True).
        """
        norm = self.l[self.N][0, 0].real
        G_N = 1 / sp.sqrt(norm)
        
        self.A[self.N] *= G_N
        
        self.l[self.N][:] *= 1 / norm
        
        #We need to do this because we changed A[N]
        if update_r:
            for n in xrange(self.N):
                self.r[n] *= 1 / norm    
    
    def restore_RCF(self, start=-1, update_l=True, normalize=True, diag_l=True):
        """Use a gauge-transformation to restore right canonical form.
        
        Implements the conditions for right canonical form from sub-section
        3.1, theorem 1 of arXiv:quant-ph/0608197v2.
        
        This performs two 'almost' gauge transformations, where the 'almost'
        means we allow the norm to vary (if "normalize" = True).
        
        The last step (A[1]) is done diffently to the others since G[0],
        the gauge-transf. matrix, is just a number, which can be found more
        efficiently and accurately without using matrix methods.
        
        The last step (A[1]) is important because, if we have successfully made 
        r[1] = 1 in the previous steps, it fully determines the normalization 
        of the state via r[0] ( = l[N]).
        
        Optionally (normalize=False), the function will not attempt to make
        A[1] satisfy the orthonorm. condition, and will take G[0] = 1 = G[N],
        thus performing a pure gauge-transformation, but not ensuring complete
        canonical form.
        
        It is also possible to begin the process from a site n other than N,
        in case the sites > n are known to be in the desired form already.
        
        It is also possible to skip the diagonalization of the l's, such that
        only the right orthonormalization condition (r_n = eye) is met.
        
        By default, the l's are updated even if diag_l=False.
        
        FIXME: Currently, "start" only affects the ON_R stage!
        
        Parameters
        ----------
        start : int
            The rightmost site to start from (defaults to N)
        update_l : bool
            Whether to call calc_l() after completion (defaults to True)
        normalize : bool
            Whether to also attempt to enforce the condition for A[1], which normalizes the state.
        diag_l : bool
            Whether to put l in diagonal form (defaults to True)
        """   
        if start < 1:
            start = self.N
        
        G_n_i = sp.eye(self.D[start], dtype=self.typ) #This is actually just the number 1
        for n in reversed(xrange(2, start + 1)):
            self.r[n - 1], G_n_i, G_n = tm.restore_RCF_r(self.A[n], self.r[n], 
                                                         G_n_i,
                                           sanity_checks=self.sanity_checks)
        
        #Now do A[1]...
        #Apply the remaining G[1]^-1 from the previous step.
        for s in xrange(self.q[1]):                
            self.A[1][s] = m.mmul(self.A[1][s], G_n_i)
                    
        #Now finish off
        tm.eps_r_noop_inplace(self.r[1], self.A[1], self.A[1], out=self.r[0])
        
        if normalize:
            G0 = 1. / sp.sqrt(self.r[0].squeeze().real)
            self.A[1] *= G0
            self.r[0][:] = 1
            
            if self.sanity_checks:
                r0 = tm.eps_r_noop(self.r[1], self.A[1], self.A[1])
                if not sp.allclose(r0, 1, atol=1E-12, rtol=1E-12):
                    print "Sanity Fail in restore_RCF!: r_0 is bad / norm failure"
                
        if diag_l:
            G_nm1 = sp.eye(self.D[0], dtype=self.typ)
            for n in xrange(1, self.N):
                self.l[n], G_nm1, G_nm1_i = tm.restore_RCF_l(self.A[n],
                                                             self.l[n - 1],
                                                             G_nm1,
                                                             self.sanity_checks)
            
            #Apply remaining G_Nm1 to A[N]
            n = self.N
            for s in xrange(self.q[n]):                
                self.A[n][s] = m.mmul(G_nm1, self.A[n][s])
                
            #Deal with final, scalar l[N]
            tm.eps_l_noop_inplace(self.l[n - 1], self.A[n], self.A[n], out=self.l[n])
            
            if self.sanity_checks:
                if not sp.allclose(self.l[self.N].real, 1, atol=1E-12, rtol=1E-12):
                    print "Sanity Fail in restore_RCF!: l_N is bad / norm failure"
                    print "l_N = " + str(self.l[self.N].squeeze().real)
                
                for n in xrange(1, self.N + 1):
                    r_nm1 = tm.eps_r_noop(m.eyemat(self.D[n], self.typ), self.A[n], self.A[n])
                    if not sp.allclose(r_nm1, self.r[n - 1], atol=1E-12, rtol=1E-12):
                        print "Sanity Fail in restore_RCF!: r_%u is bad" % n
                    
            return True #FIXME: This OK?
        elif update_l:
            res = self.calc_l()
            return res
        else:
            return True
    
    def check_RCF(self):
        """Tests for right canonical form.
        Uses the criteria listed in sub-section 3.1, theorem 1 of arXiv:quant-ph/0608197v2.
        """
        rnsOK = True
        ls_trOK = True
        ls_herm = True
        ls_pos = True
        ls_diag = True
        
        for n in xrange(1, self.N + 1):
            rnsOK = rnsOK and sp.allclose(self.r[n], sp.eye(self.r[n].shape[0]), atol=self.eps*2, rtol=0)
            ls_herm = ls_herm and sp.allclose(self.l[n] - m.H(self.l[n]), 0, atol=self.eps*2)
            ls_trOK = ls_trOK and sp.allclose(sp.trace(self.l[n]), 1, atol=self.eps*2, rtol=0)
            ls_pos = ls_pos and all(la.eigvalsh(self.l[n]) > 0)
            ls_diag = ls_diag and sp.allclose(self.l[n], sp.diag(self.l[n].diagonal()))
        
        normOK = sp.allclose(self.l[self.N], 1., atol=self.eps, rtol=0)
        
        return (rnsOK, ls_trOK, ls_pos, ls_diag, normOK)
    
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
        op = lambda s, t: o(n, s, t)
        opv = sp.vectorize(op, otypes=[sp.complex128])
        opm = sp.fromfunction(opv, (self.q[n], self.q[n]))
        
        res = tm.eps_r_op_1s(self.r[n], self.A[n], self.A[n], opm)
        return  m.adot(self.l[n - 1], res)
        
    def expect_1s_cor(self, o1, o2, n1, n2):
        """Computes the correlation of two single site operators acting on two different sites.
        
        See expect_1s().
        
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
         
        return m.adot(self.l[n1 - 1], r_n)

    def density_1s(self, n):
        """Returns a reduced density matrix for a single site.
        
        Parameters
        ----------
        n1 : int
            The site number.
        """
        rho = sp.empty((self.q[n], self.q[n]), dtype=sp.complex128)
                    
        r_n = self.r[n]
        r_nm1 = sp.empty_like(self.r[n - 1])
        for s in xrange(self.q[n]):
            for t in xrange(self.q[n]):
                r_nm1 = m.mmul(self.A[n][t], r_n, m.H(self.A[n][s]))                
                rho[s, t] = m.adot(self.l[n - 1], r_nm1)
        return rho
        
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
                r_n2 = m.mmul(self.A[n2][t2], self.r[n2], m.H(self.A[n2][s2]))
                
                r_n = r_n2
                for n in reversed(xrange(n1 + 1, n2)):
                    r_n = tm.eps_r_noop(r_n, self.A[n], self.A[n])        
                    
                for s1 in xrange(self.q[n1]):
                    for t1 in xrange(self.q[n1]):
                        r_n1 = m.mmul(self.A[n1][t1], r_n, m.H(self.A[n1][s1]))
                        tmp = m.mmul(self.l[n1 - 1], r_n1)
                        rho[s1 * self.q[n1] + s2, t1 * self.q[n1] + t2] = tmp.trace()
        return rho
    
    def save_state(self, file):
        sp.save(file, self.A)
        
    def load_state(self, file):
        self.A = sp.load(file)