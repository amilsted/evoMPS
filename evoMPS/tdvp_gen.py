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
import tdvp_common as tm
from mps_gen import EvoMPS_MPS_Generic

class EvoMPS_TDVP_Generic(EvoMPS_MPS_Generic):
            
    def __init__(self, N, D, q, h_nn, h_ext=None):
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
        
        super(EvoMPS_TDVP_Generic, self).__init__(N, D, q)
        
        self.h_nn = h_nn
        self.h_ext = h_ext
        
        #Make indicies correspond to the thesis
        self.K = sp.empty((self.N + 1), dtype=sp.ndarray) #Elements 1..N
        self.C = sp.empty((self.N), dtype=sp.ndarray) #Elements 1..N-1 

        for n in xrange(1, self.N + 1):
            self.K[n] = sp.zeros((self.D[n-1], self.D[n-1]), dtype=self.typ, order=self.odr)    
            if n < self.N:
                self.C[n] = sp.empty((self.q[n], self.q[n+1], self.D[n-1], self.D[n+1]), dtype=self.typ, order=self.odr)
        
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
    
    def update(self, restore_RCF=True):
        super(EvoMPS_TDVP_Generic, self).update(restore_RCF)
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
            x_subpart = sp.zeros((self.A[n].shape[1], Vsh.shape[2]), dtype=self.typ)
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
        