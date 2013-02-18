# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 17:29:27 2011

@author: Ashley Milsted

TODO:
    - Find a way to randomize the starting state without losing rank.

"""
import scipy as sp
import scipy.linalg as la
import matmul as m
import tdvp_common as tm

class EvoMPS_MPS_Generic(object):
    
    def __init__(self, N, D, q):
        """Creates a new EvoMPS_MPS_Generic object.
        
        This class implements basic operations on a generic MPS with
        open boundary conditions in a finite setting.
        
        Bond dimensions will be adjusted where they are too high to be useful.
        
        Sites are numbered 1 to N. The following objects are created:
            .A : ndarray
                 Parameter matrices indexed by site (entries 1 to N)
            .l : ndarray
                 Left ancilla 'density matrices' (entries 0 to N)
            .r : ndarray
                 Right ancilla 'density matrices' (entries 0 to N)
        
        Parameters
        ----------
        N : int
            The number of lattice sites.
        D : ndarray
            A 1d array, length N + 1, of integers indicating the desired 
            bond dimensions.
        q : ndarray
            A 1d array, length N + 1, of integers indicating the 
            dimension of the hilbert space for each site. 
            Entry 0 is ignored (there is no site 0).
         
        """
        
        self.odr = 'C'
        self.typ = sp.complex128
        
        self.sanity_checks = True        
        
        self.eps = sp.finfo(self.typ).eps
        
        self.N = N
        self.D = sp.array(D)
        self.q = sp.array(q)        

        if (self.D.ndim != 1) or (self.q.ndim != 1):
            raise ValueError('D and q must be 1-dimensional!')
            
        if (self.D.shape[0] != N + 1) or (self.q.shape[0] != N + 1):
            raise ValueError('D and q must have length N + 1')

        self.correct_bond_dimension()
        
        self._init_arrays()
        
        self.initialize_state()
    
    def _init_arrays(self):
        self.A = sp.empty((self.N + 1), dtype=sp.ndarray) #Elements 1..N
        
        self.r = sp.empty((self.N + 1), dtype=sp.ndarray) #Elements 0..N
        self.l = sp.empty((self.N + 1), dtype=sp.ndarray)        
        
        self.r[0] = sp.zeros((self.D[0], self.D[0]), dtype=self.typ, order=self.odr)  
        self.l[0] = sp.eye(self.D[0], self.D[0], dtype=self.typ).copy(order=self.odr) #Already set the 0th element (not a dummy)    
    
        for n in xrange(1, self.N + 1):
            self.r[n] = sp.zeros((self.D[n], self.D[n]), dtype=self.typ, order=self.odr)
            self.l[n] = sp.zeros((self.D[n], self.D[n]), dtype=self.typ, order=self.odr)
            self.A[n] = sp.empty((self.q[n], self.D[n - 1], self.D[n]), dtype=self.typ, order=self.odr)
            
        sp.fill_diagonal(self.r[self.N], 1.)        
    
    def initialize_state(self):
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
    
    
    def randomize(self, do_update=True):
        """Set A's randomly
        """
        for n in xrange(1, self.N + 1):
            self.A[n] = ((sp.rand(*self.A[n].shape) - 0.5) 
                         + 1.j * (sp.rand(*self.A[n].shape) - 0.5))
            self.A[n] /= la.norm(self.A[n])
        
        if do_update:
            self.update()

        
    def add_noise(self, fac, do_update=True):
        """Adds some random noise of a given order to the state matrices A
        This can be used to determine the influence of numerical innaccuracies
        on quantities such as observables.
        """
        for n in xrange(1, self.N + 1):
            self.A[n].real += (sp.rand(*self.A[n].shape) - 0.5) * 2 * fac
            self.A[n].imag += (sp.rand(*self.A[n].shape) - 0.5) * 2 * fac
            
        if do_update:
            self.update()
        
    def correct_bond_dimension(self):
        self.D[0] = 1
        self.D[self.N] = 1

        qacc = 1
        for n in xrange(self.N - 1, -1, -1):
            if qacc < self.D.max(): #Avoid overflow!
                qacc *= self.q[n + 1]

            if self.D[n] > qacc:
                self.D[n] = qacc
                
        qacc = 1
        for n in xrange(1, self.N + 1):
            if qacc < self.D.max(): #Avoid overflow!
                qacc *= self.q[n]

            if self.D[n] > qacc:
                self.D[n] = qacc
                
    
    def update(self, restore_RCF=True):
        if restore_RCF:
            self.restore_RCF()
        else:
            self.calc_l()
            self.calc_r()
            self.simple_renorm()
                
    
    def calc_l(self, n_low=-1, n_high=-1):
        """Updates the l matrices using the current state.
        Implements step 5 of the TDVP algorithm or, equivalently, eqn. (41).
        (arXiv:1103.0936v2 [cond-mat.str-el])
        """
        if n_low < 0:
            n_low = 1
        if n_high < 0:
            n_high = self.N
        for n in xrange(n_low, n_high + 1):
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
        for n in xrange(n_high, n_low - 1, -1):
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
        update_r : bool
            Whether to call update all the r matrices to reflect the change.
        """
        norm = self.l[self.N][0, 0].real
        G_N = 1 / sp.sqrt(norm)
        
        self.A[self.N] *= G_N
        
        self.l[self.N][:] *= 1 / norm
        
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
        for n in xrange(start, 1, -1):
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
            ls_trOK = ls_trOK and sp.allclose(sp.trace(self.l[n]), 1, atol=self.eps*1000, rtol=0)
            ls_pos = ls_pos and all(la.eigvalsh(self.l[n]) > 0)
            ls_diag = ls_diag and sp.allclose(self.l[n], sp.diag(self.l[n].diagonal()))
        
        normOK = sp.allclose(self.l[self.N], 1., atol=self.eps*1000, rtol=0)
        
        return (rnsOK, ls_trOK, ls_pos, ls_diag, normOK)
    
    def expect_1s(self, op, n):
        """Computes the expectation value of a single-site operator.
        
        The operator should be a q[n] x q[n] matrix.
        
        Assumes that the state is normalized.
        
        Parameters
        ----------
        o : ndarray or callable
            The operator matrix or function.
        n : int
            The site number.
        """        
        if callable(op):
            op = sp.vectorize(op, otypes=[sp.complex128])
            op = sp.fromfunction(op, (self.q[n], self.q[n + 1], self.q[n], self.q[n + 1]))
            
        res = tm.eps_r_op_1s(self.r[n], self.A[n], self.A[n], op)
        return  m.adot(self.l[n - 1], res)
        
    def expect_1s_cor(self, o1, o2, n1, n2):
        """Computes the correlation of two single site operators acting on two different sites.
        
        See expect_1s().
        
        n1 must be smaller than n2.
        
        Assumes that the state is normalized.
        
        Parameters
        ----------
        o1 : ndarray
            The first operator, acting on the first site.
        o2 : ndarray
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