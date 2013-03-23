# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 17:29:27 2011

@author: Ashley Milsted

"""
import scipy as sp
import scipy.linalg as la
import matmul as m
import tdvp_common as tm
import copy

class EvoMPS_MPS_Generic(object):
    
    def __init__(self, N, D, q):
        """Creates a new EvoMPS_MPS_Generic object.
        
        This class implements basic operations on a generic MPS with
        open boundary conditions on a finite chain.
        
        Performs self.correct_bond_dimension().
        
        Sites are numbered 1 to N.
        self.A[n] is the parameter tensor for site n
        with shape == (q[n], D[n - 1], D[n]).
        
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
        """Initializes the state to a hard-coded full rank state with norm 1.
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
        """Randomizes the parameter tensors self.A.
        
        Parameters
        ----------
        do_update : bool (True)
            Whether to perform self.update() after randomizing.
        """
        for n in xrange(1, self.N + 1):
            self.A[n] = ((sp.rand(*self.A[n].shape) - 0.5) 
                         + 1.j * (sp.rand(*self.A[n].shape) - 0.5))
            self.A[n] /= la.norm(self.A[n])
        
        if do_update:
            self.update()

        
    def add_noise(self, fac, do_update=True):
        """Adds some random (white) noise of a given magnitude to the parameter 
        tensors A.
        
        Parameters
        ----------
        fac : number
            A factor determining the amplitude of the random noise.
        """
        for n in xrange(1, self.N + 1):
            self.A[n].real += (sp.rand(*self.A[n].shape) - 0.5) * 2 * fac
            self.A[n].imag += (sp.rand(*self.A[n].shape) - 0.5) * 2 * fac
            
        if do_update:
            self.update()
        
    def correct_bond_dimension(self):
        """Reduces bond dimensions to the maximum physically useful values.
        
        Bond dimensions will be adjusted where they are too high to be useful
        (when they would be higher than the corresponding maximum
        Schmidt ranks). The maximum value for D[n] is the minimum of the 
        dimensions of the two partial Hilbert spaces corresponding to a cut 
        between sites n and n + 1.
        """
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
                
    
    def update(self, restore_RCF=True, auto_truncate=True, restore_RCF_after_trunc=True):
        """Updates secondary quantities to reflect the state parameters self.A.
        
        Must be used after changing the parameters self.A before calculating
        physical quantities, such as expectation values.
        
        Also (optionally) restores the right canonical form.
        
        Parameters
        ----------
        restore_RCF : bool (True)
            Whether to restore right canonical form.
        auto_truncate : bool (True)
            Whether to automatically truncate the bond-dimension if
            rank-deficiency is detected. Requires restore_RCF.
        restore_RCF_after_trunc : bool (True)
            Whether to restore_RCF after truncation.
            
        Returns
        -------
        truncated : bool (only if auto_truncate == True)
            Whether truncation was performed.
        """
        assert restore_RCF or not auto_truncate, "auto_truncate requires restore_RCF"
        
        if restore_RCF:
            self.restore_RCF()
            if auto_truncate:
                return self.auto_truncate(restore_RCF=restore_RCF_after_trunc)
        else:
            self.calc_l()
            self.calc_r()
            self.simple_renorm()
    
    def calc_l(self, n_low=-1, n_high=-1):
        """Updates the l matrices to reflect the current state parameters self.A.
        
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
        """Updates the r matrices using the current state parameters self.A.
        
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
        G[N], which represents the factor. "Almost" means G[0] =/= G[N] (the norm is allowed to change).
        
        Requires that l is up to date. 
        
        Note that this generally breaks canonical form
        because we change r[N - 1] by the same factor.
        
        By default, this also updates the r matrices to reflect the change in A[N].
        
        Parameters
        ----------
        update_r : bool (True)
            Whether to call update all the r matrices to reflect the change.
        """
        norm = self.l[self.N][0, 0].real
        G_N = 1 / sp.sqrt(norm)
        
        self.A[self.N] *= G_N
        
        self.l[self.N][:] *= 1 / norm
        
        if update_r:
            for n in xrange(self.N):
                self.r[n] *= 1 / norm    
    
    def restore_RCF(self, update_l=True, normalize=True, diag_l=True):
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
        
        Parameters
        ----------
        update_l : bool
            Whether to call calc_l() after completion (defaults to True)
        normalize : bool
            Whether to also attempt to normalize the state.
        diag_l : bool
            Whether to put l in diagonal form (defaults to True)
        """   
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
                    r_nm1 = tm.eps_r_noop(self.r[n], self.A[n], self.A[n])
                    #r_nm1 = tm.eps_r_noop(m.eyemat(self.D[n], self.typ), self.A[n], self.A[n])
                    if not sp.allclose(r_nm1, self.r[n - 1], atol=1E-11, rtol=1E-11):
                        print "Sanity Fail in restore_RCF!: r_%u is bad (off by %g)" % (n, la.norm(r_nm1 - self.r[n - 1]))
                        print (r_nm1 - self.r[n - 1]).diagonal().real
        elif update_l:
            self.calc_l()
    
    def auto_truncate(self, restore_RCF=True, zero_tol=1E-15):
        """Automatically reduces the bond-dimension in case of rank-deficiency.
        
        Canonical form is required. Always perform self.restore_RCF() first!
        
        Parameters
        ----------
            restore_RCF : bool (True)
                Whether to restore canonical form after truncation.
            zero_tol : float
                Tolerance for interpretation of values as zero.
                
        Returns
        -------
            truncated : bool
                Whether truncation was performed.
        """
        new_D = self.D.copy()
        
        for n in xrange(1, self.N + 1):
            try:
                ldiag = self.l[n].diag
            except AttributeError:
                ldiag = self.l[n].diagonal()
            
            new_D[n] = sp.count_nonzero(abs(ldiag) > zero_tol)
        
        if not sp.all(new_D == self.D):
            self.truncate(new_D, update_lr=not restore_RCF)
        
            if restore_RCF:
                self.restore_RCF()
                                        
            return True
        else:
            return False
            
        
    def truncate(self, new_D, update_lr=True):
        """Reduces the bond-dimensions by truncating the least-significant Schmidt vectors.
        
        The parameters must be in canonical form to ensure that
        the discarded parameters correspond to the smallest Schmidt 
        coefficients. Always perform self.restore_RCF() first!
        
        Each bond-dimension can either be reduced or left unchanged.
        
        The resulting parameters self.A will not generally have canonical 
        form after truncation.
        
        Parameters
        ----------
        new_D : list or ndarray of int
            The new bond-dimensions in a vector of length N + 1.
        update_lr : bool (True)
            Whether to update self.l and self.r after truncation (turn off if you plan to update them yourself).
        """
        new_D = sp.array(new_D)
        assert new_D.shape == self.D.shape, "new_D must have same shape as self.D"
        assert sp.all(new_D <= self.D), "new bond-dimensions must be less than or equal to current dimensions"
    
        last_trunc = sp.argwhere(self.D - new_D).max()
        
        if update_lr:
            tmp_r0 = self.r[0]
            tmp_l = self.l
            
        tmp_A = self.A
        
        self.D = new_D
        self._init_arrays()

        for n in xrange(1, self.N + 1):
            self.A[n][:] = tmp_A[n][:, -self.D[n - 1]:, -self.D[n]:]
        
        if update_lr:
            self.r[0] = tmp_r0
            
            for n in xrange(1, self.N):
                self.l[n] = m.simple_diag_matrix(tmp_l[n].diag[-self.D[n]:])
                
            self.l[self.N] = tmp_l[self.N]
            
            for n in xrange(self.N - 1, last_trunc - 1, - 1):
                self.r[n] = m.eyemat(self.D[n])
                
            self.calc_r(n_high=last_trunc - 1)
            
            self.simple_renorm()
        
    
    def check_RCF(self):
        """Tests for right canonical form.
        
        Uses the criteria listed in sub-section 3.1, theorem 1 of arXiv:quant-ph/0608197v2.
        
        This is a consistency check mainly intended for debugging purposes.
        
        FIXME: The tolerances appear to be too tight!
        
        Returns
        -------
        (rnsOK, ls_trOK, ls_pos, ls_diag, normOK) : tuple of bool
            rnsOK: Right orthonormalization is fullfilled (self.r[n] = eye)
            ls_trOK: all self.l[n] have trace 1
            ls_pos: all self.l[n] are positive-definite
            ls_diag: all self.l[n] are diagonal
            normOK: the state it normalized
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
        
        The operator should be a q[n] x q[n] matrix or generating function 
        such that op[s, t] or op(s, t) equals <s|op|t>.
        
        The state must be up-to-date -- see self.update()!
        
        Parameters
        ----------
        op : ndarray or callable
            The operator.
        n : int
            The site number (1 <= n <= N).
            
        Returns
        -------
        expval : floating point number
            The expectation value (data type may be complex)
        """        
        if callable(op):
            op = sp.vectorize(op, otypes=[sp.complex128])
            op = sp.fromfunction(op, (self.q[n], self.q[n]))
            
        res = tm.eps_r_op_1s(self.r[n], self.A[n], self.A[n], op)
        return  m.adot(self.l[n - 1], res)
        
    def expect_2s(self, op, n):
        """Computes the expectation value of a nearest-neighbour two-site operator.
        
        The operator should be a q[n] x q[n + 1] x q[n] x q[n + 1] array 
        such that op[s, t, u, v] = <st|op|uv> or a function of the form 
        op(s, t, u, v) = <st|op|uv>.
        
        The state must be up-to-date -- see self.update()!
        
        Parameters
        ----------
        op : ndarray or callable
            The operator array or function.
        n : int
            The leftmost site number (operator acts on n, n + 1).
            
        Returns
        -------
        expval : floating point number
            The expectation value (data type may be complex)
        """
        A = self.A[n]
        Ap1 = self.A[n + 1]
        AA = tm.calc_AA(A, Ap1)
        
        if callable(op):
            op = sp.vectorize(op, otypes=[sp.complex128])
            op = sp.fromfunction(op, (A.shape[0], Ap1.shape[0], A.shape[0], Ap1.shape[0]))
            
        C = tm.calc_C_mat_op_AA(op, AA)
        res = tm.eps_r_op_2s_C12_AA34(self.r[n + 1], C, AA)
        return m.adot(self.l[n - 1], res)
        
    def expect_1s_1s(self, op1, op2, n1, n2):
        """Computes the expectation value of two single site operators acting 
        on two different sites.
        
        The result is < op1 op2 >.
        
        See expect_1s().
        
        Requires n1 < n2.
        
        The state must be up-to-date -- see self.update()!
        
        Parameters
        ----------
        op1 : ndarray or callable
            The first operator, acting on the first site.
        op2 : ndarray or callable
            The second operator, acting on the second site.
        n1 : int
            The site number of the first site.
        n2 : int
            The site number of the second site (must be > n1).
            
        Returns
        -------
        expval : floating point number
            The expectation value (data type may be complex)
        """        
        if callable(op1):
            op1 = sp.vectorize(op1, otypes=[sp.complex128])
            op1 = sp.fromfunction(op1, (self.q[n1], self.q[n1]))
        
        if callable(op2):
            op2 = sp.vectorize(op2, otypes=[sp.complex128])
            op2 = sp.fromfunction(op2, (self.q[n2], self.q[n2])) 
        
        r_n = tm.eps_r_op_1s(self.r[n2], self.A[n2], self.A[n2], op2)

        for n in reversed(xrange(n1 + 1, n2)):
            r_n = tm.eps_r_noop(r_n, self.A[n], self.A[n])

        r_n = tm.eps_r_op_1s(r_n, self.A[n1], self.A[n1], op1)
         
        return m.adot(self.l[n1 - 1], r_n)

    def density_1s(self, n):
        """Returns a reduced density matrix for a single site.
        
        The site number basis is used: rho[s, t] 
        with 0 <= s, t < q[n].
        
        The state must be up-to-date -- see self.update()!
        
        Parameters
        ----------
        n1 : int
            The site number.
            
        Returns
        -------
        rho : ndarray
            Reduced density matrix in the number basis.
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
        """Returns a reduced density matrix for a pair of (seperated) sites.
        
        The site number basis is used: rho[s * q[n1] + u, t * q[n1] + v]
        with 0 <= s, t < q[n1] and 0 <= u, v < q[n2].
        
        The state must be up-to-date -- see self.update()!
        
        Parameters
        ----------
        n1 : int
            The site number of the first site.
        n2 : int
            The site number of the second site (must be > n1).
            
        Returns
        -------
        rho : ndarray
            Reduced density matrix in the number basis.
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
                        tmp = m.adot(self.l[n1 - 1], r_n1)
                        rho[s1 * self.q[n1] + s2, t1 * self.q[n1] + t2] = tmp
        return rho
    
    def save_state(self, file):
        """Saves the parameter tensors self.A to a file. 
        
        Uses numpy binary format.
        
        Parameters
        ----------
        file ; path or file
            The file to save the state into.
        """
        sp.save(file, self.A)
        
    def load_state(self, file):
        """Loads the parameter tensors self.A from a file.

        The saved state must contain the right number of tensors with
        the correct shape corresponding to self.N and self.q.
        self.D will be recovered from the saved state.

        Parameters
        ----------
        file ; path or file
            The file to load the state from.
        """
        tmp_A = sp.load(file)

        self.D[0] = 1
        for n in xrange(self.N):
            self.D[n + 1] = tmp_A[n + 1].shape[2]
        self._init_arrays()
        self.A = tmp_A
