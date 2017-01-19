#!/usr/bin/env python
# -*- coding: utf-8 -*-


import scipy as sp
import scipy.linalg as la
import mps_gen as mg
import tdvp_common as tm
import matmul as mm
import mps_gen as mps
import tdvp_gen as TDVP
import copy

class EvoMPS_TDVP_Generic_Dissipative_slower(TDVP.EvoMPS_TDVP_Generic):
    """ Class derived from TDVP.EvoMPS_TDVP_Generic.
    Extends it by adding dissipative Monte-Carlo evolution for one-side or
    two-site-lindblad dissipations.
    
    Methods:
    ----------
    take_step_dissipative(dt, l_nns)
        Performs dissipative and unitary evolution according to global
        hamiltonian definition and list of lindblads for single-site lindblads.  
    """
    
    def take_step_dissipative(self, dt, linds):
        """Advances real time by dt for an open system governed by Lindblad dynamics.
        
        This advances time along an individual pure-state trajectory, 
        or sample, making up part of an ensemble that represents a mixed state.
        
        Each pure sample is governed by a stochastic differential equation (SDE)
        composed of a deterministic "drift" part and a randomized "diffusion" part.
        
        The system Hamiltonian determines the drift part, while the Lindblad operators
        determine the diffusion part.
        
        Each Lindblad operator Lj must be a single, local operator with the same dimensions
        as the Hamiltonian terms. They must be supplied as tuples, where (n, Lj) is the operator
        Lj acting on sites n to n + [the range of Lj].
        
        linds = [(n, L1), (m, L2), ...]
        
        Parameters
        ----------
        dt : real
            The step size (smaller step sizes result in smaller errors)
        linds : Sequence of tuples (int, ndarray)
            
        """
        nL = len(linds)
        
        #Use precomputed C and K to get the Hamiltonian contribution
        B_H = self.calc_B()
        
        ham = self.ham
        
        q = self.q
        
        if self.ham_sites == 2:
            Ls = [(n, L, L.reshape((q[n]*q[n+1], q[n]*q[n+1])), self.expect_2s(L, n)) for (n, L) in linds]
        elif self.ham_sites == 3:
            Ls = [(n, L, L.reshape((q[n]*q[n+1]*q[n+2], q[n]*q[n+1]*q[n+2])), self.expect_3s(L, n)) for (n, L) in linds]
        else:
            assert False, "Range of Hamiltonian terms must be 2 or 3 sites."
        
        Leffs = []
        for (n, L, Lr, Le) in Ls:
            u = sp.random.normal(0, sp.sqrt(dt), (2,))
            W = (u[0] + 1.j * u[1]) / sp.sqrt(2)
            
            Leff = -0.5 * Lr.conj().T.dot(Lr).reshape(L.shape)
            Leff += (sp.conj(Le) + W/dt) * L
            
            Leffs.append((n, Leff))
                
        #An alternative approach is to put the Hamiltonian and dissipative parts together,
        #and compute only one tangent vector. This is the most efficient way, but it would
        #require more modifications to the class to avoid computing unneeded stuff.
        #HL = [-1.j * hn if hn is not None else None for hn in ham]
        
        HL = [None] * (self.N + 1)
        for (n, Leff) in Leffs:
            if HL[n] is None:
                HL[n] = Leff
            else:
                HL[n] += Leff
        
        #Compute the combined dissipative contribution. 
        #C is only computed where it is nonzero, so this should be fairly quick.
        self.ham = HL
        self.calc_C()
        self.calc_K()
        B_Leff = self.calc_B()
        
        self.ham = ham
        
        for n in xrange(1, self.N):
            if B_Leff[n] is not None:
                self.A[n] += dt * B_Leff[n]
            if B_H[n] is not None:
                self.A[n] += -1.j * dt * B_H[n]
        
    def calc_B_1s_diss(self,op,n):
        """Applies a single-site operator to a single site and returns
        the parameter tensor for that site after the change with the
        change in the norm of the state projected out.
        
        Parameters
        ----------
        op : ndarray or callable
            The single-site operator. See self.expect_1s().
        n: int
            The site to apply the operator to.
        """
        if callable(op):
            op = sp.vectorize(op, otypes=[sp.complex128])
            op = sp.fromfunction(op, (self.q[n], self.q[n]))
            
        newAn = sp.zeros_like(self.A[n])
        
        for s in xrange(self.q[n]):
            for t in xrange(self.q[n]):
                newAn[s] += self.A[n][t] * op[s, t]
                
        r_nm1 = TDVP.tm.eps_r_noop(self.r[n], newAn, newAn)
        ev = mm.adot(self.l[n - 1], r_nm1)

        newAn -= ev * self.A[n] #norm-fixing
        
        return newAn
        
class EvoMPS_TDVP_Generic_Dissipative(TDVP.EvoMPS_TDVP_Generic):
    """ Class derived from TDVP.EvoMPS_TDVP_Generic.
    Extends it by adding stochastic integration of the state
    as part of an ensemble of trajectories evolving under a Lindblad master equation.
    """
    
    def __init__(self, N, D, q, ham, linds, ham_sites=None):
        """Creates a new EvoMPS_TDVP_Generic_Dissipative object.
        
        This class implements the time-dependent variational principle (TDVP) for
        matrix product states (MPS) of a finite spin chain with open boundary
        conditions.
        
        It is derived from EvoMPS_MPS_Generic, which implements basic operations
        on the state, adding the ability to integrate the TDVP flow equations
        given a nearest-neighbour Hamiltonian.
        
        Performs EvoMPS_MPS_Generic.__init__().
        
        Sites are numbered 1 to N.
        self.A[n] is the parameter tensor for site n
        with shape == (q[n], D[n - 1], D[n]).
        
        Each Lindblad operator Lj must be a single, local operator with the same dimensions
        as the Hamiltonian terms. They must be supplied as tuples, where (n, Lj) is the operator
        Lj acting on sites n to n + [the range of Lj].
        
        linds = [(n, L1), (m, L2), ...]
        
        
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
        ham : array or callable
            Hamiltonian term for each site ham(n, *physical_indices) or 
            ham[n][*physical indices] for site n.
        linds : list of tuples (int, ndarray)
            Lindblad operators (n, op) specifying a local operator as an
            ndarray and the first site n on which it acts.
            The Lindblad operators must have the same range as the local Hamiltonian terms.
        """       

        self.linds = linds
        """The Lindblad operators."""
           
        super(EvoMPS_TDVP_Generic_Dissipative, self).__init__(N, D, q, ham, ham_sites=ham_sites)
    
    
    def update(self, restore_CF=True, normalize=True, auto_truncate=False, restore_CF_after_trunc=True):
        """Updates secondary quantities to reflect the state parameters self.A.
        
        Must be used after taking a step or otherwise changing the 
        parameters self.A before calculating
        physical quantities or taking the next step.
        
        Also (optionally) restores the canonical form.
        
        Parameters
        ----------
        restore_CF : bool
            Whether to restore canonical form.
        normalize : bool
            Whether to normalize the state in case restore_CF is False.
        auto_truncate : bool
            Whether to automatically truncate the bond-dimension if
            rank-deficiency is detected. Requires restore_CF.
        restore_CF_after_trunc : bool
            Whether to restore_CF after truncation.

        Returns
        -------
        truncated : bool (only if auto_truncate == True)
            Whether truncation was performed.
        """
        #Call the MPS method because we don't want to do the additional updates of EvoMPS_TDVP_Generic here.
        return mps.EvoMPS_MPS_Generic.update(self, restore_CF=restore_CF, 
                                                    normalize=normalize,
                                                    auto_truncate=auto_truncate,
                                                    restore_CF_after_trunc=restore_CF_after_trunc)
                                                        
        
    
    def take_step_dissipative(self, dt):
        """Advances real time by dt for an open system governed by Lindblad dynamics.
        
        This advances time along an individual pure-state trajectory, 
        or sample, making up part of an ensemble that represents a mixed state.
        
        Each pure sample is governed by a stochastic differential equation (SDE)
        composed of a deterministic "drift" part and a randomized "diffusion" part.
        
        The system Hamiltonian determines the drift part and the Lindblad operators
        determine the diffusion part.
        
        This method implements a single step of Euler-Maruyama integration of the SDE
        governing the trajectory of the present state.
        
        Parameters
        ----------
        dt : real
            The step size (smaller step sizes result in smaller errors)
        """
        nL = len(self.linds)
        
        ham = self.ham
        
        q = self.q
        
        #Reshape the Lindblad operators and compute their expectation values.
        if self.ham_sites == 2:
            Ls = [(n, L, L.reshape((q[n]*q[n+1], q[n]*q[n+1])), self.expect_2s(L, n)) for (n, L) in self.linds]
        elif self.ham_sites == 3:
            Ls = [(n, L, L.reshape((q[n]*q[n+1]*q[n+2], q[n]*q[n+1]*q[n+2])), self.expect_3s(L, n)) for (n, L) in self.linds]
        else:
            assert False, "Range of Hamiltonian terms must be 2 or 3 sites."
        
        Leffs = []
        for (n, L, Lr, Le) in Ls:
            #Sample from a complex Wiener process
            u = sp.random.normal(0, sp.sqrt(dt), (2,))
            W = (u[0] + 1.j * u[1]) / sp.sqrt(2)
            
            Leff = -0.5 * Lr.conj().T.dot(Lr).reshape(L.shape)
            
            #Add the diffusion terms to the effective Hamiltonian by dividing by the step length
            Leff += (sp.conj(Le) + W/dt) * L
            
            Leffs.append((n, Leff))
                
        HL = [-1.j * hn if hn is not None else None for hn in ham]
        for (n, Leff) in Leffs:
            if HL[n] is None:
                HL[n] = Leff
            else:
                HL[n] += Leff
        
        #Compute the combined dissipative contribution. 
        #C is only computed where it is nonzero, so this should be fairly quick.
        self.ham = HL
        self.calc_C()
        self.calc_K()
        B_Leff = self.calc_B()
        
        self.ham = ham
        
        #The following are not meaningful, since we have replaced the Hamiltonian with
        #an effective one, the expectation value of which has no obvious interpretation.
        self.h_expect.fill(sp.NaN)
        self.H_expect = sp.NaN
        
        #Note that self.eta correctly represents the norm of the effective evolution tangent vector.
        
        #Advance the state!
        for n in xrange(1, self.N):
            if B_Leff[n] is not None:
                self.A[n] += dt * B_Leff[n]
                
        
    def calc_B_1s_diss(self,op,n):
        """Applies a single-site operator to a single site and returns
        the parameter tensor for that site after the change with the
        change in the norm of the state projected out.
        
        Parameters
        ----------
        op : ndarray or callable
            The single-site operator. See self.expect_1s().
        n: int
            The site to apply the operator to.
        """
        if callable(op):
            op = sp.vectorize(op, otypes=[sp.complex128])
            op = sp.fromfunction(op, (self.q[n], self.q[n]))
            
        newAn = sp.zeros_like(self.A[n])
        
        for s in xrange(self.q[n]):
            for t in xrange(self.q[n]):
                newAn[s] += self.A[n][t] * op[s, t]
                
        r_nm1 = TDVP.tm.eps_r_noop(self.r[n], newAn, newAn)
        ev = mm.adot(self.l[n - 1], r_nm1)

        newAn -= ev * self.A[n] #norm-fixing
        
        return newAn

class EvoMPS_TDVP_Generic_Dissipative_lowmem(mps.EvoMPS_MPS_Generic):
    def __init__(self, N, D, q, ham, linds):
        """Creates a new EvoMPS_TDVP_Generic object.
        
        This class implements the time-dependent variational principle (TDVP) for
        matrix product states (MPS) of a finite spin chain with open boundary
        conditions.
        
        It is derived from EvoMPS_MPS_Generic, which implements basic operations
        on the state, adding the ability to integrate the TDVP flow equations
        given a nearest-neighbour Hamiltonian.
        
        Performs EvoMPS_MPS_Generic.__init__().
        
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
        ham : list of arrays
            Hamiltonian term for each site ham[n][*physical indices].
        linds : list of tuples (int, ndarray)
            Lindblad operators (n, op) specifying a local operator as an
            ndarray and the first site n on which it acts.
            The Lindblad operators must have the same range as the local Hamiltonian terms.
        """       

        self.ham = ham
        """The Hamiltonian. Can be changed, for example, to perform
           a quench."""
           
        self.linds = linds
        """Lindblas operators."""
        
        self.ham_sites = len(ham[1].shape) / 2
        
        if not (self.ham_sites == 2 or self.ham_sites == 3):
            raise ValueError("Only 2 or 3 site Hamiltonian terms supported!")
            
        super(EvoMPS_TDVP_Generic_Dissipative_lowmem, self).__init__(N, D, q)
        
        
    def take_step(self, dt):
        nL = len(self.linds)
        
        #Use precomputed C and K to get the Hamiltonian contribution
        B_H = self.calc_B()
        
        ham = self.ham
        
        q = self.q
        
        if self.ham_sites == 2:
            Ls = [(n, L, L.reshape((q[n]*q[n+1], q[n]*q[n+1])), self.expect_2s(L, n)) for (n, L) in self.linds]
        elif self.ham_sites == 3:
            Ls = [(n, L, L.reshape((q[n]*q[n+1]*q[n+2], q[n]*q[n+1]*q[n+2])), self.expect_3s(L, n)) for (n, L) in self.linds]
        else:
            assert False, "Range of Hamiltonian terms must be 2 or 3 sites."
        
        Leffs = []
        for (n, L, Lr, Le) in Ls:
            u = sp.random.normal(0, sp.sqrt(dt), (2,))
            W = (u[0] + 1.j * u[1]) / sp.sqrt(2)
            
            Leff = -0.5 * Lr.conj().T.dot(Lr).reshape(L.shape)
            Leff += (sp.conj(Le) + W/dt) * L
            
            Leffs.append((n, Leff))
        
        HL = [-1.j * hn if hn is not None else None for hn in ham]
        for (n, Leff) in Leffs:
            if HL[n] is None:
                HL[n] = Leff
            else:
                HL[n] += Leff
                
        Cp1 = None
        Kp1 = sp.zeros_like(self.r[self.N])
        for n in xrange(self.N, 0):
            if HL[n] is None:
                Kn, ex = (tm.eps_r_noop(Kp1, self.A[n], self.A[n]), 0)
            else:
                if self.ham_sites == 2 and self.N - n + 1 >= 2:
                    AAn = tm.calc_AA(self.A[n], self.A[n+1])
                    Cn = tm.calc_C_mat_op_AA(HL[n], AAn)
                    Kn, ex = tm.calc_K(Kp1, Cn, self.l[n - 1], 
                                       self.r[n + 1], self.A[n], AAn)
                elif self.ham_sites == 3 and self.N - n + 1 >= 3:
                    AAn = tm.calc_AAA(self.A[n], self.A[n+1], self.A[n+2])
                    Cn = tm.calc_C_3s_mat_op_AAA(HL[n], AAAn) 
                    Kn, ex = tm.calc_K_3s(Kp1, Cn, self.l[n - 1], 
                                          self.r[n + 2], self.A[n], AAAn)
                else:
                    Kn = None
                    
            lm1_s, lm1_si, rn_s, rn_si = tm.calc_l_r_roots(self.l[n - 1], self.r[n], 
                                                           zero_tol=self.zero_tol,
                                                           sanity_checks=self.sanity_checks,
                                                           sc_data=('site', n))
            
            Vrhn = tm.calc_Vsh(self.A[n], rn_s, sanity_checks=self.sanity_checks)
            
            #WIP: Must keep track of a few C's to make this work
            if self.ham_sites == 2:
                xp1 = tm.calc_x(Kp2, Cp1, Cn, rp2,
                                lm1, self.A[n], self.A[n+1], self.A[n+2],
                                ln_s, ln_si, rp1_s, rp1_si, Vshp1)
            else:
                xp2 = tm.calc_x_3s(Kp3, Cp2, Cp1, Cn, rp3, rp4, ln, 
                                   lm1, AAp1, Ap1, self.A[n], Ap3, Ap3Ap4,
                                   lp1_s, lp1_si, rp2_s, rp2_si, Vshp2)
            
            Bn = self._calc_B_l_n(n, set_eta=set_eta, l_s_m1=l_s_m1, l_si_m1=l_si_m1, r_s=r_s, r_si=r_si, Vlh=Vlh)