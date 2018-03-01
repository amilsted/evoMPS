#!/usr/bin/env python
# -*- coding: utf-8 -*-


import scipy as sp
import scipy.linalg as la
from . import mps_gen as mg
from . import tdvp_common as tm
from . import matmul as mm
from . import mps_gen as mps
from . import tdvp_gen as TDVP
import copy
        
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
                                                        
        
    
    def take_step_dissipative(self, dt, save_memory=False, calc_Y_2s=False, 
                  dynexp=False, dD_max=16, D_max=0, sv_tol=1E-14):
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
                
        #Build the effective Hamiltonian
        HL = [-1.j * hn if hn is not None else None for hn in self.ham]
        for (n, Leff) in Leffs:
            if HL[n] is None:
                HL[n] = Leff
            else:
                HL[n] += Leff
        
        #Compute the combined unitary and dissipative contributions using the effective Hamiltonian. 
        ham_tmp = self.ham
        self.ham = HL
        self.calc_C()
        self.calc_K()
        
        #Let take_step compute the tangent vector
        self.take_step(-dt, save_memory=save_memory, calc_Y_2s=calc_Y_2s, 
                        dynexp=dynexp, dD_max=dD_max, D_max=D_max, sv_tol=sv_tol)
        
        #Restore the Hamiltonian to the ham field.
        self.ham = ham_tmp
        
        #The following are not meaningful, since we have replaced the Hamiltonian with
        #an effective one, the expectation value of which has no obvious interpretation.
        self.h_expect.fill(sp.NaN)
        self.H_expect = sp.NaN
                
        
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
        
        for s in range(self.q[n]):
            for t in range(self.q[n]):
                newAn[s] += self.A[n][t] * op[s, t]
                
        r_nm1 = TDVP.tm.eps_r_noop(self.r[n], newAn, newAn)
        ev = mm.adot(self.l[n - 1], r_nm1)

        newAn -= ev * self.A[n] #norm-fixing
        
        return newAn
