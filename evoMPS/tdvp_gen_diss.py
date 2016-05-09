#!/usr/bin/env python
# -*- coding: utf-8 -*-


import scipy as sp
import scipy.linalg as la
import mps_gen as mg
import tdvp_common as tm
import matmul as mm
import tdvp_gen as TDVP

class EvoMPS_TDVP_Generic_Dissipative(TDVP.EvoMPS_TDVP_Generic):
    """ Class derived from TDVP.EvoMPS_TDVP_Generic.
    Extends it by adding dissipative Monte-Carlo evolution for one-side or
    two-site-lindblad dissipations.
    
    Methods:
    ----------
    take_step_dissipative(dt, l_nns)
        Performs dissipative and unitary evolution according to global
        hamiltonian definition and list of lindblads for single-site lindblads.
    take_step_dissipative_nonlocal(dt, MC, l_nns)
        Performs dissipative and unitary evolution according to global
        hamiltonian definition and list of lindblads for multi-site lindblads.
        WARNING: Implementation incomplete.
    apply_op_1s_diss(op,n)
        Applys a single-site operator to site n.    
    """
    
    def take_step_dissipative(self, dt, linds):
        """Advances real time by dt for an open system governed by Lindblad dynamics.
        
        This advances time along an individual pure-state trajectory, 
        or sample, making up part of an ensemble that represents a mixed state.
        
        Each pure sample is governed by a stochastic differential equation (SDE)
        composed of a deterministic "drift" part and a randomized "diffusion" part.
        
        The system Hamiltonian determines the drift part, while the Lindblad operators
        determine the diffusion part.
        
        The Lindblad operators must be supplied in the same form as the system Hamiltonian.
        In other words, each Lindblad operator must be a sum of nearest-neighbour 
        (or next-nearest neighbour) terms.
        
        linds[j] = [None, Lj_12, Lj_23, Lj_34, ..., Lj_(N-1)N]
        
        Parameters
        ----------
        dt : real
            The step size (smaller step sizes result in smaller errors)
        lines : Sequence of sequences of ndarrays
            Each entry in linds represents a Lindblad operator with the same form as the Hamiltonian.
            In other words, linds[j] is a sequence of local terms with linds[j][n] acting on sites n and (n+1) (and (n+2)).
            
        """
        nL = len(linds)
        
        #Compute tangent vector arising from system Hamiltonian evolution
        B_H = self.calc_B()
        
        L_expect = sp.zeros((nL,), dtype=sp.complex128)
        B_L = sp.empty((nL, self.N + 1), dtype=sp.ndarray) 
        
        ham = self.ham
        ham_sites = self.ham_sites
        
        for al in xrange(nL):
            #replace Hamiltonian with the Lindblad operator linds[al]
            #and compute corresponding tangent vector
            #(alternatively, we could combine the Lindblad operators first, together with the Wiener samples, 
            # then compute a single tangent vector as one operation)
            self.ham = linds[al]
            prev_ham_sites = self.ham_sites
            self.ham_sites = 0
            for n in xrange(len(linds[al])):
                if not linds[al][n] is None:
                    self.ham_sites = len(linds[al][n].shape) / 2
                    break
            if self.ham_sites == 0:
                continue
            self.calc_C(calc_AA=self.ham_sites != prev_ham_sites) #This computes AA (AAA) only if ham_sites changed
            self.calc_K()
            B_L[al,:] = self.calc_B()
            L_expect[al] = self.H_expect
                
        self.ham = ham
        self.ham_sites = ham_sites
        
        #Apply Hamiltonian evolution step
        for n in xrange(1, self.N):
            if not B_H[n] is None:
                self.A[n] += dt * 1.j * B_H[n]
               
        #Apply diffusion step
        for al in xrange(nL):
            #sample complex Wiener process
            u = sp.random.normal(0, sp.sqrt(dt), (2,))
            W = (u[0] + 1.j * u[1]) / sp.sqrt(2)
            for n in xrange(1, self.N):
                if not B_L[al,n] is None:
                    self.A[n] += B_L[al,n] * (W + (L_expect[al] * dt))


    def get_op_A_1s(self,op,n):
        """Applies an on-site operator to one site and returns
        the parameter tensor for that site after the change.
        
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
                
        return newAn