#!/usr/bin/env python
# -*- coding: utf-8 -*-


import scipy as sp
import scipy.linalg as la
import mps_gen as mg
import tdvp_common as tm
import matmul as mm
import tdvp_gen as TDVP
import copy

# def _add_local_ops(*ops):
    # lmax = max(map(len, ops))
    # resop = [None] * lmax
    # for n in xrange(1, lmax):
        # resop[n] = add_local_ops_n([op[n] if n < len(op) else None for op in ops])
    # return resop

# def _add_local_ops_n(ops):
    # res = None
    # for op in ops:
        # if op is not None:
            # if res is None:
                # res = op
            # elif res.shape == op.shape:
                # res += op
            # else:
                # if sp.ndim(op) > sp.ndim(res):
                    # res = add_ops(op, res)
                # else:
                    # res = add_ops(res, op)
                    
# def _add_ops(lr, sr):
    # n_lr = sp.ndim(lr) / 2
    # n_sr = sp.ndim(sr) / 2
    # q = lr.shape[:n_lr]
    # assert sr.shape[:n_sr] == q
    # srMdim = sp.prod(sr.shape[:n_sr])
    # sr = sr.reshape((srMdim, srMdim))
    # sr = sp.kron(sr, sp.eye(sp.prod(sr.shape[n_sr:n_lr])))
    # sr = sr.reshape(lr.shape)
    
    # return sr + lr

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