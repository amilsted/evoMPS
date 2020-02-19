# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 17:29:27 2011

@author: Ashley Milsted

"""
from __future__ import absolute_import, division, print_function

import copy as cp
import scipy as sp
import scipy.linalg as la
from . import matmul as mm
from . import tdvp_common as tm
from .tdvp_uniform import EvoMPS_TDVP_Uniform
from .mps_sandwich import EvoMPS_MPS_Sandwich
import time

def go(sim, tau, steps, force_calc_lr=False, RK4=False,
       autogrow=False, autogrow_amount=2, autogrow_max_N=1000,
       op=None, op_every=5, prev_op_data=None, op_save_as=None,
       en_save_as=None,
       entropy_save_as=None, eta_save_as=None,
       overlap_with=None,
       overlap_save_as=None,
       append_saved=True,
       save_every=10, save_as=None, counter_start=0, t_start=None,
       csv_file=None,
       tol=0,
       print_eta_n=False):
    """A 'simple' integration loop for testing"""
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
                print("No previous  en-data, or error loading!")
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
                print("No previous  entropy-data, or error loading!")
                pass
            Sf = open(entropy_save_as, "a")
        else:
            Sf = open(entropy_save_as, "w")
        
    etadata = []
    if (not eta_save_as is None):
        if append_saved:
            try:
                etadata = sp.genfromtxt(eta_save_as).tolist()
            except:
                print("No previous  eta-data, or error loading!")
                pass
            etaf = open(eta_save_as, "a")
        else:
            etaf = open(eta_save_as, "w")

    if not op_save_as is None:
        if append_saved:
            try:
                data = sp.genfromtxt(op_save_as).tolist()
            except:
                print("No previous  op-data, or error loading!")
                pass
            opf = open(op_save_as, "a")
        else:
            opf = open(op_save_as, "w")
    
    oldata = []        
    if not overlap_save_as is None:
        if append_saved:
            try:
                oldata = sp.genfromtxt(overlap_save_as).tolist()
            except:
                print("No previous  overlap data, or error loading!")
                pass
            olf = open(overlap_save_as, "a")
        else:
            olf = open(overlap_save_as, "w")
        
    if not csv_file is None:
        if append_saved:
            csvf = open(csv_file, "a")
        else:
            csvf = open(csv_file, "w")
        
    header = "\t".join(["Step", "CPU", "t", "eta", "E_nonuniform", "E - E_prev", "grown_left", 
                        "grown_right"])
    print(header)
    print()
    if not csv_file is None:
        csvf.write(header + "\n")
    
#    hs_prev = None
#    hl_prev = 0
#    hr_prev = 0
#    hlc_prev = 0
#    hrc_prev = 0
    if t_start is None:
        t_start = counter_start * tau
    t_cpu0 = time.clock()
    for i in range(counter_start, steps + 1):
        rewrite_opf = False
        if i > counter_start:
            if RK4:
                sim.take_step_RK4(tau)
            else:
                sim.take_step(tau)
                
            eta = sim.eta.real

            etasqs = sim.eta_sq[1:].copy()
        
            #Basic dynamic expansion:
            if autogrow and sim.N < autogrow_max_N:
                if etasqs[0] > sim.eta_sq_uni.mean() * 10:
                    rewrite_opf = True
                    print("Growing left by: %u" % autogrow_amount)
                    sim.grow_left(autogrow_amount)
                    if not overlap_with is None:
                        overlap_with.grow_left(autogrow_amount)
                    for j in range(autogrow_amount):
                        for row in data:                        
                            row.insert(0, 0)
                        for row in endata:
                            row.insert(0, 0)
                        for row in Sdata:
                            row.insert(0, 0)
                        for row in etadata:
                            row.insert(0, 0)
    
                if etasqs[-1] > sim.eta_sq_uni.mean() * 10:
                    rewrite_opf = True
                    print("Growing right by: %u" % autogrow_amount)
                    sim.grow_right(autogrow_amount)
                    if not overlap_with is None:
                        overlap_with.grow_right(autogrow_amount)
                    for j in range(autogrow_amount):
                        for row in data:
                            row.append(0)
                        for row in endata:
                            row.append(0)
                        for row in Sdata:
                            row.append(0)
                        for row in etadata:
                            row.append(0)

        else:            
            eta = 0
            etasqs = sp.zeros(1)
        
        #if not RK4:
        #    rcf = (i % 4 == 0)
        #else:
        rcf = True
        sim.update(restore_CF=rcf) #now we are measuring the stepped state
        h = sim.dH_expect
        #hs = sim.h_expect - sim.uni_l.h
#        if not hs_prev is None:
#            diff = hs - hs_prev
#            #print diff
#            print (sim.h_left - hl_prev, sim.h_right - hr_prev) 
#            print (sim.h_left_c - hlc_prev, sim.h_right_c - hrc_prev) 
#            print (diff.sum(), diff[15:-15].sum())
#        hs_prev = hs
#        hl_prev = sim.h_left
#        hr_prev = sim.h_right
#        hlc_prev = sim.h_left_c
#        hrc_prev = sim.h_right_c
            
        if not save_as is None and ((i % save_every == 0)
                                    or i == steps - 1):
            sim.save_state(save_as) #+ "_%u" % i)

        if i % 20 == 19:
            print(header)
            
        t = abs((i - counter_start) * tau) + abs(t_start)
        t_cpu = time.clock() - t_cpu0
        line = "\t".join(map(str, (i, "%.1f" % t_cpu, t, eta, h.real, (h - h_prev).real, 
                                   sim.grown_left, sim.grown_right)))
        print(line)
        if print_eta_n:
            print("eta_n:")
            print(etasqs.real)
        
        if not csv_file is None:
            csvf.write(line + "\n")
            csvf.flush()

        h_prev = h

        if (not op is None) and (i % op_every == 0):
            op_range = list(range(-10, sim.N + 10))
            row = [sim.expect_1s(op, n).real for n in op_range]
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
                
        if (not eta_save_as is None):
            row = sim.eta_sq.real.tolist()
            etadata.append(row)
            if rewrite_opf:
                etaf.close()
                etaf = open(eta_save_as, "w")
                for row in etadata:
                    etaf.write("\t".join(map(str, row)) + "\n")
                etaf.flush()
            else:
                etaf.write("\t".join(map(str, row)) + "\n")
                etaf.flush()

        if (not overlap_with is None) and (i % op_every == 0):
            row = [abs(sim.overlap(overlap_with))]
            oldata.append(row)
            if not overlap_save_as is None:
                olf.write("\t".join(map(str, row)) + "\n")
                olf.flush()
            
        if i > counter_start and eta < tol:
            print("Tolerance reached!")
            break
            
    if not op_save_as is None:
        opf.close()
        
    if (not en_save_as is None):
        enf.close()
        
    if (not entropy_save_as is None):
        Sf.close()
        
    if (not eta_save_as is None):
        etaf.close()

    if (not overlap_save_as is None):
        olf.close()
        
    if not csv_file is None:
        csvf.close()

    return data, endata, Sdata, oldata

class EvoMPS_TDVP_Sandwich(EvoMPS_MPS_Sandwich):#, EvoMPS_TDVP_Generic):

    def __init__(self, N, uni_ground, uni_right=None, update_bulks=True):
        
        super(EvoMPS_TDVP_Sandwich, self).__init__(
            N, uni_ground, uni_right, update_bulks=update_bulks)
        
        assert uni_ground.ham_sites == 2, 'Sandwiches only supported for \
                                           nearest-neighbour Hamiltonians at present!'
        
        self.uni_l.calc_B()
        self.eta_sq_uni = self.uni_l.eta_sq
        print("Bulk eta: ", self.eta_sq_uni)
        
        self.h_nn = None
        """The Hamiltonian for the nonuniform region. 
           Can be changed, for example, to perform
           a quench or to study an impurity problem. 
           The number of neighbouring sites acted on must be 
           specified in ham_sites."""
        
        if callable(self.uni_l.ham):
            self.h_nn = lambda n, s, t, u, v: self.uni_l.ham(s, t, u, v)
        else:
            self.h_nn = [self.uni_l.ham] * (self.N + 1)

        self.eps = sp.finfo(self.typ).eps


    def _init_arrays(self):
        super(EvoMPS_TDVP_Sandwich, self)._init_arrays()
        
        self.AA = sp.empty((self.N + 1), dtype=sp.ndarray) #Elements 1..N
        
        #Make indicies correspond to the thesis
        #Deliberately add a None to the end to catch [-1] indexing!
        self.K = sp.empty((self.N + 3), dtype=sp.ndarray) #Elements 1..N
        self.K_l = sp.empty((self.N + 3), dtype=sp.ndarray) #Elements 1..N
        self.C = sp.empty((self.N + 2), dtype=sp.ndarray) #Elements 1..N-1

        self.K_l[0] = sp.zeros((self.D[0], self.D[0]), dtype=self.typ, order=self.odr)
        self.C[0] = sp.empty((self.q[0], self.q[1], self.D[0], self.D[1]), dtype=self.typ, order=self.odr)
        for n in range(1, self.N + 1):
            self.C[n] = sp.empty((self.q[n], self.q[n+1], self.D[n-1], self.D[n+1]), dtype=self.typ, order=self.odr)
        for n in range(1, self.N_centre + 1):
            self.K_l[n] = sp.zeros((self.D[n], self.D[n]), dtype=self.typ, order=self.odr)
        for n in range(self.N_centre, self.N + 2):
            self.K[n] = sp.zeros((self.D[n - 1], self.D[n - 1]), dtype=self.typ, order=self.odr)
            
        self.eta_sq = sp.zeros((self.N + 1), dtype=self.typ)
        self.eta_sq.fill(0)
        """The per-site contributions to the norm-squared of the TDVP tangent vector 
           (projection of the exact time evolution onto the MPS tangent plane. 
           Only available after calling take_step()."""
           
        """The norm of the TDVP tangent vector.
        """
        self.eta = sp.NaN
        
        self.h_expect = sp.empty((self.N + 1), dtype=self.typ)
        self.h_expect.fill(sp.NaN)
        """The local energy expectation values (of each Hamiltonian term), 
           available after calling update() or calc_K()."""
           
        self.dH_expect = sp.NaN
        """The energy-difference (compared to the uniform bulk state) 
           expectation value, available after calling update()
           or calc_K()."""

    @classmethod
    def from_mps(cls, mps, ham_L, ham_C, ham_R, ham_sites=None):
        if len(ham_C) != mps.N + 1:
            raise ValueError(
                "Central window hamiltonian has {} terms. {} required!".format(
                    len(ham_C), mps.N + 1))
        uni_l = EvoMPS_TDVP_Uniform.from_mps(mps.uni_l, ham_L, ham_sites=ham_sites)
        uni_r = EvoMPS_TDVP_Uniform.from_mps(mps.uni_r, ham_R, ham_sites=ham_sites)
        tdvp = cls(mps.N, uni_l, uni_r, update_bulks=False)
        tdvp.A = cp.deepcopy(mps.A)
        tdvp.l = cp.deepcopy(mps.l)
        tdvp.r = cp.deepcopy(mps.r)
        tdvp.D = cp.deepcopy(mps.D)
        tdvp.h_nn = ham_C
        # TODO: Instead of updating, copy more stuff?
        tdvp.update(restore_CF=False, normalize=True)
        return tdvp

    def get_AA(self, n):
        if n < 0:
            return self.uni_l.AA[(n - 1) % self.uni_l.L]
        elif n > self.N:
            return self.uni_r.AA[(n - self.N - 1) % self.uni_r.L]
        else:
            return self.AA[n]

    def calc_C(self, n_low=-1, n_high=-1):
        """Generates the C matrices used to calculate the K's and ultimately the B's

        These are to be used on one side of the super-operator when applying the
        nearest-neighbour Hamiltonian, similarly to C in eqn. (44) of
        arXiv:1103.0936v2 [cond-mat.str-el], except being for the non-norm-preserving case.

        Makes use only of the nearest-neighbour hamiltonian, and of the A's.

        C[n] depends on A[n] and A[n + 1].
        
        This calculation can be significantly faster if h_nn is in array form.

        """
        if self.h_nn is None:
            return

        if n_low < 1:
            n_low = 0
        if n_high < 1:
            n_high = self.N + 1
        
        if callable(self.h_nn):
            for n in range(n_low, n_high):
                self.C[n] = tm.calc_C_func_op(lambda s,t,u,v: self.h_nn(n,s,t,u,v), 
                                              self.get_A(n), self.get_A(n + 1))
        else:
            for n in range(n_low, n_high):                   
                self.AA[n] = tm.calc_AA(self.get_A(n), self.get_A(n + 1))
                
                self.C[n] = tm.calc_C_mat_op_AA(self.h_nn[n], self.AA[n])

    def calc_K(self):
        """Generates the right K matrices used to calculate the B's
        
        K[n] contains 'column-vectors' such that <l[n]|K[n]> = trace(l[n].dot(K[n])).
        K_l[n] contains 'bra-vectors' such that <K_l[n]|r[n]> = trace(K_l[n].dot(r[n])).
        """
   
        self.h_expect = sp.zeros((self.N + 1), dtype=self.typ)
        
        self.uni_r.calc_AA()
        self.uni_r.calc_C()
        self.uni_r.calc_K()
        self.K[self.N + 1][:] = self.uni_r.K[0]

        self.uni_l.calc_AA()
        self.uni_l.calc_C()
        K_left, h_left_uni = self.uni_l.calc_K_l()
        
        self.K_l[0][:] = K_left[-1]

        for n in range(self.N, self.N_centre - 1, -1):
            self.K[n], he = tm.calc_K(self.K[n + 1], self.C[n], self.get_l(n - 1),
                                      self.r[n + 1], self.A[n], self.get_AA(n))
                
            self.h_expect[n] = he

        for n in range(1, self.N_centre + 1):
            self.K_l[n], he = tm.calc_K_l(self.K_l[n - 1], self.C[n - 1], self.get_l(n - 2),
                                          self.r[n], self.A[n], self.get_AA(n - 1))
                
            self.h_expect[n - 1] = he

        self.dH_expect = (mm.adot_noconj(self.K_l[self.N_centre], self.r[self.N_centre]) 
                          + mm.adot(self.l[self.N_centre - 1], self.K[self.N_centre]) 
                          - (self.N + 1) * self.uni_r.h_expect)
        
#        self.h_left = mm.adot_noconj(K_left, self.r[0])
#        self.h_right = mm.adot(self.l[self.N], self.K[self.N + 1])
#        
#        self.h_left_c = mm.adot_noconj(self.K_l[self.N_centre], self.r[self.N_centre])
#        self.h_right_c = mm.adot(self.l[self.N_centre - 1], self.K[self.N_centre])
#        print self.h
#        print (mm.adot_noconj(K_left, self.r[0]) + mm.adot(self.l[self.N], self.K[self.N + 1])
#               + self.h_expect.sum() - (self.N + 1) * self.uni_r.h)
#        print (mm.adot_noconj(K_left), self.r[0]), mm.adot(self.l[self.N], self.K[self.N + 1]),
#               self.h_expect.sum() - (self.N + 1) * self.uni_r.h)
#        print self.h_expect


    def calc_x(self, n, Vsh, sqrt_l, sqrt_r, sqrt_l_inv, sqrt_r_inv, right=True):
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
        
        if right:
            x = tm.calc_x(self.K[n + 1], C, self.C[n - 1], self.r[n + 1],
                          lm2, self.get_A(n - 1), self.A[n], self.get_A(n + 1),
                          sqrt_l, sqrt_l_inv, sqrt_r, sqrt_r_inv, Vsh)
        else:
            x = tm.calc_x_l(self.K_l[n - 1], C, self.C[n - 1], self.r[n + 1],
                          lm2, self.get_A(n - 1), self.A[n], self.get_A(n + 1),
                          sqrt_l, sqrt_l_inv, sqrt_r, sqrt_r_inv, Vsh)

        return x
        

    def calc_B_centre(self):
        """Calculate the optimal B_centre given right gauge-fixing on n_centre+1..N and
        left gauge-fixing on 1..n_centre-1.
        
        We use the non-norm-preserving K's, since the norm-preservation
        is not needed elsewhere. It is cleaner to subtract the relevant
        norm-changing terms from the K's here than to generate all K's
        with norm-preservation.
        """
        Bc = sp.empty_like(self.A[self.N_centre])
        
        Nc = self.N_centre

        try:
            rc_i = self.r[Nc].inv()
        except AttributeError:
            rc_i = mm.invmh(self.r[Nc])
            
        try:
            lcm1_i = self.l[Nc - 1].inv()
        except AttributeError:
            lcm1_i = mm.invmh(self.l[Nc - 1])
        
        Acm1 = self.get_A(Nc - 1)
        Ac = self.A[Nc]
        Acp1 = self.get_A(Nc + 1)
        rc = self.r[Nc]
        rcp1 = self.get_r(Nc + 1)
        lcm1 = self.get_l(Nc - 1)
        lcm2 = self.get_l(Nc - 2)
        
        #Note: this is a 'bra-vector'
        K_l_cm1 = self.K_l[Nc - 1] - lcm1 * mm.adot_noconj(self.K_l[Nc - 1], self.r[Nc - 1])
        
        Kcp1 = self.K[Nc + 1] - rc * mm.adot(self.l[Nc], self.K[Nc + 1])
        
        Cc = self.C[Nc] - self.h_expect[Nc] * self.AA[Nc]
        Ccm1 = self.C[Nc - 1] - self.h_expect[Nc - 1] * self.AA[Nc - 1]
        
        for s in range(self.q[1]):
            try: #3
                Bc[s] = Ac[s].dot(rc_i.dot_left(Kcp1))
            except AttributeError:
                Bc[s] = Ac[s].dot(Kcp1.dot(rc_i))
            
            for t in range(self.q[2]): #1
                try:
                    Bc[s] += Cc[s, t].dot(rcp1.dot(rc_i.dot_left(mm.H(Acp1[t]))))
                except AttributeError:
                    Bc[s] += Cc[s, t].dot(rcp1.dot(mm.H(Acp1[t]).dot(rc_i)))                    
                
            Bcsbit = K_l_cm1.dot(Ac[s]) #4
                            
            for t in range(self.q[0]): #2
                Bcsbit += mm.H(Acm1[t]).dot(lcm2.dot(Ccm1[t,s]))
                
            Bc[s] += lcm1_i.dot(Bcsbit)
           
        rb = tm.eps_r_noop(rc, Bc, Bc)
        eta_sq = mm.adot(lcm1, rb)
                
        return Bc, eta_sq

    def calc_B_n(self, n, set_eta=True):
        """Generates the B[n] tangent vector corresponding to physical evolution of the state.

        In other words, this returns B[n][x*] (equiv. eqn. (47) of
        arXiv:1103.0936v2 [cond-mat.str-el])
        with x* the parameter matrices satisfying the Euler-Lagrange equations
        as closely as possible.
        
        In the case of Bc, use the general Bc generated in calc_B_centre().
        """
        if n == self.N_centre:
            B, eta_sq_c = self.calc_B_centre()
            if set_eta:
                self.eta_sq[self.N_centre] = eta_sq_c
        else:
            l_sqrt, r_sqrt, l_sqrt_inv, r_sqrt_inv = self.calc_l_r_roots(n)
            
            if n > self.N_centre:
                Vsh = tm.calc_Vsh(self.A[n], r_sqrt, sanity_checks=self.sanity_checks)
                if Vsh is None:
                    return None
                x = self.calc_x(n, Vsh, l_sqrt, r_sqrt, l_sqrt_inv, r_sqrt_inv, right=True)
                
                B = sp.empty_like(self.A[n])
                for s in range(self.q[n]):
                    B[s] = mm.mmul(l_sqrt_inv, x, mm.H(Vsh[s]), r_sqrt_inv)
                    
                if self.sanity_checks:
                    M = tm.eps_r_noop(self.r[n], B, self.A[n])
                    if not sp.allclose(M, 0):
                        print("Sanity Fail in calc_B!: B_%u does not satisfy GFC!" % n)
            else:
                Vsh = tm.calc_Vsh_l(self.A[n], l_sqrt, sanity_checks=self.sanity_checks)
                if Vsh is None:
                    return None
                x = self.calc_x(n, Vsh, l_sqrt, r_sqrt, l_sqrt_inv, r_sqrt_inv, right=False)
                
                B = sp.empty_like(self.A[n])
                for s in range(self.q[n]):
                    B[s] = mm.mmul(l_sqrt_inv, mm.H(Vsh[s]), x, r_sqrt_inv)
                    
                if self.sanity_checks:
                    M = tm.eps_l_noop(self.l[n - 1], B, self.A[n])
                    if not sp.allclose(M, 0):
                        print("Sanity Fail in calc_B!: B_%u does not satisfy GFC!" % n)
            
            if set_eta:
                self.eta_sq[n] = mm.adot(x, x)

        return B

    def calc_B(self):
        eta_tot = 0
        self.eta_sq.fill(0)
        
        B = [None]
        for n in range(1, self.N + 1):
            B.append(self.calc_B_n(n))
            eta_tot += self.eta_sq[n]
        self.eta = sp.sqrt(self.eta_sq.sum())

        return B

    def calc_l_r_roots(self, n):
        """Returns the matrix square roots (and inverses) needed to calculate B.

        Hermiticity of l[n] and r[n] is used to speed this up.
        If an exception occurs here, it is probably because these matrices
        are no longer Hermitian (enough).
        
        If l[n] or r[n] are diagonal or the identity, further optimizations are
        used.
        """
        assert 0 < n <= self.N, 'calc_l_r_roots: Bad n!'
        
        l_sqrt, l_sqrt_i, r_sqrt, r_sqrt_i = tm.calc_l_r_roots(self.l[n - 1], 
                                                               self.r[n], 
                                                               zero_tol=self.zero_tol,
                                                               sanity_checks=self.sanity_checks)

        return l_sqrt, r_sqrt, l_sqrt_i, r_sqrt_i

    def update(self, restore_CF=True, normalize=True, auto_truncate=False):
        """Perform all necessary steps needed before taking the next step,
        or calculating expectation values etc., is possible.
        
        Return the excess energy.
        """
        super(EvoMPS_TDVP_Sandwich, self).update(
            restore_CF=restore_CF, normalize=normalize, auto_truncate=auto_truncate)
        
        self.calc_C()
        self.calc_K()

    def compute_projection_error(self):
        """Quick copy-paste job to get the projection error for a 2-site Ham.
        """
        #Y = sp.empty((self.N + 1), dtype=sp.ndarray)
        etaBB_sq = sp.zeros((self.N + 1), dtype=sp.complex128)
        for n in range(1, self.N):
            l_s_nm1, l_si_nm1, r_s_np1, r_si_np1 = tm.calc_l_r_roots(
                self.get_l(n - 1),
                self.get_r(n + 1),
                zero_tol=self.zero_tol,
                sanity_checks=self.sanity_checks,
                sc_data=('site', n))
            Vlh_n = tm.calc_Vsh_l(
                self.get_A(n), l_s_nm1, sanity_checks=self.sanity_checks)
            Vrh_np1 = tm.calc_Vsh(
                self.get_A(n+1), r_s_np1, sanity_checks=self.sanity_checks)

            if (not Vrh_np1 is None and not Vlh_n is None):
                _, etaBB_sq[n] = tm.calc_BB_Y_2s(self.C[n], Vlh_n, 
                                           Vrh_np1, l_s_nm1, r_s_np1)
        return etaBB_sq

    def calc_BB(self, sv_tol=1E-10, dD_max=16, D_max=0):
        dD_maxes = sp.minimum(D_max - self.D, dD_max)

        BB12 = [None] * (self.N + 1)
        BB21 = [None] * (self.N + 1)
        dD = [0] * (self.N + 1)
        etaBB_sq = sp.zeros((self.N + 1), dtype=sp.complex128)
        for n in range(1, self.N):
            l_s_nm1, l_si_nm1, r_s_np1, r_si_np1 = tm.calc_l_r_roots(
                self.get_l(n - 1),
                self.get_r(n + 1),
                zero_tol=self.zero_tol,
                sanity_checks=self.sanity_checks,
                sc_data=('site', n))
            Vlh_n = tm.calc_Vsh_l(
                self.get_A(n), l_s_nm1, sanity_checks=self.sanity_checks)
            Vrh_np1 = tm.calc_Vsh(
                self.get_A(n+1), r_s_np1, sanity_checks=self.sanity_checks)

            if (not Vrh_np1 is None and not Vlh_n is None):
                Y_n, etaBB_sq[n] = tm.calc_BB_Y_2s(self.C[n], Vlh_n, 
                                           Vrh_np1, l_s_nm1, r_s_np1)

                BB12[n], BB21[n + 1], dD[n] = tm.calc_BB_2s(
                    Y_n,
                    Vlh_n,
                    Vrh_np1,
                    l_si_nm1,
                    r_si_np1,
                    dD_max=dD_maxes[n],
                    sv_tol=sv_tol)

                #if BB12[n] is None:
                #    log.warn("calc_BB_2s: Could not calculate BB_2s at n=%u", n)
        return BB12, BB21, dD

    def take_step(self, dtau, B=None, dynexp=False, dD_max=16, D_max=0, sv_tol=1E-14):
        """Performs a complete forward-Euler step of imaginary time dtau.

        If dtau is itself imaginary, real-time evolution results.

        Parameters
        ----------
        dtau : complex
            The (imaginary or real) amount of imaginary time (tau) to step.
        B : optional list of tensors
            The current evolution vector, to avoid duplicating computations.
        """
        if B is None:
            B = self.calc_B()
        if dynexp:
            BB12, BB21, dD = self.calc_BB(sv_tol, dD_max, D_max)

        for n in range(1, self.N + 1):
            if not B[n] is None:
                self.A[n] += -dtau * B[n]
                B[n] = None

            if dynexp and (BB12[n] is not None or BB21[n] is not None):
                self.D[n] += dD[n]
                newAn = sp.zeros((self.q[n], self.D[n-1], self.D[n]), dtype=self.typ, order=self.odr)
                oldDnm1, oldDn = self.A[n].shape[1:]
                newAn[:, :oldDnm1, :oldDn] = self.A[n]

                if not BB12[n] is None:
                    newAn[:, :oldDnm1, oldDn:] = -1.j * sp.sqrt(dtau) * BB12[n]
                if not BB21[n] is None:
                    newAn[:, oldDnm1:, :oldDn] = -1.j * sp.sqrt(dtau) * BB21[n]

                self.A[n] = newAn

        if dynexp:
            return dD

    def take_step_RK4(self, dtau, B_i=None):
        """Take a step using the fourth-order explicit Runge-Kutta method.

        This requires more memory than a simple forward Euler step, and also
        more than a backward Euler step. It is, however, far more accurate
        and stable than forward Euler.

        Parameters
        ----------
        dtau : complex
            The (imaginary or real) amount of imaginary time (tau) to step.
        B_i : optional list of tensors
            The current evolution vector, to avoid duplicating computations.
        """        
        eta_tot = 0
        self.eta_sq.fill(0)

        #Take a copy of the current state
        A0 = sp.empty_like(self.A)
        for n in range(1, self.N + 1):
            A0[n] = self.A[n].copy()

        B = self.calc_B() if B_i is None else B_i[:]
        B_fin = B[:]

        for n in range(1, self.N + 1):
            if not B[n] is None:
                self.A[n] = A0[n] - dtau/2 * B[n]
                B[n] = None

        self.update(restore_CF=False, normalize=False)
        
        B = [None]
        for n in range(1, self.N + 1):
            B.append(self.calc_B_n(n, set_eta=False)) #k2

        for n in range(1, self.N + 1):
            if not B[n] is None:
                self.A[n] = A0[n] - dtau/2 * B[n]
                B_fin[n] += 2 * B[n]
                B[n] = None

        self.update(restore_CF=False, normalize=False)

        B = [None]
        for n in range(1, self.N + 1):
            B.append(self.calc_B_n(n, set_eta=False)) #k3
            
        for n in range(1, self.N + 1):
            if not B[n] is None:
                self.A[n] = A0[n] - dtau * B[n]
                B_fin[n] += 2 * B[n]
                B[n] = None

        self.update(restore_CF=False, normalize=False)

        for n in range(1, self.N + 1):
            B = self.calc_B_n(n, set_eta=False) #k4
            if not B is None:
                B_fin[n] += B

        for n in range(1, self.N + 1):
            if not B_fin[n] is None:
                self.A[n] = A0[n] - dtau /6 * B_fin[n]
        
    def grow_left(self, m):
        super(EvoMPS_TDVP_Sandwich, self).grow_left(m)
        if not callable(self.h_nn):
            self.h_nn = [self.uni_l.ham] * m * self.uni_l.L + list(self.h_nn)
        self.N_centre += m
            
    def grow_right(self, m):
        super(EvoMPS_TDVP_Sandwich, self).grow_right(m)
        if not callable(self.h_nn):
            self.h_nn = list(self.h_nn) + [self.uni_r.ham] * m * self.uni_r.L

    def save_state(self, file_name, userdata=None):
        tosave = sp.empty((9), dtype=sp.ndarray)
        
        Asave = self.A.copy()
        Asave[0] = self.uni_l.A
        Asave[self.N + 1] = self.uni_r.A
        
        tosave[0] = Asave
        tosave[1] = self.l[0]        
        tosave[2] = self.uni_l.r[-1]
        tosave[3] = self.uni_l.K_left[-1]
        tosave[4] = self.r[self.N]
        tosave[5] = self.uni_r.l[-1]
        tosave[6] = self.uni_r.K[0]
        tosave[7] = sp.array([[self.grown_left, self.grown_right], 
                             [self.shrunk_left, self.shrunk_right]])
        tosave[8] = userdata
        
        sp.save(file_name, tosave)

    def load_state(self, file_name, autogrow=False, do_update=True):
        toload = sp.load(file_name, allow_pickle=True)
        
        try:
            if toload.shape[0] != 9:
                print("Error loading state: Bad data!")
                return
                
            if autogrow and toload[0].shape[0] != self.A.shape[0]:
                newN = toload[0].shape[0] - 3
                print("Changing N to: %u" % newN)
                self.grow_left(newN - self.N)
                
            if toload[0].shape != self.A.shape:
                print("Cannot load state: Dimension mismatch!")
                return
            
            self.A = toload[0]
            self.l[0] = toload[1]
            self.uni_l.r[-1] = toload[2]
            if toload[3]:
                self.uni_l.K_left[-1] = toload[3]
            self.r[self.N] = toload[4]
            self.r[self.N + 1] = self.r[self.N]
            self.uni_r.l[-1] = toload[5]
            if toload[6]:
                self.uni_r.K[0] = toload[6]
            
            self.grown_left = toload[7][0, 0]
            self.grown_right = toload[7][0, 1]
            self.shrunk_left = toload[7][1, 0]
            self.shrunk_right = toload[7][1, 1]
            
            self.uni_l.A = self.A[0]
            self.uni_l.l[-1] = self.l[0]
            self.uni_l.l_before_CF = self.uni_l.l[-1]
            self.uni_l.r_before_CF = self.uni_l.r[-1]
            
            self.uni_r.A = self.A[self.N + 1]
            self.uni_r.r[-1] = self.r[self.N]
            self.uni_r.l_before_CF = self.uni_r.l[-1]
            self.uni_r.r_before_CF = self.uni_r.r[-1]
            
            try:
                if len(self.uni_l.A.shape) == 3:
                    self.uni_l.A = [self.uni_l.A]
            except AttributeError:
                pass
            
            try:
                if len(self.uni_r.A.shape) == 3:
                    self.uni_r.A = [self.uni_r.A]
            except AttributeError:
                pass
            
            self.A[0] = None
            self.A[self.N + 1] = None

            print("loaded.")
            
            if do_update:
                self.update()

            return toload[8]
            
        except AttributeError:
            print("Error loading state: Bad data!")
            return
