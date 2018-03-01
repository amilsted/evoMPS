# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 14:35:13 2013

@author: ash
"""

import numpy as np
import scipy as sp
import scipy.linalg as la
import scipy.sparse.linalg as las
import tdvp_common as tm
import matmul as m
from mps_uniform_pinv import pinv_1mE
import logging

log = logging.getLogger(__name__)
        
class Excite_H_Op:
    def __init__(self, tdvp, tdvp2, p, pinv_tol=1E-12,
                 sanity_checks=False, sanity_tol=1E-12):
        """Creates an Excite_H_Op object, which is a LinearOperator.
        
        This wraps the effective Hamiltonian in terms of MPS tangent vectors
        as a LinearOperator that can be used with SciPy's sparse linear
        algebra routines.
        
        Parameters
        ----------
        tdvp : EvoMPS_TDVP_Uniform
            tdvp object providing the required operations in the matrix representation.
        tdvp2 : EvoMPS_TDVP_Uniform
            Second tdvp object (can be the same as tdvp), for example containing a different ground state.
        p : float
            Momentum in units of inverse lattice spacing.
        """
        assert tdvp.L == 1, 'Excite_H_Op only supports blocks of size 1'
        
        assert tdvp.L == tdvp2.L, 'Block sizes must match!'
        assert tdvp.q == tdvp2.q, 'Hilbert spaces must match!'
        assert tdvp.D == tdvp2.D, 'Bond-dimensions must match!'
        
        self.tdvp = tdvp
        self.tdvp2 = tdvp2
        self.p = p
        
        self.D = tdvp.D
        self.q = tdvp.q
        self.ham_sites = tdvp.ham_sites
        self.ham = tdvp.ham
        
        self.sanity_checks = sanity_checks
        self.sanity_tol = sanity_tol
        
        self.pinv_solver = None
        self.pinv_tol = pinv_tol
        self.pinv_CUDA = False
        self.pinv_CUDA_batch = False
        
        d = (self.q - 1) * self.D**2
        self.shape = (d, d)
        
        self.dtype = np.dtype(tdvp.typ)
        
        self.prereq = (self.calc_BHB_prereq(tdvp, tdvp2))
        
        self.calls = 0
        
        self.M_prev = None
        self.y_pi_prev = None
        
    def calc_BHB_prereq(self, tdvp, tdvp2):
        """Calculates prerequisites for the application of the effective Hamiltonian in terms of tangent vectors.
        
        This is called (indirectly) by the self.excite.. functions.
        
        Parameters
        ----------
        tdvp2: EvoMPS_TDVP_Uniform
            Second state (may be the same, or another ground state).
            
        Returns
        -------
        A lot of stuff.
        """
        l = tdvp.l[0]
        r_ = tdvp2.r[0]
        r__sqrt = tdvp2.r_sqrt[0]
        r__sqrt_i = tdvp2.r_sqrt_i[0]
        A = tdvp.A[0]
        A_ = tdvp2.A[0]
        AA = tdvp.AA[0]
        AA_ = tdvp2.AA[0]
        AAA_ = tdvp2.AAA[0]
        
        eyed = np.eye(self.q**self.ham_sites)
        eyed = eyed.reshape(tuple([self.q] * self.ham_sites * 2))
        ham_ = self.ham - tdvp.h_expect.real * eyed
            
        V_ = sp.transpose(tdvp2.Vsh[0], axes=(0, 2, 1)).conj().copy(order='C')
        
        Vri_ = sp.zeros_like(V_)
        try:
            for s in xrange(self.q):
                Vri_[s] = r__sqrt_i.dot_left(V_[s])
        except AttributeError:
            for s in xrange(self.q):
                Vri_[s] = V_[s].dot(r__sqrt_i)

        Vr_ = sp.zeros_like(V_)            
        try:
            for s in xrange(self.q):
                Vr_[s] = r__sqrt.dot_left(V_[s])
        except AttributeError:
            for s in xrange(self.q):
                Vr_[s] = V_[s].dot(r__sqrt)
                
        Vri_A_ = tm.calc_AA(Vri_, A_)
                
        if self.ham_sites == 2:
            _C_AhlA = np.empty((self.q, self.q, A.shape[2], A.shape[2]), dtype=tdvp.typ)
            for u in xrange(self.q):
                for s in xrange(self.q):
                    _C_AhlA[u, s] = A[u].conj().T.dot(l.dot(A[s]))
            C_AhlA = sp.tensordot(ham_, _C_AhlA, ((0, 2), (0, 1)))
            C_AhlA = sp.transpose(C_AhlA, axes=(1, 0, 2, 3)).copy(order='C')
            
            _C_A_Vrh_ = tm.calc_AA(A_, sp.transpose(Vr_, axes=(0, 2, 1)).conj())
            C_A_Vrh_ = sp.tensordot(ham_, _C_A_Vrh_, ((3, 1), (0, 1)))
            C_A_Vrh_ = sp.transpose(C_A_Vrh_, axes=(1, 0, 2, 3)).copy(order='C')
            
            C_Vri_A_conj = tm.calc_C_conj_mat_op_AA(ham_, Vri_A_).copy(order='C')
    
            C_ = tm.calc_C_mat_op_AA(ham_, AA_).copy(order='C')
            C_conj = tm.calc_C_conj_mat_op_AA(ham_, AA_).copy(order='C')
            
            rhs10 = tm.eps_r_op_2s_AA12_C34(r_, AA_, C_Vri_A_conj)
            
            return C_, C_conj, V_, Vr_, Vri_, C_Vri_A_conj, C_AhlA, C_A_Vrh_, rhs10
        elif self.ham_sites == 3:
            C_Vri_AA_ = np.empty((self.q, self.q, self.q, Vri_.shape[1], A_.shape[2]), dtype=tdvp.typ)
            for s in xrange(self.q):
                for t in xrange(self.q):
                    for u in xrange(self.q):
                        C_Vri_AA_[s, t, u] = Vri_[s].dot(AA_[t, u])
            C_Vri_AA_ = sp.tensordot(ham_, C_Vri_AA_, ((3, 4, 5), (0, 1, 2))).copy(order='C')
            
            C_AAA_r_Ah_Vrih = np.empty((self.q, self.q, self.q, self.q, self.q, #FIXME: could be too memory-intensive
                                        A_.shape[1], Vri_.shape[1]), 
                                       dtype=tdvp.typ)
            for s in xrange(self.q):
                for t in xrange(self.q):
                    for u in xrange(self.q):
                        for k in xrange(self.q):
                            for j in xrange(self.q):
                                C_AAA_r_Ah_Vrih[s, t, u, k, j] = AAA_[s, t, u].dot(r_.dot(A_[k].conj().T)).dot(Vri_[j].conj().T)
            C_AAA_r_Ah_Vrih = sp.tensordot(ham_, C_AAA_r_Ah_Vrih, ((3, 4, 5, 2, 1), (0, 1, 2, 3, 4))).copy(order='C')
            
            C_AhAhlAA = np.empty((self.q, self.q, self.q, self.q,
                                  A_.shape[2], A.shape[2]), dtype=tdvp.typ)
            for t in xrange(self.q):
                for j in xrange(self.q):
                    for i in xrange(self.q):
                        for s in xrange(self.q):
                            C_AhAhlAA[j, t, i, s] = AA[i, j].conj().T.dot(l.dot(AA[s, t]))
            C_AhAhlAA = sp.tensordot(ham_, C_AhAhlAA, ((4, 1, 0, 3), (1, 0, 2, 3))).copy(order='C')
            
            C_AA_r_Ah_Vrih_ = np.empty((self.q, self.q, self.q, self.q,
                                        A_.shape[1], Vri_.shape[1]), dtype=tdvp.typ)
            for t in xrange(self.q):
                for u in xrange(self.q):
                    for k in xrange(self.q):
                        for j in xrange(self.q):
                            C_AA_r_Ah_Vrih_[u, t, k, j] = AA_[t, u].dot(r_.dot(A_[k].conj().T)).dot(Vri_[j].conj().T)
            C_AA_r_Ah_Vrih_ = sp.tensordot(ham_, C_AA_r_Ah_Vrih_, ((4, 5, 2, 1), (1, 0, 2, 3))).copy(order='C')
            
            C_AAA_Vrh_ = np.empty((self.q, self.q, self.q, self.q,
                                   A_.shape[1], Vri_.shape[1]), dtype=tdvp.typ)
            for s in xrange(self.q):
                for t in xrange(self.q):
                    for u in xrange(self.q):
                        for k in xrange(self.q):
                            C_AAA_Vrh_[s, t, u, k] = AAA_[s, t, u].dot(Vr_[k].conj().T)
            C_AAA_Vrh_ = sp.tensordot(ham_, C_AAA_Vrh_, ((3, 4, 5, 2), (0, 1, 2, 3))).copy(order='C')
            
            C_Vri_A_r_Ah_ = np.empty((self.q, self.q, self.q,
                                      A_.shape[2], Vri_.shape[1]), dtype=tdvp.typ)
            for u in xrange(self.q):
                for k in xrange(self.q):
                    for j in xrange(self.q):
                        C_Vri_A_r_Ah_[u, k, j] = Vri_[j].dot(A_[k]).dot(r_.dot(A_[u].conj().T))
            C_Vri_A_r_Ah_ = sp.tensordot(ham_.conj(), C_Vri_A_r_Ah_, ((5, 2, 1), (0, 1, 2))).copy(order='C')
            
            C_AhlAA = np.empty((self.q, self.q, self.q,
                                      A_.shape[2], A.shape[2]), dtype=tdvp.typ)
            for j in xrange(self.q):
                for i in xrange(self.q):
                    for s in xrange(self.q):
                        C_AhlAA[j, i, s] = A[s].conj().T.dot(l.dot(AA[i, j]))
            C_AhlAA_conj = sp.tensordot(ham_.conj(), C_AhlAA, ((1, 0, 3), (0, 1, 2))).copy(order='C')
            C_AhlAA = sp.tensordot(ham_, C_AhlAA, ((4, 3, 0), (0, 1, 2)))
            C_AhlAA = sp.transpose(C_AhlAA, axes=(2, 0, 1, 3, 4)).copy(order='C')
            
            C_AA_Vrh = np.empty((self.q, self.q, self.q,
                                      A_.shape[2], Vr_.shape[1]), dtype=tdvp.typ)
            for t in xrange(self.q):
                for u in xrange(self.q):
                    for k in xrange(self.q):
                        C_AA_Vrh[k, u, t] = AA_[t, u].dot(Vr_[k].conj().T)
            C_AA_Vrh = sp.tensordot(ham_, C_AA_Vrh, ((4, 5, 2), (2, 1, 0))).copy(order='C')
            
            C_ = sp.tensordot(ham_, AAA_, ((3, 4, 5), (0, 1, 2))).copy(order='C')
            
            rhs10 = tm.eps_r_op_3s_C123_AAA456(r_, AAA_, C_Vri_AA_)

            #NOTE: These C's are good as C12 or C34, but only because h is Hermitian!
            #TODO: Make this consistent with the updated 2-site case above.
            
            return V_, Vr_, Vri_, Vri_A_, C_, C_Vri_AA_, C_AAA_r_Ah_Vrih, C_AhAhlAA, C_AA_r_Ah_Vrih_, C_AAA_Vrh_, C_Vri_A_r_Ah_, C_AhlAA, C_AhlAA_conj, C_AA_Vrh, rhs10,

    
    def calc_BHB(self, x, p, tdvp, tdvp2, prereq,
                    M_prev=None, y_pi_prev=None, pinv_solver=None):
        if pinv_solver is None:
            pinv_solver = las.gmres
            
        if self.ham_sites == 3:
            V_, Vr_, Vri_, Vri_A_, C_, C_Vri_AA_, C_AAA_r_Ah_Vrih, \
                    C_AhAhlAA, C_AA_r_Ah_Vrih_, C_AAA_Vrh_, C_Vri_A_r_Ah_, \
                    C_AhlAA, C_AhlAA_conj, C_AA_Vrh, rhs10 = prereq
        else:
            C_, C_conj, V_, Vr_, Vri_, C_Vri_A_conj, C_AhlA, C_A_Vrh_, rhs10 = prereq
        
        A = tdvp.A[0]
        A_ = tdvp2.A[0]
        AA = tdvp.AA[0]
        
        l = tdvp.l[0]
        r_ = tdvp2.r[0]
        
        l_sqrt = tdvp.l_sqrt[0]
        l_sqrt_i = tdvp.l_sqrt_i[0]
        
        r__sqrt = tdvp2.r_sqrt[0]
        r__sqrt_i = tdvp2.r_sqrt_i[0]
        
        K__r = tdvp2.K[0]
        K_l = tdvp.K_left[0]
        
        pseudo = tdvp2 is tdvp
        
        B = tdvp2.get_B_from_x(x, tdvp2.Vsh[0], l_sqrt_i, r__sqrt_i)
        
        #Skip zeros due to rank-deficiency
        if la.norm(B) == 0:
            return sp.zeros_like(x), M_prev, y_pi_prev
        
        if self.sanity_checks:
            tst = tm.eps_r_noop(r_, B, A_)
            if not la.norm(tst) > self.sanity_tol:
                log.warning("Sanity check failed: Gauge-fixing violation! " 
                            + str(la.norm(tst)))

        if self.sanity_checks:
            B2 = np.zeros_like(B)
            for s in xrange(self.q):
                B2[s] = l_sqrt_i.dot(x.dot(Vri_[s]))
            if la.norm(B - B2) / la.norm(B) > self.sanity_tol:
                log.warning("Sanity Fail in calc_BHB! Bad Vri!")
        
        BA_ = tm.calc_AA(B, A_)
        AB = tm.calc_AA(A, B)
        if self.ham_sites == 3:
            BAA_ = tm.calc_AAA_AA(BA_, A_)
            ABA_ = tm.calc_AAA_AA(AB, A_)
            AAB = tm.calc_AAA_AA(AA, B)

        y = tm.eps_l_noop(l, B, A)
        
#        if pseudo:
#            y = y - m.adot(r_, y) * l #should just = y due to gauge-fixing
        M = pinv_1mE(y, [A_], [A], l, r_, p=-p, left=True, pseudo=pseudo, 
                     out=M_prev, tol=self.pinv_tol, solver=pinv_solver,
                     use_CUDA=self.pinv_CUDA, CUDA_use_batch=self.pinv_CUDA_batch,
                     sanity_checks=self.sanity_checks, sc_data='M')
        
        #print m.adot(r, M)
        if self.sanity_checks:
            y2 = M - sp.exp(+1.j * p) * tm.eps_l_noop(M, A_, A)
            norm = la.norm(y.ravel())
            if norm == 0:
                norm = 1
            tst = la.norm(y - y2) / norm
            if tst > self.sanity_tol:
                log.warning("Sanity Fail in calc_BHB! Bad M. Off by: %g", tst)
#        if pseudo:
#            M = M - l * m.adot(r_, M)
        Mh = M.conj().T.copy(order='C')
        
        if self.ham_sites == 3:
            tmp = BAA_ + sp.exp(+1.j * p) * ABA_ + sp.exp(+2.j * p) * AAB
            res = l_sqrt.dot(tm.eps_r_op_3s_C123_AAA456(r_, tmp, C_Vri_AA_)) #1 1D, #3, #3c
        else:
            tmp = BA_ + sp.exp(+1.j * p) * AB
            res = l_sqrt.dot(tm.eps_r_op_2s_AA12_C34(r_, tmp, C_Vri_A_conj)) #1, #3 OK
                    
        res += sp.exp(-1.j * p) * l_sqrt_i.dot(Mh.dot(rhs10)) #10
        
        exp = sp.exp
        subres = sp.zeros_like(res)
        eye = m.eyemat(C_.shape[2], dtype=tdvp.typ)
        eye2 = m.eyemat(A.shape[2], dtype=tdvp.typ)
        if self.ham_sites == 3:
            subres += exp(-2.j * p) * tm.eps_l_noop(Mh, A, C_AAA_r_Ah_Vrih) #12
            subres += exp(-3.j * p) * tm.eps_l_op_2s_AA12_C34(Mh, AA, C_AAA_Vrh_) #12b
            for s in xrange(self.q):
                #subres += exp(-2.j * p) * A[s].conj().T.dot(Mh.dot(C_AAA_r_Ah_Vrih[s])) #12
                subres += tm.eps_r_noop(B[s], C_AhAhlAA[s, :], Vr_) #2b
                subres += exp(-1.j * p) * tm.eps_l_noop(l.dot(B[s]), A, C_AA_r_Ah_Vrih_[s, :]) #4
                subres += A[s].conj().T.dot(l.dot(tm.eps_r_op_2s_AA12_C34(eye2, AB, C_Vri_A_r_Ah_[s, :, :]))) #2 -ive of that it should be....
                subres += exp(-1.j * p) * tm.eps_l_op_2s_AA12_C34(eye2, C_AhlAA_conj[s, :, :], BA_).dot(Vr_[s].conj().T) #4b
                subres += exp(-2.j * p) * tm.eps_l_op_2s_AA12_C34(l.dot(B[s]), AA, C_AA_Vrh[s, :, :]) #4c
                
                try:
                    Bs_r = r_.dot_left(B[s])
                except AttributeError:
                    Bs_r = B[s].dot(r_)
                subres += exp(+1.j * p) * tm.eps_r_op_2s_AA12_C34(Bs_r, C_AhlAA[s, :, :], Vri_A_) #3b
                    
                #for t in xrange(self.q):
                    #subres += (C_AhAhlAA[t, s].dot(B[s]).dot(Vr_[t].conj().T)) #2b
                    #subres += (exp(-1.j * p) * A[s].conj().T.dot(l.dot(B[t])).dot(C_AA_r_Ah_Vrih_[s, t])) #4
                    #subres += (exp(-3.j * p) * AA[t, s].conj().T.dot(Mh).dot(C_AAA_Vrh_[t, s])) #12b
                    
                    #for u in xrange(self.q):
                        #subres += A[s].conj().T.dot(l.dot(AB[t, u]).dot(C_A_r_Ah_Vrih[s, t, u])) #2 -ive of that it should be....
                        #subres += (exp(+1.j * p) * C_AhlAA[t, s, s].dot(B[u]).dot(r_.dot(A_[u].conj().T)).dot(Vri_[t].conj().T)) #3b
                        #subres += (exp(-1.j * p) * C_AhAhlA[s, t, u].dot(BA_[t, u]).dot(Vr_[s].conj().T)) #4b
                        #subres += (exp(-2.j * p) * AA[t, s].conj().T.dot(l.dot(B[u])).dot(C_AA_Vrh[t, s, u])) #4c
        else:
            for s in xrange(self.q):
                #subres += C_AhlA[s, t].dot(B[s]).dot(Vr_[t].conj().T) #2 OK
                subres += tm.eps_r_noop(B[s], C_AhlA[s, :], Vr_) #2
                #+ exp(-1.j * p) * A[t].conj().T.dot(l.dot(B[s])).dot(C_A_Vrh_[t, s]) #4 OK with 3
                subres += exp(-1.j * p) * tm.eps_l_noop(l.dot(B[s]), A, C_A_Vrh_[s, :]) #4
                #+ exp(-2.j * p) * A[s].conj().T.dot(Mh.dot(C_[s, t])).dot(Vr_[t].conj().T)) #12
                subres += exp(-2.j * p) * A[s].conj().T.dot(Mh).dot(tm.eps_r_noop(eye, C_[s, :], Vr_)) #12
                    
        res += l_sqrt_i.dot(subres)
        
        res += l_sqrt.dot(tm.eps_r_noop(K__r, B, Vri_)) #5
        
        res += l_sqrt_i.dot(K_l.dot(tm.eps_r_noop(r__sqrt, B, V_))) #6
        
        res += sp.exp(-1.j * p) * l_sqrt_i.dot(Mh.dot(tm.eps_r_noop(K__r, A_, Vri_))) #8
        
        y1 = sp.exp(+1.j * p) * tm.eps_r_noop(K__r, B, A_) #7
        
        if self.ham_sites == 3:
            tmp = sp.exp(+1.j * p) * BAA_ + sp.exp(+2.j * p) * ABA_ + sp.exp(+3.j * p) * AAB #9, #11, #11b
            y = y1 + tm.eps_r_op_3s_C123_AAA456(r_, tmp, C_)
        elif self.ham_sites == 2:
            tmp = sp.exp(+1.j * p) * BA_ + sp.exp(+2.j * p) * AB #9, #11
            y = y1 + tm.eps_r_op_2s_AA12_C34(r_, tmp, C_conj)
        
        if pseudo:
            y = y - m.adot(l, y) * r_
        y_pi = pinv_1mE(y, [A], [A_], l, r_, p=p, left=False, 
                        pseudo=pseudo, out=y_pi_prev, tol=self.pinv_tol, 
                        solver=pinv_solver, use_CUDA=self.pinv_CUDA,
                        CUDA_use_batch=self.pinv_CUDA_batch,
                        sanity_checks=self.sanity_checks, sc_data='y_pi')
        #print m.adot(l, y_pi)
        if self.sanity_checks:
            z = y_pi - sp.exp(+1.j * p) * tm.eps_r_noop(y_pi, A, A_)
            tst = la.norm((y - z).ravel()) / la.norm(y.ravel())
            if tst > self.sanity_tol:
                log.warning("Sanity Fail in calc_BHB! Bad x_pi. Off by: %g", tst)
        
        res += l_sqrt.dot(tm.eps_r_noop(y_pi, A, Vri_))
        
        if self.sanity_checks:
            expval = m.adot(x, res) / m.adot(x, x)
            #print "expval = " + str(expval)
            if expval < -self.sanity_tol:
                log.warning("Sanity Fail in calc_BHB! H is not pos. semi-definite (%s)", expval)
            if abs(expval.imag) > self.sanity_tol:
                log.warning("Sanity Fail in calc_BHB! H is not Hermitian (%s)", expval)
        
        return res, M, y_pi   
    
    def matvec(self, v):
        x = v.reshape((self.D, (self.q - 1)*self.D))
        
        self.calls += 1
        log.debug("Calls: %u", self.calls)
        
        res, self.M_prev, self.y_pi_prev = self.calc_BHB(x, self.p, self.tdvp, 
                                                         self.tdvp2, 
                                                         self.prereq,
                                                         M_prev=self.M_prev, 
                                                         y_pi_prev=self.y_pi_prev,
                                                         pinv_solver=self.pinv_solver)
        
        return res.ravel()

def get_Aop(A, op_tp, index, conj=False):
    Aops = []
    for op_tp_ in op_tp:
        if index < 2:
            if conj:
                op = op_tp_[index].conj()
            else:
                op = op_tp_[index]
            A_op = op.T.dot(A.reshape((A.shape[0], A.shape[1] * A.shape[2]))).reshape(A.shape)
        else:
            if conj:
                op = op_tp_[index - 2].conj()
            else:
                op = op_tp_[index - 2]
            A_op = op.dot(A.reshape((A.shape[0], A.shape[1] * A.shape[2]))).reshape(A.shape)
        Aops.append(A_op)
        
    return Aops

def get_A_ops(A0, A1, op_tp, conj=False):
    A_ops_ = []
    
    if conj:
        for op_tp_ in op_tp:
            A0_op0 = op_tp_[0].conj().T.dot(A0.reshape((A0.shape[0], A0.shape[1] * A0.shape[2]))).reshape(A0.shape)
            A1_op1 = op_tp_[1].conj().T.dot(A1.reshape((A1.shape[0], A1.shape[1] * A1.shape[2]))).reshape(A1.shape)
            A_ops_.append([A0_op0, A1_op1])        
    else:
        for op_tp_ in op_tp:
            A0_op0 = op_tp_[0].dot(A0.reshape((A0.shape[0], A0.shape[1] * A0.shape[2]))).reshape(A0.shape)
            A1_op1 = op_tp_[1].dot(A1.reshape((A1.shape[0], A1.shape[1] * A1.shape[2]))).reshape(A1.shape)
            A_ops_.append([A0_op0, A1_op1])
        
    return A_ops_

class Excite_H_Op_tp:
    def __init__(self, tdvp, tdvp2, p, pinv_tol=1E-12,
                 sanity_checks=False, sanity_tol=1E-12):
        """Creates an Excite_H_Op object, which is a LinearOperator.
        
        This wraps the effective Hamiltonian in terms of MPS tangent vectors
        as a LinearOperator that can be used with SciPy's sparse linear
        algebra routines.
        
        Parameters
        ----------
        tdvp : EvoMPS_TDVP_Uniform
            tdvp object providing the required operations in the matrix representation.
        tdvp2 : EvoMPS_TDVP_Uniform
            Second tdvp object (can be the same as tdvp), for example containing a different ground state.
        p : float
            Momentum in units of inverse lattice spacing.
        """
        assert tdvp.L == 1, 'Excite_H_Op only supports blocks of size 1'
        
        assert tdvp.L == tdvp2.L, 'Block sizes must match!'
        assert tdvp.q == tdvp2.q, 'Hilbert spaces must match!'
        assert tdvp.D == tdvp2.D, 'Bond-dimensions must match!'
        
        self.tdvp = tdvp
        self.tdvp2 = tdvp2
        self.p = p
        
        self.D = tdvp.D
        self.q = tdvp.q
        self.ham_sites = tdvp.ham_sites
        self.ham_tp = tdvp.ham_tp
        
        self.sanity_checks = sanity_checks
        self.sanity_tol = sanity_tol
        
        self.pinv_solver = None
        self.pinv_tol = pinv_tol
        self.pinv_CUDA = False
        self.pinv_CUDA_batch = False
        
        d = (self.q - 1) * self.D**2
        self.shape = (d, d)
        
        self.dtype = np.dtype(tdvp.typ)
        
        self.prereq = (self.calc_BHB_prereq(tdvp, tdvp2))
        
        self.calls = 0
        
        self.M_prev = None
        self.y_pi_prev = None
    
    def calc_BHB_prereq(self, tdvp, tdvp2):
        """Calculates prerequisites for the application of the effective Hamiltonian in terms of tangent vectors.
        
        This is called (indirectly) by the self.excite.. functions.
        
        Parameters
        ----------
        tdvp2: EvoMPS_TDVP_Uniform
            Second state (may be the same, or another ground state).
            
        Returns
        -------
        A lot of stuff.
        """
        l = tdvp.l[0]
        r_ = tdvp2.r[0]
        r__sqrt = tdvp2.r_sqrt[0]
        r__sqrt_i = tdvp2.r_sqrt_i[0]
        A = tdvp.A[0]
        A_ = tdvp2.A[0]
            
        #Note: V has ~ D**2 * q**2 elements. We avoid making any copies of it except this one.
        #      This one is only needed because low-level routines force V_[s] to be contiguous.
        #      TODO: Store V instead of Vsh in tdvp_uniform too...
        V_ = sp.transpose(tdvp2.Vsh[0], axes=(0, 2, 1)).conj().copy(order='C')
                
        if self.ham_sites == 2:
            #eyeham = m.eyemat(self.q, dtype=sp.complex128)
            eyeham = sp.eye(self.q, dtype=sp.complex128)
            #diham = m.simple_diag_matrix(sp.repeat([-tdvp.h_expect.real], self.q))
            diham = -tdvp.h_expect.real * sp.eye(self.q, dtype=sp.complex128)
            _ham_tp = self.ham_tp + [[diham, eyeham]]  #subtract norm dof
            
            Ao1 = get_Aop(A, _ham_tp, 2, conj=False)
            
            AhlAo1 = [tm.eps_l_op_1s(l, A, A, o1.conj().T) for o1, o2 in _ham_tp]
            
            A_o2c = get_Aop(A_, _ham_tp, 1, conj=True)
            
            Ao1c = get_Aop(A, _ham_tp, 0, conj=True)
            
            A_Vr_ho2 = [tm.eps_r_op_1s(r__sqrt, A_, V_, o2) for o1, o2 in _ham_tp]
            
            A_A_o12c = get_A_ops(A_, A_, _ham_tp, conj=True)
            
            A_o1 = get_Aop(A_, _ham_tp, 2, conj=False)
            tmp = sp.empty((A_.shape[1], V_.shape[1]), dtype=A.dtype, order='C')
            tmp2 = sp.empty((A_.shape[1], A_o2c[0].shape[1]), dtype=A.dtype, order='C')
            rhs10 = 0
            for al in xrange(len(A_o1)):
                tmp2 = tm.eps_r_noop_inplace(r_, A_, A_o2c[al], tmp2)
                tmp3 = m.mmul(tmp2, r__sqrt_i)
                rhs10 += tm.eps_r_noop_inplace(tmp3, A_o1[al], V_, tmp)
                
            return V_, AhlAo1, A_o2c, Ao1, Ao1c, A_Vr_ho2, A_A_o12c, rhs10, _ham_tp
            
        elif self.ham_sites == 3:
            return

    def calc_BHB(self, x, p, tdvp, tdvp2, prereq,
                    M_prev=None, y_pi_prev=None, pinv_solver=None):
        if pinv_solver is None:
            pinv_solver = las.gmres
            
        if self.ham_sites == 3:
            return
        else:
            V_, AhlAo1, A_o2c, Ao1, Ao1c, A_Vr_ho2, A_A_o12c, rhs10, _ham_tp = prereq
        
        A = tdvp.A[0]
        A_ = tdvp2.A[0]
        
        l = tdvp.l[0]
        r_ = tdvp2.r[0]
        
        l_sqrt = tdvp.l_sqrt[0]
        l_sqrt_i = tdvp.l_sqrt_i[0]
        
        r__sqrt = tdvp2.r_sqrt[0]
        r__sqrt_i = tdvp2.r_sqrt_i[0]
        
        K__r = tdvp2.K[0]
        K_l = tdvp.K_left[0]
        
        pseudo = tdvp2 is tdvp
        
        B = tdvp2.get_B_from_x(x, tdvp2.Vsh[0], l_sqrt_i, r__sqrt_i)
        
        #Skip zeros due to rank-deficiency
        if la.norm(B) == 0:
            return sp.zeros_like(x), M_prev, y_pi_prev
        
        if self.sanity_checks:
            tst = tm.eps_r_noop(r_, B, A_)
            if not la.norm(tst) > self.sanity_tol:
                log.warning("Sanity check failed: Gauge-fixing violation! " 
                            + str(la.norm(tst)))

        if self.sanity_checks:
            B2 = np.zeros_like(B)
            for s in xrange(self.q):
                B2[s] = l_sqrt_i.dot(x.dot(m.mmul(V_[s], r__sqrt_i)))
            if la.norm(B - B2) / la.norm(B) > self.sanity_tol:
                log.warning("Sanity Fail in calc_BHB! Bad Vri!")

        y = tm.eps_l_noop(l, B, A)
        
#        if pseudo:
#            y = y - m.adot(r_, y) * l #should just = y due to gauge-fixing
        M = pinv_1mE(y, [A_], [A], l, r_, p=-p, left=True, pseudo=pseudo, 
                     out=M_prev, tol=self.pinv_tol, solver=pinv_solver,
                     use_CUDA=self.pinv_CUDA, CUDA_use_batch=self.pinv_CUDA_batch,
                     sanity_checks=self.sanity_checks, sc_data='M')
        
        #print m.adot(r, M)
        if self.sanity_checks:
            y2 = M - sp.exp(+1.j * p) * tm.eps_l_noop(M, A_, A)
            norm = la.norm(y.ravel())
            if norm == 0:
                norm = 1
            tst = la.norm(y - y2) / norm
            if tst > self.sanity_tol:
                log.warning("Sanity Fail in calc_BHB! Bad M. Off by: %g", tst)
#        if pseudo:
#            M = M - l * m.adot(r_, M)
        Mh = M.conj().T.copy(order='C')
        
        res_ls = 0
        res_lsi = 0
        
        exp = sp.exp
        
        if self.ham_sites == 3:
            pass
        else:
            Bo1 = get_Aop(B, _ham_tp, 2, conj=False)
            tmp = sp.empty((B.shape[1], V_.shape[1]), dtype=A.dtype, order='C')
            tmp2 = sp.empty((A_.shape[1], A_o2c[0].shape[1]), dtype=A.dtype, order='C')
            tmp3 = sp.empty_like(tmp2, order='C')
            for al in xrange(len(Bo1)):
                tmp3 = m.dot_inplace(tm.eps_r_noop_inplace(r_, A_, A_o2c[al], tmp2), r__sqrt_i, tmp3)
                res_ls += tm.eps_r_noop_inplace(tmp3, Bo1[al], V_, tmp) #1
                
                tmp3 = m.dot_inplace(tm.eps_r_noop_inplace(r_, B, A_o2c[al], tmp2), r__sqrt_i, tmp3)
                tmp = tm.eps_r_noop_inplace(tmp3, Ao1[al], V_, tmp) #3
                tmp *= exp(+1.j * p)
                res_ls += tmp
            del(tmp)
            del(tmp2)
            del(tmp3)
                        
        res_lsi += sp.exp(-1.j * p) * Mh.dot(rhs10) #10
        
        if self.ham_sites == 3:
            pass
        else:
            Bo2 = get_Aop(B, _ham_tp, 3, conj=False)
            for al in xrange(len(AhlAo1)):
                res_lsi += AhlAo1[al].dot(tm.eps_r_noop(r__sqrt, Bo2[al], V_)) #2
                res_lsi += exp(-1.j * p) * tm.eps_l_noop(l, Ao1c[al], B).dot(A_Vr_ho2[al]) #4
                res_lsi += exp(-2.j * p) * tm.eps_l_noop(Mh, Ao1c[al], A_).dot(A_Vr_ho2[al]) #12
                    
        K__rri = m.mmul(K__r, r__sqrt_i)
        res_ls += tm.eps_r_noop(K__rri, B, V_) #5
        
        res_lsi += K_l.dot(tm.eps_r_noop(r__sqrt, B, V_)) #6
        
        res_lsi += sp.exp(-1.j * p) * Mh.dot(tm.eps_r_noop(K__rri, A_, V_)) #8

        y1 = sp.exp(+1.j * p) * tm.eps_r_noop(K__r, B, A_) #7
        
        if self.ham_sites == 3:
            pass
        elif self.ham_sites == 2:
            tmp = 0
            for al in xrange(len(A_A_o12c)):
                tmp += sp.exp(+1.j * p) * tm.eps_r_noop(tm.eps_r_noop(r_, A_, A_A_o12c[al][1]), B, A_A_o12c[al][0]) #9
                tmp += sp.exp(+2.j * p) * tm.eps_r_noop(tm.eps_r_noop(r_, B, A_A_o12c[al][1]), A, A_A_o12c[al][0]) #11
            y = y1 + tmp #7, 9, 11
            del(tmp)
        
        if pseudo:
            y = y - m.adot(l, y) * r_
        y_pi = pinv_1mE(y, [A], [A_], l, r_, p=p, left=False, 
                        pseudo=pseudo, out=y_pi_prev, tol=self.pinv_tol, 
                        solver=pinv_solver, use_CUDA=self.pinv_CUDA,
                        CUDA_use_batch=self.pinv_CUDA_batch,
                        sanity_checks=self.sanity_checks, sc_data='y_pi')
        #print m.adot(l, y_pi)
        if self.sanity_checks:
            z = y_pi - sp.exp(+1.j * p) * tm.eps_r_noop(y_pi, A, A_)
            tst = la.norm((y - z).ravel()) / la.norm(y.ravel())
            if tst > self.sanity_tol:
                log.warning("Sanity Fail in calc_BHB! Bad x_pi. Off by: %g", tst)
        
        res_ls += tm.eps_r_noop(m.mmul(y_pi, r__sqrt_i), A, V_)
        
        res = l_sqrt.dot(res_ls)
        res += l_sqrt_i.dot(res_lsi)
        
        if self.sanity_checks:
            expval = m.adot(x, res) / m.adot(x, x)
            #print "expval = " + str(expval)
            if expval < -self.sanity_tol:
                log.warning("Sanity Fail in calc_BHB! H is not pos. semi-definite (%s)", expval)
            if abs(expval.imag) > self.sanity_tol:
                log.warning("Sanity Fail in calc_BHB! H is not Hermitian (%s)", expval)
        
        return res, M, y_pi  
    
    def matvec(self, v):
        x = v.reshape((self.D, (self.q - 1)*self.D))
        
        self.calls += 1
        log.debug("Calls: %u", self.calls)
        
        res, self.M_prev, self.y_pi_prev = self.calc_BHB(x, self.p, self.tdvp, 
                                                         self.tdvp2, 
                                                         self.prereq,
                                                         M_prev=self.M_prev, 
                                                         y_pi_prev=self.y_pi_prev,
                                                         pinv_solver=self.pinv_solver)
        
        return res.ravel()

        
def _excite_correlation_1s_1s(AL, AR, B, lL, rR, op1, op2, d, g):
    pseudo = AL is AR
    
    BBL = pinv_1mE(tm.eps_l_noop(lL, B, B), [AR], [AR], lL, rR, p=0, 
                   left=True, pseudo=True)
                   
    B1L = pinv_1mE(tm.eps_l_noop(lL, B, AL), [AR], [AL], lL, rR, p=0, 
                   left=True, pseudo=pseudo)
                   
    B2L = pinv_1mE(tm.eps_l_noop(lL, AL, B), [AL], [AR], lL, rR, p=0, 
                   left=True, pseudo=pseudo)
    
    B1B2L = pinv_1mE(tm.eps_l_noop(B1L, AR, B), [AR], [AR], lL, rR, p=0, 
                     left=True, pseudo=True)
                     
    B2B1L = pinv_1mE(tm.eps_l_noop(B2L, B, AR), [AR], [AR], lL, rR, p=0, 
                     left=True, pseudo=True)
    
    BBR = pinv_1mE(tm.eps_r_noop(rR, B, B), [AL], [AL], lL, rR, p=0, 
                   left=False, pseudo=True)
                   
    print "gauge fixing:", la.norm(tm.eps_r_noop(rR, B, AR))
                   
    res = [0.] * (d + 1)
    
    #B's on the left terms (0, 1, 2, 5)
    res[0] += m.adot(tm.eps_l_op_1s(BBL + B1B2L + B2B1L, AR, AR, op1.dot(op2) - g[0] * sp.eye(len(op1))), rR)
    res[0] += m.adot(tm.eps_l_op_1s(B1L, AR, B, op1.dot(op2)), rR)
    res[0] += m.adot(tm.eps_l_op_1s(B2L, B, AR, op1.dot(op2)), rR)
    res[0] += m.adot(tm.eps_l_op_1s(lL, B, B, op1.dot(op2) - g[0] * sp.eye(len(op1))), rR)
    
    l1 = tm.eps_l_op_1s(BBL + B1B2L + B2B1L, AR, AR, op1)
    l1 += tm.eps_l_op_1s(B1L, AR, B, op1)
    l1 += tm.eps_l_op_1s(B2L, B, AR, op1)
    l1 += tm.eps_l_op_1s(lL, B, B, op1)
    for n in xrange(1, d + 1):
        res[n] += m.adot(l1, tm.eps_r_op_1s(rR, AR, AR, op2))
        res[n] -= 2 * g[n]
        if n == d:
            break
        l1 = tm.eps_l_noop(l1, AR, AR)
        
    print 1, res
    
    #Terms 3, 4, 6, 7, a
    ls = [None] * (d + 1)
    ls[0] = tm.eps_l_op_1s(B1L, AR, AL, op1) + tm.eps_l_op_1s(lL, B, AL, op1)
    for k in xrange(1, d + 1):
        ls[k] = tm.eps_l_noop(ls[k - 1], AR, AL)
        
    rs = [None] * (d + 1)
    rs[d - 1] = tm.eps_r_op_1s(rR, AR, AR, op2)
    for k in xrange(d - 1, 0, -1):
        rs[k - 1] = tm.eps_r_noop(rs[k], AR, AR)
        
    for n in xrange(2, d + 1):
        for k in xrange(2, n):
            res[n] += m.adot(ls[k - 1], tm.eps_r_noop(rs[-(n + 1):][k], AR, B))
        
        res[n] += m.adot(ls[n - 1], tm.eps_r_op_1s(rR, AR, B, op2))
        
    print 2, res
        
    #Terms 3, 4, 6, 7, b
    ls = [None] * (d + 1)
    ls[0] = tm.eps_l_op_1s(B2L, AL, AR, op1) + tm.eps_l_op_1s(lL, AL, B, op1)
    for k in xrange(1, d + 1):
        ls[k] = tm.eps_l_noop(ls[k - 1], AL, AR)
        
    for n in xrange(2, d + 1):
        for k in xrange(2, n):
            res[n] += m.adot(ls[k - 1], tm.eps_r_noop(rs[-(n + 1):][k], B, AR))
        
        res[n] += m.adot(ls[n - 1], tm.eps_r_op_1s(rR, B, AR, op2))
        
    print 3, res
    
    #Term 8
    ls = [None] * (d + 1)
    ls[0] = tm.eps_l_op_1s(lL, AL, AL, op1)
    for k in xrange(1, d + 1):
        ls[k] = tm.eps_l_noop(ls[k - 1], AL, AL)
        
    for n in xrange(2, d + 1):
        for k in xrange(2, n):
            res[n] += m.adot(ls[k - 1], tm.eps_r_noop(rs[-(n + 1):][k], B, B))
            res[n] -= g[n]
            
    print 4, res
            
    #Terms 9 and 10
    for n in xrange(2, d + 1):
        for k in xrange(2, n):
            lj = tm.eps_l_noop(ls[k - 1], AL, B)
            for j in xrange(k + 1, n):
                res[n] += m.adot(lj, tm.eps_r_noop(rs[-(n + 1):][j], B, AR))
                lj = tm.eps_l_noop(lj, AL, AR)
            res[n] += m.adot(lj, tm.eps_r_op_1s(rR, B, AR, op2))

            lj = tm.eps_l_noop(ls[k - 1], B, AL)
            for j in xrange(k + 1, n):
                res[n] += m.adot(lj, tm.eps_r_noop(rs[-(n + 1):][j], AR, B))
                lj = tm.eps_l_noop(lj, AL, AR)
            res[n] += m.adot(lj, tm.eps_r_op_1s(rR, AR, B, op2))
            
    print 5, res
    
    #Term 11
    res[0] += m.adot(tm.eps_l_op_1s(lL, AL, AL, op1.dot(op2) - g[0] * sp.eye(len(op1))), BBR)
    for n in xrange(1, d + 1):
        res[n] += m.adot(tm.eps_l_op_1s(ls[n - 1], AL, AL, op2), BBR)
        res[n] -= g[n]
    
    return res
    
def _excite_expect_2s_tp_sep(AL, AR, B, lL, rR, op, d, n1, n2):
    ls = [None] * (d + 1)
    ls[0] = lL
    rs = [None] * (d + 1)
    rs[-1] = rR
    
    As1 = [None] + [AL] * (n1 - 1) + [B] + [AR] * (d - n1 + 1)
    As2 = [None] + [AL] * (n2 - 1) + [B] + [AR] * (d - n2 + 1)
    
    for n in xrange(1, d):
        ls[n] = tm.eps_l_noop(ls[n - 1], As1[n], As2[n])
        
    for n in xrange(d, 0, -1):
        rs[n - 1] = tm.eps_r_noop(rs[n], As1[n], As2[n])
        
    Cs1 = [None] * (d + 1)
    for n in xrange(1, d):
        Cs1[n] = tm.calc_C_tp(op, As1[n], As1[n + 1])
        
    res = [m.adot(ls[n - 1], tm.eps_r_op_2s_C12_tp(rs[n + 1], Cs1[n], As2[n], As2[n + 1]))
           for n in xrange(1, d)]
               
    return sp.array(res)
