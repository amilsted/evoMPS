# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 17:29:27 2011

@author: Ashley Milsted

"""
import numpy as np
import scipy as sp
import scipy.linalg as la
import scipy.sparse.linalg as las
import scipy.optimize as opti
import tdvp_common as tm
import matmul as m
from mps_uniform import EvoMPS_MPS_Uniform, EOp

try:
    import tdvp_calc_C as tc
except ImportError:
    tc = None
    print "Warning! Cython version of Calc_C was not available. Performance may suffer for large q."


class PPInvOp:    
    def __init__(self, tdvp, p, left, pseudo, A1, A2, r):
        self.A1 = A1
        self.A2 = A2
        self.l = tdvp.l
        self.r = r
        self.p = p
        self.left = left
        self.pseudo = pseudo
        
        self.D = tdvp.D
        
        self.shape = (self.D**2, self.D**2)
        
        self.dtype = np.dtype(tdvp.typ)
        
        self.out = np.empty_like(self.l)
    
    def matvec(self, v):
        x = v.reshape((self.D, self.D))
        
        if self.left: #Multiplying from the left, but x is a col. vector, so use mat_dagger
            Ehx = tm.eps_l_noop_inplace(x, self.A1, self.A2, self.out)
            if self.pseudo:
                QEQhx = Ehx - self.l * m.adot(self.r, x)
                res = x - sp.exp(-1.j * self.p) * QEQhx
            else:
                res = x - sp.exp(-1.j * self.p) * Ehx
        else:
            Ex = tm.eps_r_noop_inplace(x, self.A1, self.A2, self.out)
            if self.pseudo:
                QEQx = Ex - self.r * m.adot(self.l, x)
                res = x - sp.exp(1.j * self.p) * QEQx
            else:
                res = x - sp.exp(1.j * self.p) * Ex
        
        return res.ravel()
        
class Excite_H_Op:
    def __init__(self, tdvp, donor, p):
        self.donor = donor
        self.p = p
        
        self.D = tdvp.D
        self.q = tdvp.q
        
        d = (self.q - 1) * self.D**2
        self.shape = (d, d)
        
        self.dtype = np.dtype(tdvp.typ)
        
        self.prereq = (tdvp.calc_BHB_prereq(donor))
        
        self.calc_BHB = tdvp.calc_BHB
        
        self.calls = 0
        
        self.M_prev = None
        self.y_pi_prev = None
    
    def matvec(self, v):
        x = v.reshape((self.D, (self.q - 1)*self.D))
        
        self.calls += 1
        print "Calls: %u" % self.calls
        
        res, self.M_prev, self.y_pi_prev = self.calc_BHB(x, self.p, self.donor, 
                                                         *self.prereq,
                                                         M_prev=self.M_prev, 
                                                         y_pi_prev=self.y_pi_prev)
        
        return res.ravel()
        
class EvoMPS_TDVP_Uniform(EvoMPS_MPS_Uniform):
        
    def __init__(self, D, q, h_nn, typ=None):
        
        super(EvoMPS_TDVP_Uniform, self).__init__(D, q, typ=typ)
                
        self.h_nn = h_nn
        self.h_nn_cptr = None
        self.h_nn_mat = None
                        
        self.eta = 0
    
    def _init_arrays(self, D, q):
        super(EvoMPS_TDVP_Uniform, self)._init_arrays(D, q)
        
        self.C = np.zeros((q, q, D, D), dtype=self.typ, order=self.odr)
        
        self.K = np.ones_like(self.A[0])
        self.K_left = None
            
    def gen_h_matrix(self):
        """Generates a matrix form for h_nn, which can speed up parts of the
        algorithm by avoiding excess loops and python calls.
        """
        hv = np.vectorize(self.h_nn, otypes=[np.complex128])
        self.h_nn_mat = np.fromfunction(hv, (self.q, self.q, self.q, self.q))  

    def calc_C(self):
        if not tc is None and not self.h_nn_cptr is None and np.iscomplexobj(self.C):
            self.C = tc.calc_C(self.AA, self.h_nn_cptr, self.C)
        elif not self.h_nn_mat is None:
            self.C[:] = tm.calc_C_mat_op_AA(self.h_nn_mat, self.AA)
        else:
            self.C[:] = tm.calc_C_func_op_AA(self.h_nn, self.AA)
    
    def calc_PPinv(self, x, p=0, out=None, left=False, A1=None, A2=None, r=None, pseudo=True):
        if out is None:
            out = np.ones_like(self.A[0])
            
        if A1 is None:
            A1 = self.A
            
        if A2 is None:
            A2 = self.A
            
        if r is None:
            r = self.r
        
        op = PPInvOp(self, p, left, pseudo, A1, A2, r)
        
        res = out.ravel()
        x = x.ravel()
        
        res, info = las.bicgstab(op, x, x0=res, maxiter=2000, 
                                 tol=self.itr_rtol) #tol: norm( b - A*x ) / norm( b )
        
        if info > 0:
            print "Warning: Did not converge on solution for ppinv!"
        
        #Test
        if self.sanity_checks:
            RHS_test = op.matvec(res)
            d = la.norm(RHS_test - x) / la.norm(x)
            if not d < self.itr_rtol*self.check_fac:
                print "Sanity check failed: Bad ppinv solution! Off by: " + str(
                        d)
        
        res = res.reshape((self.D, self.D))
            
        if False and self.sanity_checks and self.D < 16:
            pinvE = self.pinvE_brute(p, A1, A2, r, pseudo)
            
            if left:
                res_brute = (x.reshape((1, self.D**2)).conj().dot(pinvE)).ravel().conj().reshape((self.D, self.D))
                #res_brute = (pinvE.T.dot(x)).reshape((self.D, self.D))
            else:
                res_brute = pinvE.dot(x).reshape((self.D, self.D))
            
            if not np.allclose(res, res_brute):
                print "Sanity Fail in calc_PPinv (left: %s): Bad brute check! Off by: %g" % (str(left), la.norm(res - res_brute))
        
        out[:] = res
        
        return out
        
    def calc_K(self):
        Hr = tm.eps_r_op_2s_C12_AA34(self.r, self.C, self.AA)
        
        self.h = m.adot(self.l, Hr)
        
        QHr = Hr - self.r * self.h
        
        self.calc_PPinv(QHr, out=self.K)
        
        if self.sanity_checks:
            Ex = tm.eps_r_noop(self.K, self.A, self.A)
            QEQ = Ex - self.r * m.adot(self.l, self.K)
            res = self.K - QEQ
            if not np.allclose(res, QHr):
                print "Sanity check failed: Bad K!"
                print "Off by: " + str(la.norm(res - QHr))
        
    def calc_K_l(self):
        #Using C is allowed because h is Hermitian
        lH = tm.eps_l_op_2s_AA12_C34(self.l, self.AA, self.C)
        
        h = m.adot(self.r, lH)
        
        lHQ = lH - self.l * h
        
        self.K_left = self.calc_PPinv(lHQ, left=True, out=self.K_left)
        
        if self.sanity_checks:
            xE = tm.eps_l_noop(self.K_left, self.A, self.A)
            QEQ = xE - self.l * m.adot(self.r, self.K_left)
            res = self.K_left - QEQ
            if not np.allclose(res, lHQ):
                print "Sanity check failed: Bad K_left!"
                print "Off by: " + str(la.norm(res - lHQ))
        
        return self.K_left, h
        
    def calc_x(self, l_sqrt, l_sqrt_i, r_sqrt, r_sqrt_i, Vsh, out=None):
        if out is None:
            out = np.zeros((self.D, (self.q - 1) * self.D), dtype=self.typ, 
                           order=self.odr)
        
        tmp = np.zeros_like(out)
        for s in xrange(self.q):
            tmp2 = m.mmul(self.A[s], self.K)
            for t in xrange(self.q):
                tmp2 += m.mmul(self.C[s, t], self.r, m.H(self.A[t]))
            tmp += m.mmul(tmp2, r_sqrt_i, Vsh[s])
        out += l_sqrt.dot(tmp)
        
        tmp.fill(0)
        for s in xrange(self.q):
            tmp2.fill(0)
            for t in xrange(self.q):
                tmp2 += m.mmul(m.H(self.A[t]), self.l, self.C[t, s])
            tmp += m.mmul(tmp2, r_sqrt, Vsh[s])
        out += l_sqrt_i.dot(tmp)
        
        return out
        
    def get_B_from_x(self, x, Vsh, l_sqrt_i, r_sqrt_i, out=None):
        if out is None:
            out = np.zeros_like(self.A)
            
        for s in xrange(self.q):
            out[s] = m.mmul(l_sqrt_i, x, m.H(Vsh[s]), r_sqrt_i)
            
        return out
        
    def calc_l_r_roots(self):
        self.l_sqrt, self.l_sqrt_i, self.r_sqrt, self.r_sqrt_i = tm.calc_l_r_roots(self.l, self.r, self.sanity_checks)
        
    def calc_B(self, set_eta=True):
        self.calc_l_r_roots()
                
        self.Vsh = tm.calc_Vsh(self.A, self.r_sqrt, sanity_checks=self.sanity_checks)
        
        self.x = tm.calc_x(self.K, self.C, self.C, self.r, self.l, self.A, 
                           self.A, self.A, self.l_sqrt, self.l_sqrt_i,
                           self.r_sqrt, self.r_sqrt_i, self.Vsh)
        
        if set_eta:
            self.eta = sp.sqrt(m.adot(self.x, self.x))
        
        B = self.get_B_from_x(self.x, self.Vsh, self.l_sqrt_i, self.r_sqrt_i)
        
        if self.sanity_checks:
            #Test gauge-fixing:
            tst = tm.eps_r_noop(self.r, B, self.A)
            if not np.allclose(tst, 0):
                print "Sanity check failed: Gauge-fixing violation!"

        return B
        
    def update(self, restore_CF=True):
        super(EvoMPS_TDVP_Uniform, self).update(restore_CF=restore_CF)
        self.calc_C()
        self.calc_K()
        
    def take_step(self, dtau, B=None):
        if B is None:
            B = self.calc_B()
        
        self.A += -dtau * B
            
    def take_step_RK4(self, dtau, B_i=None):
        def update():
            self.calc_lr()
            #self.restore_CF() #this really messes things up...
            self.calc_AA()
            self.calc_C()
            self.calc_K()            

        A0 = self.A.copy()
            
        B_fin = np.empty_like(self.A)

        if not B_i is None:
            B = B_i
        else:
            B = self.calc_B() #k1
        B_fin = B
        self.A = A0 - dtau/2 * B
        
        update()
        
        B = self.calc_B(set_eta=False) #k2                
        self.A = A0 - dtau/2 * B
        B_fin += 2 * B         
            
        update()
            
        B = self.calc_B(set_eta=False) #k3                
        self.A = A0 - dtau * B
        B_fin += 2 * B

        update()
        
        B = self.calc_B(set_eta=False) #k4
        B_fin += B
            
        self.A = A0 - dtau /6 * B_fin
            
    def pinvE_brute(self, p, A1, A2, r, pseudo=True):
        E = np.zeros((self.D**2, self.D**2), dtype=self.typ)

        for s in xrange(self.q):
            E += np.kron(A1[s], A2[s].conj())
        
        l = np.asarray(self.l)
        r = np.asarray(r)
        
        if pseudo:
            QEQ = E - r.reshape((self.D**2, 1)).dot(l.reshape((1, self.D**2)).conj())
        else:
            QEQ = E
        
        EyemE = np.eye(self.D**2, dtype=self.typ) - sp.exp(1.j * p) * QEQ
        
        return la.inv(EyemE)
        
    def calc_BHB_prereq(self, donor):
        l = self.l
        r_ = donor.r
        r__sqrt = donor.r_sqrt
        r__sqrt_i = donor.r_sqrt_i
        A = self.A
        A_ = donor.A
        AA_ = donor.AA
                    
#        def h_nn(s,t,u,v):
#            h = self.h_nn(s,t,u,v)
#            if s == u and t == v:
#                h -= self.h.real
#            return h
#        h_nn = np.vectorize(h_nn, otypes=[self.typ])
#        h_nn_mat = np.fromfunction(h_nn, (self.q, self.q, self.q, self.q), dtype=self.typ)
        eyed = np.eye(self.q**2)
        eyed = eyed.reshape((self.q, self.q, self.q, self.q))
        h_nn_mat = self.h_nn_mat - self.h.real * eyed
        def h_nn(s,t,u,v):
            return h_nn_mat[s,t,u,v]
            
        V_ = sp.zeros((donor.Vsh.shape[0], donor.Vsh.shape[2], donor.Vsh.shape[1]), dtype=self.typ)
        for s in xrange(donor.q):
            V_[s] = m.H(donor.Vsh[s])
        
        Vri_ = sp.zeros_like(V_)
        for s in xrange(donor.q):
            Vri_[s] = r__sqrt_i.dot_left(V_[s])
            
        Vr_ = sp.zeros_like(V_)
        for s in xrange(donor.q):
            Vr_[s] = r__sqrt.dot_left(V_[s])
            
        C_AhlA = np.empty_like(self.C)
        for u in xrange(self.q):
            for s in xrange(self.q):
                C_AhlA[u, s] = m.H(A[u]).dot(l.dot(A[s]))
        C_AhlA = sp.tensordot(h_nn_mat, C_AhlA, ((2, 0), (0, 1)))
        
        C_A_Vrh_ = np.empty((self.q, self.q, A_.shape[1], Vr_.shape[1]), dtype=self.typ)
        for t in xrange(self.q):
            for v in xrange(self.q):
                C_A_Vrh_[t, v] = A_[t].dot(m.H(Vr_[v]))
        C_A_Vrh_ = sp.tensordot(h_nn_mat, C_A_Vrh_, ((1, 3), (0, 1)))
                
        C_Vri_A_ = np.empty((self.q, self.q, Vri_.shape[1], A_.shape[2]), dtype=self.typ)
        for s in xrange(self.q):
            for t in xrange(self.q):
                C_Vri_A_[s, t] = Vri_[s].dot(A_[t])
        C_Vri_A_ = sp.tensordot(h_nn_mat, C_Vri_A_, ((2, 3), (0, 1)))
        
        C = sp.tensordot(h_nn_mat, self.AA, ((2, 3), (0, 1)))

        C_ = sp.tensordot(h_nn_mat, AA_, ((2, 3), (0, 1)))
        
        rhs10 = tm.eps_r_op_2s_AA12_C34(r_, AA_, C_Vri_A_)
        
        #NOTE: These C's are good as C12 or C34, but only because h is Hermitian!
        
        return h_nn, h_nn_mat, C, C_, V_, Vr_, Vri_, C_Vri_A_, C_AhlA, C_A_Vrh_, rhs10
            
    def calc_BHB(self, x, p, donor, h_nn, h_nn_mat, C, C_, V_, Vr_, Vri_, 
                 C_Vri_A_, C_AhlA, C_A_Vrh_, rhs10, M_prev=None, y_pi_prev=None): 
        """For a good approx. ground state, H should be Hermitian pos. semi-def.
        """        
        A = self.A
        A_ = donor.A
        
        l = self.l
        r_ = donor.r
        
        l_sqrt = self.l_sqrt
        l_sqrt_i = self.l_sqrt_i
        
        r__sqrt = donor.r_sqrt
        r__sqrt_i = donor.r_sqrt_i
        
        K__r = donor.K
        K_l = self.K_left
        
        pseudo = donor is self
        
        B = donor.get_B_from_x(x, donor.Vsh, l_sqrt_i, r__sqrt_i)
        
        if self.sanity_checks:
            tst = tm.eps_r_noop(r_, B, A_)
            if not np.allclose(tst, 0):
                print "Sanity check failed: Gauge-fixing violation!"

        if self.sanity_checks:
            B2 = np.zeros_like(B)
            for s in xrange(self.q):
                B2[s] = l_sqrt_i.dot(x.dot(Vri_[s]))
            if not sp.allclose(B, B2, rtol=self.itr_rtol*self.check_fac,
                               atol=self.itr_atol*self.check_fac):
                print "Sanity Fail in calc_BHB! Bad Vri!"
            
        BA_ = tm.calc_AA(B, A_)
        AB = tm.calc_AA(self.A, B)
            
        y = tm.eps_l_noop(l, B, self.A)
        
        if pseudo:
            y = y - m.adot(r_, y) * l #should just = y due to gauge-fixing
        M = self.calc_PPinv(y, p=-p, left=True, A1=A_, r=r_, pseudo=pseudo, out=M_prev)
        #print m.adot(r, M)
        if self.sanity_checks:
            y2 = M - sp.exp(+1.j * p) * tm.eps_l_noop(M, A_, self.A)
            if not sp.allclose(y, y2):
                print "Sanity Fail in calc_BHB! Bad M. Off by: %g" % (la.norm((y - y2).ravel()) / la.norm(y.ravel()))
        Mh = m.H(M)

        res = l_sqrt.dot(
               tm.eps_r_op_2s_AA12_C34(r_, BA_, C_Vri_A_) #1 OK
               + sp.exp(+1.j * p) * tm.eps_r_op_2s_AA12_C34(r_, AB, C_Vri_A_) #3 OK with 4
              )
        
        res += sp.exp(-1.j * p) * l_sqrt_i.dot(Mh.dot(rhs10)) #10
        
        exp = sp.exp
        subres = sp.zeros_like(res)
        for s in xrange(self.q):
            for t in xrange(self.q):
                subres += (C_AhlA[s, t].dot(B[s]).dot(Vr_[t].conj().T) #2 OK
                         + exp(-1.j * p) * A[t].conj().T.dot(l.dot(B[s])).dot(C_A_Vrh_[s, t]) #4 OK with 3
                         + exp(-2.j * p) * A[s].conj().T.dot(Mh.dot(C_[s, t])).dot(Vr_[t].conj().T)) #12
        res += l_sqrt_i.dot(subres)
        
        res += l_sqrt.dot(tm.eps_r_noop(K__r, B, Vri_)) #5 OK
        
        res += l_sqrt_i.dot(m.H(K_l).dot(tm.eps_r_noop(r__sqrt, B, V_))) #6
        
        res += sp.exp(-1.j * p) * l_sqrt_i.dot(Mh.dot(tm.eps_r_noop(K__r, A_, Vri_))) #8
        
        y1 = sp.exp(+1.j * p) * tm.eps_r_noop(K__r, B, A_) #7
        y2 = sp.exp(+1.j * p) * tm.eps_r_op_2s_AA12_C34(r_, BA_, C_) #9
        y3 = sp.exp(+2.j * p) * tm.eps_r_op_2s_AA12_C34(r_, AB, C_) #11
        
        y = y1 + y2 + y3
        if pseudo:
            y = y - m.adot(l, y) * r_
        y_pi = self.calc_PPinv(y, p=p, A2=A_, r=r_, pseudo=pseudo, out=y_pi_prev)
        #print m.adot(l, y_pi)
        if self.sanity_checks:
            y2 = y_pi - sp.exp(+1.j * p) * tm.eps_r_noop(y_pi, self.A, A_)
            if not sp.allclose(y, y2):
                print "Sanity Fail in calc_BHB! Bad x_pi. Off by: %g" % (la.norm((y - y2).ravel()) / la.norm(y.ravel()))
        
        res += l_sqrt.dot(tm.eps_r_noop(y_pi, self.A, Vri_))
        
        if self.sanity_checks:
            expval = m.adot(x, res) / m.adot(x, x)
            #print "expval = " + str(expval)
            if expval < 0:
                print "Sanity Fail in calc_BHB! H is not pos. semi-definite (" + str(expval) + ")"
            if not abs(expval.imag) < 1E-9:
                print "Sanity Fail in calc_BHB! H is not Hermitian (" + str(expval) + ")"
        
        return res, M, y_pi
    
    def _prepare_excite_op_top_triv(self, p):
        self.calc_K_l()
        self.calc_l_r_roots()
        self.Vsh = tm.calc_Vsh(self.A, self.r_sqrt, sanity_checks=self.sanity_checks)
        
        op = Excite_H_Op(self, self, p)

        return op        
    
    def excite_top_triv(self, p, k=6, tol=0, max_itr=None, v0=None, ncv=None,
                        sigma=None,
                        which='SM', return_eigenvectors=False):
        self.gen_h_matrix()
        self.calc_K_l()
        self.calc_l_r_roots()
        self.Vsh = tm.calc_Vsh(self.A, self.r_sqrt, sanity_checks=self.sanity_checks)
        
        op = Excite_H_Op(self, self, p)
        res = las.eigsh(op, which=which, k=k, v0=v0, ncv=ncv,
                         return_eigenvectors=return_eigenvectors, 
                         maxiter=max_itr, tol=tol, sigma=sigma)
                          
        return res
    
    def excite_top_triv_brute(self, p):
        self.calc_K_l()
        self.calc_l_r_roots()
        self.Vsh = tm.calc_Vsh(self.A, self.r_sqrt, sanity_checks=self.sanity_checks)
        
        op = Excite_H_Op(self, self, p)
        
        x = np.empty(((self.q - 1)*self.D**2), dtype=self.typ)
        y = np.empty_like(x)
        
        H = np.zeros((x.shape[0], x.shape[0]), dtype=self.typ)
        
        #Only fill the lower triangle
        for i in xrange(x.shape[0]):
            x.fill(0)
            x[i] = 1
            for j in xrange(i + 1):
                y.fill(0)
                y[j] = 1
                
                H[i, j] = x.dot(op.matvec(y))

        #print np.allclose(H, m.H(H))
               
        return la.eigvalsh(H)


    def excite_top_nontriv(self, donor, p, k=6, tol=0, max_itr=None, v0=None,
                           which='SM', return_eigenvectors=False, sigma=None,
                           ncv=None):
        self.gen_h_matrix()
        donor.gen_h_matrix()
        self.calc_lr()
        self.restore_CF()
        donor.calc_lr()
        donor.restore_CF()
        
        #Phase-alignment
        if self.D == 1:
            ev = 0
            for s in xrange(self.q):
                ev += self.A[s] * donor.A[s].conj()
            donor.A *= ev / abs(ev)
        else:
            opE = EOp(donor, self.A, donor.A, False)
            ev = las.eigs(opE, which='LM', k=1)
            donor.A *= ev[0] / abs(ev[0])
        
        self.update()
        donor.update()

        self.calc_K_l()
        self.calc_l_r_roots()
        donor.calc_l_r_roots()
        donor.Vsh = tm.calc_Vsh(donor.A, donor.r_sqrt, sanity_checks=self.sanity_checks)
        
        op = Excite_H_Op(self, donor, p)
                            
        res = las.eigsh(op, sigma=sigma, which=which, k=k, v0=v0,
                            return_eigenvectors=return_eigenvectors, 
                            maxiter=max_itr, tol=tol, ncv=ncv)
        
        return res
        
    def find_min_h(self, B, dtau_init, tol=5E-2):
        dtau = dtau_init
        d = 1.0
        #dh_dtau = 0
        
        tau_min = 0
        
        A0 = self.A.copy()                
        AA0 = self.AA.copy()
        C0 = self.C.copy()
        
        try:
            l0 = self.l
            self.l = self.l.A
        except:
            l0 = self.l.copy()
            pass
        
        try:
            r0 = self.r
            self.r = self.r.A
        except:
            r0 = self.r.copy()
            pass
        
        h_min = self.h
        A_min = self.A.copy()
        
        #l_min = np.array(self.l, copy=True)
        #r_min = np.array(self.r, copy=True)
        
        itr = 0
        while itr == 0 or itr < 30 and (abs(dtau) / tau_min > tol or tau_min == 0):
            itr += 1
            for s in xrange(self.q):
                self.A[s] = A_min[s] -d * dtau * B[s]
            
            #self.l[:] = l_min
            #self.r[:] = r_min
            
            self.calc_lr(reset=True)
            self.calc_AA()
            self.calc_C()
            
            self.h = self.expect_2s(self.h_nn)
            
            #dh_dtau = d * (self.h - h_min) / dtau
            
            print (tau_min + dtau, self.h.real, tau_min)
            
            if self.h.real < h_min.real:
                #self.restore_CF()
                h_min = self.h
                A_min[:] = self.A
                #l_min[:] = self.l
                #r_min[:] = self.r
                
                dtau = min(dtau * 1.1, dtau_init * 10)
                
#                if tau + d * dtau > 0:
                tau_min += d * dtau
#                else:
#                    d = -1.0
#                    dtau = tau
            else:
                d *= -1.0
                dtau = dtau / 2.0
                
#                if tau + d * dtau < 0:
#                    dtau = tau #only happens if dtau is -ive
                
        #Must restore everything needed for take_step
        self.A = A0
        self.l = l0
        self.r = r0
        self.AA = AA0
        self.C = C0
        
        return tau_min
        
    def find_min_h_brent(self, B, dtau_init, tol=5E-2, skipIfLower=False, 
                         trybracket=True):
        #These were parameters...but it seems like cython was keeping
        #their values across calls...?!
        taus=[]
        hs=[]
        
        if len(taus) == 0:
            ls = []
            rs = []
        else:
            ls = [self.l.copy()] * len(taus)
            rs = [self.r.copy()] * len(taus)
        
        def f(tau, *args):
            if tau == 0:
                print (0, "tau=0")
                return self.h.real                
            try:
                i = taus.index(tau)
                print (tau, hs[i], hs[i] - self.h.real, "from stored")
                return hs[i]
            except ValueError:
                for s in xrange(self.q):
                    self.A[s] = A0[s] - tau * B[s]
                
                if len(taus) > 0:
                    nearest_tau_ind = abs(np.array(taus) - tau).argmin()
                    self.l = ls[nearest_tau_ind]
                    self.r = rs[nearest_tau_ind]

                self.calc_lr(auto_reset=False)
                self.calc_AA()
                self.calc_C()
                
                h = self.expect_2s(self.h_nn)
                
                print (tau, h.real, h.real - self.h.real, self.itr_l, self.itr_r)
                
                res = h.real
                
                taus.append(tau)
                hs.append(res)
                ls.append(self.l.copy())
                rs.append(self.r.copy())
                
                return res
        
        A0 = self.A.copy()                
        AA0 = self.AA.copy()
        C0 = self.C.copy()
        
        try:
            l0 = self.l
            self.l = self.l.A
        except:
            l0 = self.l.copy()
            pass
        
        try:
            r0 = self.r
            self.r = self.r.A
        except:
            r0 = self.r.copy()
            pass
        
        if skipIfLower:
            if f(dtau_init) < self.h.real:
                return dtau_init
        
        fb_brack = (dtau_init * 0.9, dtau_init * 1.1)
        if trybracket:
            brack = (dtau_init * 0.1, dtau_init, dtau_init * 2.0)
        else:
            brack = fb_brack
                
        try:
            tau_opt, h_min, itr, calls = opti.brent(f, 
                                                    brack=brack, 
                                                    tol=tol,
                                                    maxiter=20,
                                                    full_output=True)
        except ValueError:
            print "Bracketing attempt failed..."
            tau_opt, h_min, itr, calls = opti.brent(f, 
                                                    brack=fb_brack, 
                                                    tol=tol,
                                                    maxiter=20,
                                                    full_output=True)
        
        #Must restore everything needed for take_step
        self.A = A0
        self.l = l0
        self.r = r0
        self.AA = AA0
        self.C = C0
        
        #hopefully optimize next calc_lr
        nearest_tau_ind = abs(np.array(taus) - tau_opt).argmin()
        self.l_before_CF = ls[nearest_tau_ind]
        self.r_before_CF = rs[nearest_tau_ind]
        
        return tau_opt, h_min
        
    def step_reduces_h(self, B, dtau):
        A0 = self.A.copy()
        AA0 = self.AA.copy()
        C0 = self.C.copy()
        
        try:
            l0 = self.l
            self.l = self.l.A
        except:
            l0 = self.l.copy()
            pass
        
        try:
            r0 = self.r
            self.r = self.r.A
        except:
            r0 = self.r.copy()
            pass
        
        for s in xrange(self.q):
            self.A[s] = A0[s] - dtau * B[s]
        
        self.calc_lr()
        self.calc_AA()
        self.calc_C()
        
        h = self.expect_2s(self.h_nn)
        
        #Must restore everything needed for take_step
        self.A = A0
        self.l = l0
        self.r = r0
        self.AA = AA0
        self.C = C0
        
        return h.real < self.h.real, h

    def calc_B_CG(self, B_CG_0, x_0, eta_0, dtau_init, reset=False,
                 skipIfLower=False, brent=True):
        B = self.calc_B()
        eta = self.eta
        x = self.x
        
        if reset:
            beta = 0.
            print "RESET CG"
            
            B_CG = B
        else:
            beta = (eta**2) / eta_0**2
            
            #xy = m.adot(x_0, x)
            #betaPR = (eta**2 - xy) / eta_0**2
        
            print "BetaFR = " + str(beta)
            #print "BetaPR = " + str(betaPR)
        
            beta = max(0, beta.real)
        
            B_CG = B + beta * B_CG_0
        
        taus = []
        hs = []
        
        lb0 = self.l_before_CF.copy()
        rb0 = self.r_before_CF.copy()
        
        if skipIfLower:
            stepRedH, h = self.step_reduces_h(B_CG, dtau_init)
            taus.append(dtau_init)
            hs.append(h)
        
        if skipIfLower and stepRedH:
            tau = self.find_min_h(B_CG, dtau_init)
        else:
            if brent:
                tau, h_min = self.find_min_h_brent(B_CG, dtau_init,
                                                   trybracket=False)
            else:
                tau = self.find_min_h(B_CG, dtau_init)
        
#        if tau < 0:
#            print "RESET due to negative dtau!"
#            B_CG = B
#            tau, h_min = self.find_min_h_brent(B_CG, dtau_init)
            
        if self.h.real < h_min:
            print "RESET due to energy rise!"
            B_CG = B
            self.l_before_CF = lb0
            self.r_before_CF = rb0
            tau, h_min = self.find_min_h_brent(B_CG, dtau_init * 0.1, trybracket=False)
        
            if self.h.real < h_min:
                print "RESET FAILED: Setting tau=0!"
                self.l_before_CF = lb0
                self.r_before_CF = rb0
                tau = 0
        
        return B_CG, B, x, eta, tau
        
            
    def export_state(self, userdata=None):
        if userdata is None:
            userdata = self.userdata

        l = np.asarray(self.l)
        r = np.asarray(self.r)
            
        tosave = np.empty((5), dtype=np.ndarray)
        tosave[0] = self.A
        tosave[1] = l
        tosave[2] = r
        tosave[3] = self.K
        tosave[4] = np.asarray(userdata)
        
        return tosave
            
    def save_state(self, file, userdata=None):
        np.save(file, self.export_state(userdata))
        
    def import_state(self, state, expand=False, expand_q=False, shrink_q=False, refac=0.1, imfac=0.1):
        newA = state[0]
        newl = state[1]
        newr = state[2]
        newK = state[3]
        if state.shape[0] > 4:
            self.userdata = state[4]
        
        if (newA.shape == self.A.shape):
            self.A[:] = newA
            self.K[:] = newK

            self.l = np.asarray(newl)
            self.r = np.asarray(newr)
            self.l_before_CF = self.l
            self.r_before_CF = self.r
                
            return True
        elif expand and (len(newA.shape) == 3) and (newA.shape[0] == 
        self.A.shape[0]) and (newA.shape[1] == newA.shape[2]) and (newA.shape[1]
        <= self.A.shape[1]):
            newD = self.D
            savedD = newA.shape[1]
            self._init_arrays(savedD, self.q)
            self.A[:] = newA
            self.l = newl
            self.r = newr            
            self.K[:] = newK
            self.expand_D(newD, refac, imfac)
            self.l_before_CF = self.l
            self.r_before_CF = self.r
            print "EXPANDED!"
        elif expand_q and (len(newA.shape) == 3) and (newA.shape[0] <= 
        self.A.shape[0]) and (newA.shape[1] == newA.shape[2]) and (newA.shape[1]
        == self.A.shape[1]):
            newQ = self.q
            savedQ = newA.shape[0]
            self._init_arrays(self.D, savedQ)
            self.A[:] = newA
            self.l = newl
            self.r = newr
            self.K[:] = newK
            self.expand_q(newQ)
            self.l_before_CF = self.l
            self.r_before_CF = self.r
            print "EXPANDED in q!"
        elif shrink_q and (len(newA.shape) == 3) and (newA.shape[0] >= 
        self.A.shape[0]) and (newA.shape[1] == newA.shape[2]) and (newA.shape[1]
        == self.A.shape[1]):
            newQ = self.q
            savedQ = newA.shape[0]
            self._init_arrays(self.D, savedQ)
            self.A[:] = newA
            self.l = newl
            self.r = newr
            self.K[:] = newK
            self.shrink_q(newQ)
            self.l_before_CF = self.l
            self.r_before_CF = self.r
            print "SHRUNK in q!"
        else:
            return False
            
    def load_state(self, file, expand=False, expand_q=False, shrink_q=False, refac=0.1, imfac=0.1):
        state = np.load(file)
        return self.import_state(state, expand=expand, expand_q=expand_q, shrink_q=shrink_q, refac=refac, imfac=imfac)
            
    def expand_q(self, newq):
        oldK = self.K
        
        super(EvoMPS_TDVP_Uniform, self).expand_q(newq)
        #self._init_arrays(self.D, newq) 
        
        self.K = oldK
        
    def shrink_q(self, newq):
        oldK = self.K
                
        super(EvoMPS_TDVP_Uniform, self).shrink_q(newq)
        #self._init_arrays(self.D, newq) 
        
        self.K = oldK
                    
    def expand_D(self, newD, refac=100, imfac=0):
        oldK = self.K
        oldD = self.D
                
        super(EvoMPS_TDVP_Uniform, self).expand_D(newD, refac=refac, imfac=imfac)
        #self._init_arrays(newD, self.q)
                
        self.K[:oldD, :oldD] = oldK
        self.K[oldD:, :oldD].fill(la.norm(oldK) / oldD**2)
        self.K[:oldD, oldD:].fill(la.norm(oldK) / oldD**2)
        self.K[oldD:, oldD:].fill(la.norm(oldK) / oldD**2)
        