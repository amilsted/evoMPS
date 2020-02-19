# -*- coding: utf-8 -*-
"""Split-step methods for nonuniform MPS.
"""
import copy as cp
import scipy as sp
import scipy.linalg as la
import scipy.sparse.linalg as las
from . import matmul as m
from . import tdvp_common as tm
import logging


log = logging.getLogger(__name__)


class Vari_Opt_Single_Site_Op:
    def __init__(self, mps, n, AA, ham, ham_sites, KLnm1, KRnp1, tau=1.,
                 HML=None, HMR=None, HMn=None,
                 use_local_ham=True,
                 sanity_checks=False):
        """
        """
        self.D = mps.D
        self.q = mps.q
        self.mps = mps
        self.n = n
        self.AA = AA
        self.ham = ham
        self.ham_sites = ham_sites
        self.KLnm1 = KLnm1
        self.KRnp1 = KRnp1
        
        self.HML = HML
        self.HMR = HMR
        self.HMn = HMn
        
        self.sanity_checks = sanity_checks
        self.sanity_tol = 1E-12
        
        d = self.D[n - 1] * self.D[n] * self.q[n]
        self.shape = (d, d)
        
        self.dtype = sp.dtype(mps.typ)
        
        self.calls = 0
        
        self.tau = tau
        
        self.ham_local = use_local_ham
        
        self.ham_MPO = not HMn is None
        
    def apply_ham_local(self, An, res):
        m = self.mps
        n = self.n
        
        #Assuming RCF        
        if self.ham_sites == 2:
            Anm1 = m.get_A(n - 1)
            if Anm1 is not None:
                AAnm1 = tm.calc_AA(Anm1, An)
                Cnm1 = tm.calc_C_mat_op_AA(self.ham[n - 1], AAnm1)
                Cnm1 = sp.transpose(Cnm1, axes=(1, 0, 2, 3)).copy()
                for s in range(m.q[n]):
                    res[s] += tm.eps_l_noop(m.get_l(n - 2), Anm1, Cnm1[s, :])
            Anp1 = m.get_A(n + 1)
            if Anp1 is not None:
                AAn = tm.calc_AA(An, Anp1)
                Cn = tm.calc_C_mat_op_AA(self.ham[n], AAn)
                for s in range(m.q[n]):
                    res[s] += tm.eps_r_noop(m.get_r(n + 1), Cn[s, :], Anp1)
                
        if self.ham_sites == 3:
            Anm2 = m.get_A(n - 2)
            if Anm2 is not None:
                AAAnm2 = tm.calc_AAA_AAr(Anm2, self.AA[n - 1])
                if n > 1:
                    AAnm2 = self.AA[n - 2]
                else:
                    AAnm2 = tm.calc_AA(Anm2, m.get_A(n - 1))
                Cnm2 = tm.calc_C_3s_mat_op_AAA(self.ham[n - 2], AAAnm2)
                Cnm2 = sp.transpose(Cnm2, axes=(2, 0, 1, 3, 4)).copy()
                for s in range(m.q[n]):
                    res[s] += tm.eps_l_op_2s_AA12_C34(m.get_l(n - 3), AAnm2, Cnm2[s, :, :])
                    
            if n > 1 and n < m.N:
                AAnm1 = tm.calc_AA(m.get_A(n - 1), An)  
                AAAnm1 = tm.calc_AAA_AA(AAnm1, m.get_A(n + 1))
                Cnm1 = tm.calc_C_3s_mat_op_AAA(self.ham[n - 1], AAAnm1)
                for s in range(m.q[n]):
                    for u in range(m.q[n - 1]):
                        res[s] += m.get_A(n - 1)[u].conj().m.dot(m.get_l(n - 2).dot(
                                  tm.eps_r_noop(m.get_r(n + 1), Cnm1[u, s, :], m.get_A(n + 1))))
                                  
            if n < m.N - 1:
                AAn = tm.calc_AA(An, m.get_A(n + 1))
                AAAn = tm.calc_AAA_AA(AAn, m.get_A(n + 2))
                Cn = tm.calc_C_3s_mat_op_AAA(self.ham[n], AAAn)
                for s in range(m.q[n]):
                    res[s] += tm.eps_r_op_2s_AA12_C34(m.get_r(n + 2), Cn[s, :, :], self.AA[n + 1])
            
        if self.KLnm1 is not None:
            for s in range(m.q[n]):
                res[s] += self.KLnm1.dot(An[s])
                
        if self.KRnp1 is not None:
            for s in range(m.q[n]):
                res[s] += An[s].dot(self.KRnp1)
                
    def apply_ham_MPO(self, An, res):
        n = self.n
        
        HMAn = tm.apply_MPO_local(self.HMn, An)
        #print self.HML.shape, HMAn[0].shape, self.HMR.shape, An[0].shape
        for s in range(self.q[n]):
            res[s] += self.HML.conj().T.dot(HMAn[s]).dot(self.HMR)
        
    def matvec(self, x):
        self.calls += 1
        #print self.calls
        
        n = self.n
        
        #x = sp.asarray(x, dtype=self.dtype) #ensure the type is right!
        An = x.reshape((self.q[n], self.D[n - 1], self.D[n]))
        
        res = sp.zeros_like(An)
        
        if self.ham_local:
            self.apply_ham_local(An, res)
        
        if self.ham_MPO:
            self.apply_ham_MPO(An, res)
                
        #print "en = ", (sp.inner(An.conj().ravel(), res.ravel())
        #                / sp.inner(An.conj().ravel(), An.ravel()))
        
        return res.reshape(x.shape) * self.tau

class Vari_Opt_Two_Site_Op:
    def __init__(self, mps, n, ham, ham_sites, KLnm1, KRnp2, tau=1., HML=None, HMR=None, HMn=None,
                 use_local_ham=True,
                 sanity_checks=False):
        """
        """
        self.D = mps.D
        self.q = mps.q
        self.mps = mps
        self.n = n
        self.ham = ham
        self.ham_sites = ham_sites
        self.KLnm1 = KLnm1
        self.KRnp2 = KRnp2
        
        self.HML = HML
        self.HMR = HMR
        self.HMn = HMn
        
        self.sanity_checks = sanity_checks
        self.sanity_tol = 1E-12
        
        d = self.D[n - 1] * self.D[n + 1] * self.q[n] * self.q[n + 1]
        self.shape = (d, d)
        
        self.dtype = sp.dtype(mps.typ)
        
        self.calls = 0
        
        self.tau = tau
        
        self.ham_local = use_local_ham
        
        self.ham_MPO = not HMn is None
        
    def apply_ham_local(self, AAn, res):
        m = self.mps
        n = self.n        
               
        if self.ham_sites == 2:
            # Assuming r[n+1] = eye 
            Anm1 = m.get_A(n - 1)
            if Anm1 is not None:
                # nn-term overlapping with the left site of the 2-site tensor

                # first contract the ham term with the single-site tensor on
                # the left
                tmp = sp.zeros([m.q[n], m.q[n], m.D[n - 1], m.D[n - 1]], dtype=res.dtype)
                for s_in in range(m.q[n]):
                    for s_out in range(m.q[n]):
                        tmp[s_out,s_in] += tm.eps_l_op_1s(
                            m.get_l(n - 2),
                            Anm1,
                            Anm1,
                            self.ham[n - 1][:,s_out,:,s_in]).T.conj()
                
                # now contract the left part with the two-site tensor
                res_sub = sp.tensordot(tmp, AAn, axes=((1,3),(0,2))) #[s_out_tmp,D1_tmp,s2_AAn,D2_AAn]
                res += sp.transpose(res_sub, (0,2,1,3)) #[s_out_tmp,s2_AAn,D1_tmp,D2_AAn]

            # nn-term applied to 2-site tensor
            Cn = tm.calc_C_mat_op_AA(self.ham[n], AAn)
            # Assumes central gauging, i.e. r[n+1] = l[n-2] = eye
            res += Cn

            # Assuming l_(n-2) = eye 
            Anp2 = m.get_A(n + 2)
            if Anp2 is not None:
                # nn-term overlapping with the right site of the 2-site tensor

                # first contract the ham term with the single-site tensor on
                # the right
                tmp = sp.zeros([m.q[n+1], m.q[n+1], m.D[n+1], m.D[n+1]], dtype=res.dtype)
                for s_in in range(m.q[n+1]):
                    for s_out in range(m.q[n+1]):
                        tmp[s_out,s_in] += tm.eps_r_op_1s(
                            m.get_r(n + 2),
                            Anp2,
                            Anp2,
                            self.ham[n + 1][s_out,:,s_in,:])

                # now contract the right part with the two-site tensor
                res_sub = sp.tensordot(AAn, tmp, axes=((1,3),(1,2))) #[s1_AAn,D1_AAn,s_out_tmp,D2_tmp]
                res += sp.transpose(res_sub, (0,2,1,3)) #[s1_AAn,s_out_tmp,D1_AAn,D2_tmp]

        else:
            raise NotImplementedError()
            
        if self.KLnm1 is not None: #Assuming r[n+1] = eye 
            # left disconnected ham contributions
            for s1 in range(m.q[n]):
                for s2 in range(m.q[n+1]):
                    res[s1, s2] += self.KLnm1.dot(AAn[s1, s2])
                
        if self.KRnp2 is not None: #Assuming l_(n-2) = eye 
            # right disconnected ham contributions
            for s1 in range(m.q[n]):
                for s2 in range(m.q[n+1]):
                    res[s1, s2] += AAn[s1,s2].dot(self.KRnp2)
        
    def matvec(self, x):
        self.calls += 1
        #print self.calls
        
        n = self.n
        
        AAn = x.reshape((self.q[n], self.q[n+1], self.D[n-1], self.D[n+1]))
        
        res = sp.zeros_like(AAn)
        
        if self.ham_local:
            self.apply_ham_local(AAn, res)
        
        if self.ham_MPO:
            raise NotImplementedError()
                
        #print "en = ", (sp.inner(An.conj().ravel(), res.ravel())
        #                / sp.inner(An.conj().ravel(), An.ravel()))
        
        return res.reshape(x.shape) * self.tau
        
class Vari_Opt_SC_op:
    def __init__(self, mps, n, AA, ham, ham_sites, KLn, KRnp1, tau=1,
                 HML=None, HMR=None,
                 use_local_ham=True, sanity_checks=False):
        """
        """
        self.D = mps.D
        self.q = mps.q
        self.mps = mps
        self.n = n
        self.AA = AA
        self.ham = ham
        self.ham_sites = ham_sites
        self.KLn = KLn
        self.KRnp1 = KRnp1
        
        self.HML = HML
        self.HMR = HMR
        
        self.sanity_checks = sanity_checks
        self.sanity_tol = 1E-12
        
        d = self.D[n] * self.D[n]
        self.shape = (d, d)
        
        self.dtype = sp.dtype(mps.typ)
        
        self.calls = 0
        
        self.tau = tau
        
        self.ham_local = use_local_ham
        
        self.ham_MPO = not HML is None
        
    def apply_ham_MPO(self, Gn, res):
        HMGn = sp.kron(Gn, sp.eye(self.HMR.shape[0] // self.D[self.n]))
        res += self.HML.conj().T.dot(HMGn).dot(self.HMR)
        
    def apply_ham_local(self, Gn, res):
        m = self.mps
        n = self.n

        if self.KLn is not None:
            res += self.KLn.dot(Gn)
        
        if self.KRnp1 is not None:
            res += Gn.dot(self.KRnp1)
        
        Ap1 = sp.array([Gn.dot(As) for As in m.A[n + 1]])
        
        if self.ham_sites == 2:
            AAn = tm.calc_AA(m.A[n], Ap1)
            Cn = tm.calc_C_mat_op_AA(self.ham[n], AAn)
            for s in range(m.q[n]):
                sres = tm.eps_r_noop(m.r[n + 1], Cn[s, :], m.A[n + 1])
                res += m.A[n][s].conj().T.dot(m.l[n - 1].dot(sres))
        elif self.ham_sites == 3:
            Anp2 = m.get_A(n + 2)
            if Anp2 is not None:
                AAn = tm.calc_AA(m.A[n], Ap1)
                AAAn = tm.calc_AAA_AA(AAn, Anp2)
                Cn = tm.calc_C_3s_mat_op_AAA(self.ham[n], AAAn)
                for s in range(m.q[n]):
                    res += m.A[n][s].conj().T.dot(
                             tm.eps_r_op_2s_AA12_C34(m.get_r(n + 2), Cn[s, :, :], self.AA[n + 1]))
            lnm2 = m.get_l(n - 2)
            if lnm2 is not None:
                AAAm1 = tm.calc_AAA_AA(self.AA[n - 1], Ap1)
                Cm1 = tm.calc_C_3s_mat_op_AAA(self.ham[n - 1], AAAm1)
                Cm1 = sp.transpose(Cm1, axes=(2, 0, 1, 3, 4)).copy()
                for s in range(m.q[n + 1]):
                    res += tm.eps_l_op_2s_AA12_C34(
                        lnm2, self.AA[n - 1], Cm1[s, :, :]
                      ).dot(m.A[n + 1][s].conj().T)
            
    def matvec(self, x):
        self.calls += 1
        #print self.calls

        n = self.n
        
        Gn = x.reshape((self.D[n], self.D[n]))

        res = sp.zeros_like(Gn)
        
        if self.ham_local:
            self.apply_ham_local(Gn, res)
            
        if self.ham_MPO:
            self.apply_ham_MPO(Gn, res)
        
        return res.reshape(x.shape) * self.tau


def evolve_split(mps, ham, ham_sites, dtau, num_steps, ham_is_Herm=True, HMPO=None, 
                    use_local_ham=True, ncv=20, tol=1E-14, expm_max_steps=10,
                    DMRG=False,
                    print_progress=True, norm_est=1.0,
                    two_site=False, D_max=None, min_schmidt=None,
                    KL_bulk=None, KR_bulk=None,
                    cb_func=None):
    """Take a time-step dtau using the split-step integrator.
    
    This is the one-site version of a DMRG-like time integrator described
    at:
      http://arxiv.org/abs/1408.5056
    
    It has a fourth-order local error and is symmetric. It requires
    iteratively computing two matrix exponentials per site, and thus
    has less predictable CPU time requirements than the Euler or RK4 
    methods.
    
    Parameters
    ----------
    dtau : complex
        The (imaginary or real) amount of imaginary time (tau) to step.
    ham_is_Herm : bool
        Whether the Hamiltonian is really Hermitian. If so, the lanczos
        method will be used for imaginary time evolution.
    """
    #mps.eta_sq.fill(0)
    #mps.eta = 0
    
    if not DMRG:
        dtau *= -1
        from .sexpmv import gexpmv

        if sp.iscomplex(dtau):
            op_is_herm = False
            fac = 1.j
            dtau = sp.imag(dtau)
        else:
            if ham_is_Herm:
                op_is_herm = True
            else:
                op_is_herm = False
            fac = 1
    
    #assert mps.canonical_form == 'right', 'take_step_split only implemented for right canonical form'
    # FIXME: Check form manually
    assert ham_sites == 2 or ham_sites == 3
    import sys
            
    def evolve_A(n, step, norm_est, calc_norm_est=False):
        lop = Vari_Opt_Single_Site_Op(mps, n, AA, ham, ham_sites, 
                                      KL[n - 1], KR[n + 1], tau=fac, 
                                      HML=HML[n - 1], HMR=HMR[n], HMn=HM[n],
                                      use_local_ham=use_local_ham,
                                      sanity_checks=mps.sanity_checks)
        An_old = mps.A[n].ravel()

        if calc_norm_est: #simple attempt at approximating the norm
            nres = lop.matvec(sp.asarray(sp.randn(len(An_old)), dtype=An_old.dtype))
            norm_est = max(norm_est, la.norm(nres, ord=sp.inf))
            #print("norm_est=", norm_est)
        
        #An = zexpmv(lop, An_old, dtau/2., norm_est=norm_est, m=ncv, tol=tol,
        #            A_is_Herm=op_is_herm)
        #FIXME: Currently we don't take advantage of Hermiticity.
        ncv_An = min(ncv, len(An_old)-1)
        An, conv, nstep, brkdown, mb, err = gexpmv(
            lop, An_old, step, norm_est, m=ncv_An, tol=tol, mxstep=expm_max_steps)
        expm_info = {
            'converged': conv,
            'max_error': err[0],
            'summed_error': err[1],
            'num_krylov': mb,
            'num_steps': nstep}
        if not conv:
            log.warn("Krylov exp(M)*v solver for An did not converge in %u steps for site %u.", nstep, n)
        mps.A[n] = An.reshape((mps.q[n], mps.D[n - 1], mps.D[n]))
        mps.A[n] /= sp.sqrt(m.adot(mps.A[n], mps.A[n]))
        return norm_est, expm_info

    def evolve_AA(n, step, norm_est, calc_norm_est=False, moving_right=True):
        lop = Vari_Opt_Two_Site_Op(mps, n, ham, ham_sites, KL[n - 1], KR[n + 2], tau=fac, 
                                      HML=HML[n - 1], HMR=HMR[n], HMn=HM[n],
                                      use_local_ham=use_local_ham,
                                      sanity_checks=mps.sanity_checks)
        AAn_old = tm.calc_AA(mps.A[n], mps.A[n+1]).ravel()

        if calc_norm_est: #simple attempt at approximating the norm
            nres = lop.matvec(sp.asarray(sp.randn(len(AAn_old)), dtype=AAn_old.dtype))
            norm_est = max(norm_est, la.norm(nres, ord=sp.inf))
            #print("norm_est=", norm_est)

        ncv_AAn = min(ncv, len(AAn_old)-1)
        AAn, conv, nstep, brkdown, mb, err = gexpmv(
            lop, AAn_old, step, norm_est, m=ncv_AAn, tol=tol, mxstep=expm_max_steps)
        expm_info = {
            'converged': conv,
            'max_error': err[0],
            'summed_error': err[1],
            'num_krylov': mb,
            'num_steps': nstep}
        if not conv:
            log.warn("Krylov exp(M)*v solver for AAn did not converge in %u steps for site %u.", nstep, n)
        AAn = AAn.reshape([mps.q[n], mps.q[n+1], mps.D[n-1], mps.D[n+1]])
        An, G, Anp1, s_rest = split_twosite(AAn, D_max, min_schmidt)
        trunc_err = la.norm(s_rest) if len(s_rest) > 0 else 0.0
        G /= sp.sqrt(m.adot(G, G))
        D_new = G.shape[0]
        if moving_right:
            for s in range(mps.q[n+1]):
                Anp1[s] = G.dot(Anp1[s])
        else:
            for s in range(mps.q[n]):
                An[s] = An[s].dot(G)
        mps.D[n] = D_new
        mps.A[n] = An
        mps.A[n+1] = Anp1
        return norm_est, expm_info, trunc_err
        
    def evolve_G(n, step, G, norm_est):
        lop2 = Vari_Opt_SC_op(mps, n, AA, ham, ham_sites, KL[n], KR[n + 1], tau=fac,
                              HML=HML[n], HMR=HMR[n],
                              use_local_ham=use_local_ham,
                              sanity_checks=mps.sanity_checks)
        Gold = G.ravel()
        #G = zexpmv(lop2, Gold, -dtau/2., norm_est=norm_est, m=ncv, tol=tol,
        #           A_is_Herm=op_is_herm)
        ncv_G = min(ncv, len(Gold)-1)
        G, conv, nstep, brkdown, mb, err = gexpmv(
            lop2, Gold, step, norm_est, m=ncv_G, tol=tol, mxstep=expm_max_steps)
        expm_info = {
            'converged': conv,
            'max_error': err[0],
            'summed_error': err[1],
            'num_krylov': mb,
            'num_steps': nstep}
        if not conv:
            log.warn("Krylov exp(M)*v solver for G did not converge in %u steps for site %u.", nstep, n)
        G = G.reshape((mps.D[n], mps.D[n]))
        G /= sp.sqrt(m.adot(G, G))
        return G, expm_info
        
    def opt_A(n):
        lop = Vari_Opt_Single_Site_Op(mps, n, AA, ham, ham_sites, KL[n - 1], KR[n + 1],
                                      HML=HML[n - 1], HMR=HMR[n], HMn=HM[n],
                                      use_local_ham=use_local_ham,
                                      sanity_checks=mps.sanity_checks)
        if ham_is_Herm:
            evs, eVs = las.eigsh(lop, k=1, which='SA', sigma=None, 
                              v0=mps.A[n].ravel(), ncv=ncv, tol=tol)
        else:
            evs, eVs = las.eigs(lop, k=1, which='SA', sigma=None, 
                              v0=mps.A[n].ravel(), ncv=ncv, tol=tol)
        
        mps.A[n] = eVs[:, 0].reshape((mps.q[n], mps.D[n - 1], mps.D[n]))
        norm = m.adot(mps.A[n], mps.A[n])
        mps.A[n] /= sp.sqrt(norm)

    def opt_AA(n, moving_right=True):
        lop = Vari_Opt_Two_Site_Op(mps, n, ham, ham_sites, KL[n - 1], KR[n + 2],
                                      HML=HML[n - 1], HMR=HMR[n], HMn=HM[n],
                                      use_local_ham=use_local_ham,
                                      sanity_checks=mps.sanity_checks)
        AAn_old = tm.calc_AA(mps.A[n], mps.A[n+1]).ravel()

        if ham_is_Herm:
            evs, eVs = las.eigsh(lop, k=1, which='SA', sigma=None, 
                                v0=AAn_old.ravel(), ncv=ncv, tol=tol)
        else:
            evs, eVs = las.eigs(lop, k=1, which='SA', sigma=None, 
                                v0=AAn_old.ravel(), ncv=ncv, tol=tol)

        AAn = eVs[:, 0].reshape((mps.q[n], mps.q[n+1], mps.D[n - 1], mps.D[n + 1]))
        An, G, Anp1, s_rest = split_twosite(AAn, D_max, min_schmidt)
        trunc_err = la.norm(s_rest) if len(s_rest) > 0 else 0.0
        G /= sp.sqrt(m.adot(G, G))
        D_new = G.shape[0]
        if moving_right:
            for s in range(mps.q[n+1]):
                Anp1[s] = G.dot(Anp1[s])
        else:
            for s in range(mps.q[n]):
                An[s] = An[s].dot(G)
        mps.D[n] = D_new
        mps.A[n] = An
        mps.A[n+1] = Anp1
        return trunc_err

    def split_twosite(AA, D_max, min_schmidt):
        q1, q2, D0, D2 = AA.shape
        AA_mat = sp.reshape(
            sp.transpose(AA, [0,2,1,3]),
            [q1 * D0, q2 * D2])
        u, s, vh = la.svd(AA_mat)
        D1 = min(D_max, len(s))
        for i in range(D1):
            if s[i] < min_schmidt:
                D1 = i
                break
        s_rest = s[D1:]
        u = u[:, :D1]
        s = s[:D1]
        vh = vh[:D1, :]
        A1 = sp.reshape(u, [q1, D0, D1])
        A2 = sp.transpose(sp.reshape(vh, [D1, q2, D2]), [1,0,2])
        G = sp.diag(s)
        return A1, G, A2, s_rest
    
    def update_Heff_left(n):
        Anm1 = mps.get_A(n - 1)
        if ham_sites == 2 and Anm1 is not None:
            AA[n - 1] = tm.calc_AA(Anm1, mps.A[n])
            Cnm1 = tm.calc_C_mat_op_AA(ham[n - 1], AA[n - 1])
            KLnm1 = KL[n - 1]
            if KLnm1 is None:
                KLnm1 = sp.zeros((mps.D[n-1], mps.D[n-1]), dtype=mps.typ)
            KL[n], _ = tm.calc_K_l(KLnm1, Cnm1, mps.get_l(n - 2),
                                    None, mps.A[n], AA[n - 1])
        elif ham_sites == 3:
            if Anm1 is not None:
                #for next step and bond
                AA[n - 1] = tm.calc_AA(Anm1, mps.A[n])
            Anm2 = mps.get_A(n - 2)
            if Anm2 is not None:
                AAAnm2 = tm.calc_AAA_AAr(Anm2, AA[n - 1])
                Cnm2 = tm.calc_C_3s_mat_op_AAA(ham[n - 2], AAAnm2)
                KL[n], _ = tm.calc_K_3s_l(KL[n - 1], Cnm2, mps.l[n - 3], 
                                            None, mps.A[n], AAAnm2)
                                    
        if not HMPO is None:
            HMA[n] = tm.apply_MPO_local(HM[n], mps.A[n])
            HML[n] = mps.calc_MPO_l(HMA[n], n, HML[n - 1])
            
    def update_Heff_right(n):
        Anp1 = mps.get_A(n + 1)
        if ham_sites == 2 and Anp1 is not None:
            AA[n] = tm.calc_AA(mps.A[n], Anp1)                    
            Cn = tm.calc_C_mat_op_AA(ham[n], AA[n])
            KRnp1 = KR[n + 1]
            if KRnp1 is None:
                KRnp1 = sp.zeros((mps.D[n], mps.D[n]), dtype=mps.typ)
            KR[n], _ = tm.calc_K(KRnp1, Cn, None, 
                mps.get_r(n + 1), mps.A[n], AA[n])
        
        if ham_sites == 3:
            if Anp1 is not None:
                AA[n] = tm.calc_AA(mps.A[n], Anp1)
            Anp2 = mps.get_A(n + 2)
            if Anp2 is not None:
                AAAn = tm.calc_AAA_AA(AA[n], Anp2)
                Cn = tm.calc_C_3s_mat_op_AAA(ham[n], AAAn)
                KR[n], _ = tm.calc_K_3s(KR[n + 1], Cn, None, 
                    mps.get_r(n + 2), mps.A[n], AAAn)
            
        if not HMPO is None:
            HMA[n] = tm.apply_MPO_local(HM[n], mps.A[n])
            HMR[n - 1] = mps.calc_MPO_rm1(HMA[n], n, HMR[n])

    def right_move_1site(n, norm_est):
        if DMRG:
            opt_A(n)
            expm_info_A = None
        else:
            norm_est, expm_info_A = evolve_A(n, dtau/2., norm_est, calc_norm_est=True)

        expm_info_G = None
        if n < mps.N:
            #shift centre matrix right (RCF is like having a centre "matrix" at "1")
            G = tm.restore_LCF_l_seq(mps.A[n - 1:n + 1], mps.l[n - 1:n + 1],
                                    sanity_checks=mps.sanity_checks) 

            update_Heff_left(n)
            
            if not DMRG:
                G, expm_info_G = evolve_G(n, -dtau/2., G, norm_est)
            
            for s in range(mps.q[n + 1]):
                mps.A[n + 1][s] = G.dot(mps.A[n + 1][s])

        return norm_est, expm_info_A, expm_info_G

    def right_move_2site(n, norm_est):
        if n == mps.N:
            return norm_est, None, None, 0.0

        if DMRG:
            terr = opt_AA(n, moving_right=n < mps.N - 1)
            expm_info_AA = None
        else:
            norm_est, expm_info_AA, terr = evolve_AA(
                n, dtau/2., norm_est, calc_norm_est=True, moving_right=n < mps.N - 1)
        # centre matrix has now moved one site right

        mps.l[n] = m.eyemat(mps.D[n], dtype=mps.A[n].dtype)

        expm_info_A = None
        if n < mps.N - 1:
            update_Heff_left(n)

            if not DMRG:
                _, expm_info_A = evolve_A(n + 1, -dtau/2., norm_est)

        return norm_est, expm_info_AA, expm_info_A, terr

    def left_move_1site(n, norm_est):
        if DMRG:
            opt_A(n)
            expm_info_A = None
        else:
            _, expm_info_A = evolve_A(n, dtau/2., norm_est)
        
        if n > 1:
            #shift centre matrix left (LCF is like having a centre "matrix" at "N")
            Gi = tm.restore_RCF_r_seq(mps.A[n - 1:n + 1], mps.r[n - 1:n + 1],
                                    sanity_checks=mps.sanity_checks)
                                    
        update_Heff_right(n)

        expm_info_G = None
        if n > 1:
            if not DMRG:
                Gi, expm_info_G = evolve_G(n - 1, -dtau/2., Gi, norm_est)

            for s in range(mps.q[n - 1]):
                mps.A[n - 1][s] = mps.A[n - 1][s].dot(Gi)

        return norm_est, expm_info_A, expm_info_G

    def left_move_2site(n, norm_est):
        if n == mps.N:
            return norm_est, None, None, 0.0

        if DMRG:
            terr = opt_AA(n, moving_right=False)
            expm_info_AA = None
        else:
            norm_est, expm_info_AA, terr = evolve_AA(
                n, dtau/2., norm_est, moving_right=False)
        # centre matrix has now moved one site left

        mps.r[n] = m.eyemat(mps.D[n], dtype=mps.A[n].dtype)

        update_Heff_right(n+1)
        if n == 1:
            update_Heff_right(1)  # so we have the energy expectation value

        if not DMRG and n > 1:
            _, expm_info_A = evolve_A(n, -dtau/2., norm_est)
        else:
            expm_info_A = None

        return norm_est, expm_info_AA, expm_info_A, terr

    AA = [None] * (mps.N + 1)

    KL = [None] * (mps.N + 1)
    KL[0] = KL_bulk

    KR = [None] * (mps.N + 2)
    KR[mps.N + 1] = KR_bulk
    for n in reversed(range(1, mps.N + 1)):
        update_Heff_right(n)

    HMA = [None] * (mps.N + 1)
    HML = [sp.eye(1, dtype=mps.typ)] + [None] * mps.N
    HMR = [None] * mps.N + [sp.eye(1, dtype=mps.typ)]
    if HMPO is None:
        HM = [None] * (mps.N + 1)
    else:
        HM = HMPO
        for n in range(mps.N, 0, -1):
            HMA[n] = tm.apply_MPO_local(HM[n], mps.A[n])
            HMR[n - 1] = mps.calc_MPO_rm1(HMA[n], n, HMR[n])

    for i in range(num_steps):
        iexpm_lr = [None] * (2 * mps.N + 1)
        terr_lr = sp.zeros(mps.N+1)
        for n in range(1, mps.N + 1):
            if print_progress:
                print('{0}\r'.format("Sweep LR:" + str(n) + '        '), end=' ')
            sys.stdout.flush()
            if two_site:
                norm_est, iexpm_lr[2*n-1], iexpm_lr[2*n], terr_lr[n] = right_move_2site(n, norm_est)
            else:
                norm_est, iexpm_lr[2*n-1], iexpm_lr[2*n] = right_move_1site(n, norm_est)
        if print_progress:
            print()
        
        iexpm_rl = [None] * (2 * mps.N + 1)
        terr_rl = sp.zeros(mps.N+1)
        for n in range(mps.N, 0, -1):
            if print_progress:
                print('{0}\r'.format("Sweep RL:" + str(n) + '        '), end=' ')
            sys.stdout.flush()
            if two_site:
                norm_est, iexpm_rl[2*n-1], iexpm_rl[2*n], terr_rl[n] = left_move_2site(n, norm_est)
            else:
                norm_est, iexpm_rl[2*n-1], iexpm_rl[2*n] = left_move_1site(n, norm_est)
        if print_progress:
            print()
    
        if cb_func is not None:
            cb_func(mps, i,
                expm_info_lr=iexpm_lr,
                expm_info_rl=iexpm_rl,
                truncerr_lr=terr_lr,
                truncerr_rl=terr_rl)

    return mps