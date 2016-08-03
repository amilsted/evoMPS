# -*- coding: utf-8 -*-
"""
Created on Tue Oct 21 15:56:10 2014

@author: ash
"""

import scipy as sp
try:
    import expokit
except ImportError, e:
    print "ERROR: expokit not available! The extension module may not have been compiled."
    raise e
    
ideg = 6

def zexpmv(A, v, t, norm_est=1., m=5, tol=0., trace=False, A_is_Herm=False):
    assert A.dtype.type is sp.complex128
    assert v.dtype.type is sp.complex128
    
    #Override expokit deault precision to match scipy sparse eigs more closely.
    if tol == 0:
        tol = sp.finfo(sp.complex128).eps * 2 #cannot take eps, as expokit changes this to sqrt(eps)!
    
    xn = A.shape[0]
    vf = sp.ones((xn,), dtype=A.dtype)
    
    m = min(xn - 1, m)
    
    nwsp = max(10, xn * (m + 2) + 5 * (m + 2)**2 + ideg + 1)
    wsp = sp.zeros((nwsp,), dtype=A.dtype)
    
    niwsp = max(7, m + 2)
    iwsp = sp.zeros((niwsp,), dtype=sp.int32)
    
    iflag = sp.zeros((1,), dtype=sp.int32)
    itrace = sp.array([int(trace)])
    
    if A_is_Herm:
        expokit.zhexpv(m, [t], v, vf, [tol], [norm_est], 
                       wsp, iwsp, A.matvec, itrace, iflag, n=[xn], 
                       lwsp=[len(wsp)], liwsp=[len(iwsp)])
    else:
        expokit.zgexpv(m, [t], v, vf, [tol], [norm_est], 
                       wsp, iwsp, A.matvec, itrace, iflag, n=[xn], 
                       lwsp=[len(wsp)], liwsp=[len(iwsp)])

    if iflag[0] == 1:
        print "Max steps reached!"
    elif iflag[0] == 2:
        print "Tolerance too high!"
    elif iflag[0] < 0:
        print "Bad arguments!"
    elif iflag[0] > 0:
        print "Unknown error!"
        
    return vf
    
