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

def zexpmvh(A, v, t, norm_est=1., m=20):
    xn = A.shape[0]
    vf = sp.ones((xn,), dtype=A.dtype)
    m = min(xn - 1, m)
    wsp = sp.zeros((xn * (m + 2) + 5 * (m + 2)**2 + 6 + 1,), dtype=A.dtype)
    iwsp = sp.zeros((m + 2,), dtype=int)
    iflag = sp.zeros((1,), dtype=int)
    itrace = sp.array([0])
    expokit.zhexpv(m, [t], v, vf, [0.], [norm_est], 
                   wsp, iwsp, A.matvec, itrace, iflag, n=[xn], 
                   lwsp=[len(wsp)], liwsp=[len(iwsp)])
    if iflag[0] == 1:
        print "Max steps reached!"
    elif iflag[0] == 2:
        print "Tolerance too high!"
    return vf
    
def zexpmv(A, v, t, norm_est=1., m=20):
    xn = A.shape[0]
    vf = sp.ones((xn,), dtype=A.dtype)
    m = min(xn - 1, m)
    wsp = sp.zeros((xn * (m + 2) + 5 * (m + 2)**2 + 6 + 1,), dtype=A.dtype)
    iwsp = sp.zeros((m + 2,), dtype=int)
    iflag = sp.zeros((1,), dtype=int)
    itrace = sp.array([0])
    expokit.zgexpv(m, [t], v, vf, [0.], [norm_est], 
                   wsp, iwsp, A.matvec, itrace, iflag, n=[xn], 
                   lwsp=[len(wsp)], liwsp=[len(iwsp)])
    if iflag[0] == 1:
        print "Max steps reached!"
    elif iflag[0] == 2:
        print "Tolerance too high!"
    return vf