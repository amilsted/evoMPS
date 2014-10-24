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

def zexpmvh(A, v, t, norm_est=1., m=5):
    xn = A.shape[0]
    vf = sp.ones((xn,), dtype=A.dtype)
    m = min(xn - 1, m)
    nwsp = max(10, xn * (m + 2) + 5 * (m + 2)**2 + 6 + 1)
    wsp = sp.zeros((nwsp,), dtype=A.dtype)
    niwsp = max(7, m + 2)
    iwsp = sp.zeros((niwsp,), dtype=sp.int64)
    iflag = sp.zeros((1,), dtype=sp.int64)
    itrace = sp.array([0])
    expokit.zhexpv(m, [t], v, vf, [0.], [norm_est], 
                   wsp, iwsp, A.matvec, itrace, iflag, n=[xn], 
                   lwsp=[len(wsp)], liwsp=[len(iwsp)])
    if iflag[0] == 1:
        print "Max steps reached!"
    elif iflag[0] == 2:
        print "Tolerance too high!"
    return vf
    
def zexpmv(A, v, t, norm_est=1., m=5):
    xn = A.shape[0]
    vf = sp.ones((xn,), dtype=A.dtype)
    m = min(xn - 1, m)
    nwsp = max(10, xn * (m + 2) + 5 * (m + 2)**2 + 6 + 1)
    wsp = sp.zeros((nwsp,), dtype=A.dtype)
    niwsp = max(7, m + 2)
    iwsp = sp.zeros((niwsp,), dtype=sp.int64)
    iflag = sp.zeros((1,), dtype=sp.int64)
    itrace = sp.array([0])
    expokit.zgexpv(m, [t], v, vf, [0.], [norm_est], 
                   wsp, iwsp, A.matvec, itrace, iflag, n=[xn], 
                   lwsp=[len(wsp)], liwsp=[len(iwsp)])
    if iflag[0] == 1:
        print "Max steps reached!"
    elif iflag[0] == 2:
        print "Tolerance too high!"
    return vf