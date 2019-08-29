# -*- coding: utf-8 -*-
"""
This is a Python 3 port of FORTRAN code from the EXPOKIT package.

gexpmv is a port of the _GEXPV routines.

See R.B. Sidje, ACM Trans. Math. Softw., 24(1):130-156, 1998
and http://www.maths.uq.edu.au/expokit

@author: Ashley Milsted
"""
from __future__ import absolute_import, division, print_function

import scipy as sp
import scipy.linalg as la
from math import sqrt, log10, copysign, trunc

def gexpmv(A, v, t, anorm, m=None, tol=0.0, w=None, verbose=False, mxstep=500, break_tol=None):
      mxreject = 0 #matlab version has this set to 10
      delta = 1.2
      gamma = 0.9

      if break_tol is None:
            #break_tol = tol  
            break_tol = anorm*tol

      n = A.shape[0]

      if hasattr(A, "matvec"):
            matvec = A.matvec
      else:
            matvec = lambda v: sp.dot(A,v)

      if m is None:
            m = min(20, n-1)
      
      if len(A.shape) != 2 or A.shape[0] != A.shape[1]:
            raise ValueError("A is not a square matrix")
      
      if m >= n or m <= 0:
            raise ValueError("m is invalid")

      k1 = 2
      mh = m + 2

      ibrkflag = 0
      mbrkdwn  = m
      nmult    = 0
      nreject  = 0
      nexph    = 0
      nscale   = 0

      t_out    = abs( t )
      tbrkdwn  = 0.0
      step_min = t_out
      step_max = 0.0
      nstep    = 0
      s_error  = 0.0
      x_error  = 0.0
      t_now    = 0.0
      t_new    = 0.0
      
      eps = sp.finfo(A.dtype).eps
      rndoff = eps*anorm

      sgn = sp.sign(t)

      if w is None: #allow supplying a starting vector
            w = v.copy()
      else:
            w[:] = v
            
      beta = la.norm(w)
      vnorm = beta
      hump = beta 

      #obtain the very first stepsize ...

      SQR1 = sqrt( 0.1 )
      xm = 1.0/m
      p2 = tol*(((m+1)/sp.e)**(m+1)) * sqrt(2.0*sp.pi*(m+1))
      t_new = (1.0/anorm) * (p2 / (4.0*beta*anorm))**xm
      p1 = 10.0**(round( log10( t_new )-SQR1 )-1)
      t_new = trunc( t_new/p1 + 0.55 ) * p1

      vs = sp.zeros((m+2,n), A.dtype)  # to hold the Krylov subspace

      axpy = la.blas.get_blas_funcs('axpy', arrays=(vs[0, :], vs[0, :]))

      #step-by-step integration ...
      while t_now < t_out:

            nstep = nstep + 1
            t_step = min( t_out-t_now, t_new )

            #initialize Krylov subspace
            vs.fill(0)
            vs[0,:] = w / beta

            H = sp.zeros((mh,mh), A.dtype)

            #Arnoldi loop ...
            for j in range(1,m+1):
                  nmult = nmult + 1
                  vs[j,:] = matvec(vs[j-1,:])

                  for i in range(1,j+1):
                        #Compute overlaps of new vector Av with all other Kyrlov vectors
                        #(these are elements of an upper Hessenberg matrix)
                        hij = sp.vdot(vs[i-1,:], vs[j,:])
                        axpy(x=vs[i-1,:], y=vs[j,:], a=-hij) #orthogonalize new vector.
                        H[i-1,j-1] = hij #store matrix element

                  hj1j = la.norm( vs[j,:] )
                  #if the orthogonalized Krylov vector is zero, stop!
                  if hj1j <= break_tol:
                        if verbose:
                              print('breakdown: mbrkdwn =',j,' h =',hj1j)
                        k1 = 0
                        ibrkflag = 1
                        mbrkdwn = j
                        tbrkdwn = t_now
                        t_step = t_out-t_now
                        break

                  H[j,j-1] = hj1j
                  vs[j,:] *= 1.0/hj1j

            if ibrkflag == 0: #if we didn't break down
                  nmult = nmult + 1
                  vs[m+1,:] = matvec(vs[m,:])
                  avnorm = la.norm( vs[m+1,:] )

            #Orig: set 1 for the 2-corrected scheme
            H[m+1, m] = 1.0

            #loop while ireject<mxreject until the tolerance is reached
            ireject = 0

            #compute w = beta*V*exp(t_step*H)*e1 ...

            #First compute expH for a good step size
            while True:
                  nexph = nexph + 1
                  mx = mbrkdwn + k1 #max(mx) = m+2
                  #irreducible rational Pade approximation. scipy's implementation automatically chooses an order
                  expH = la.expm(sgn * t_step * H[:mx,:mx])
                  #nscale = nscale + ns #don't have this info
                  
                  #local error estimation
                  if k1 == 0: #if breakdown has occured (the Krylov subspace is complete)
                        err_loc = tol #matlab uses break_tol
                  else:
                        p1 = abs( expH[m,0] ) * beta #wsp(iexph+m) 
                        p2 = abs( expH[m+1,0] ) * beta * avnorm #avnorm is defined when k1 != 0
                        if p1 > 10.0*p2:
                              err_loc = p2
                              xm = 1.0/m
                        elif p1 > p2:
                              err_loc = (p1*p2)/(p1-p2)
                              xm = 1.0/m
                        else:
                              err_loc = p1
                              xm = 1.0/(m-1)

                  #reject the step-size if the error is not acceptable ...

                  if ( (k1 != 0) and (err_loc > delta*t_step*tol) and
                  (mxreject == 0 or ireject < mxreject) ):
                        t_old = t_step
                        t_step = gamma * t_step * (t_step*tol/err_loc)**xm
                        p1 = 10.0**(round( log10( t_step )-SQR1 )-1)
                        t_step = trunc( t_step/p1 + 0.55 ) * p1
                        if verbose:
                              print('t_step = ',t_old)
                              print('err_loc = ',err_loc)
                              print('err_required = ',delta*t_old*tol)
                              print('stepsize rejected, stepping down to: ',t_step)

                        ireject = ireject + 1
                        nreject = nreject + 1
                        if mxreject != 0 and ireject > mxreject:
                              print("Failure in gexpmv: ---")
                              print("The requested tolerance is too high.")
                              print("Rerun with a smaller value.")
                              iflag = 2
                              return
                  else:
                        break #step size OK (happens after a breakdown)

            #now update w = beta*V*exp(t_step*H)*e1 and the hump ...
            mx = mbrkdwn + max( 0, k1-1 ) #max(mx) = m+1
            w = beta * vs[:mx,:].T.dot(expH[:mx,0])
            beta = la.norm(w)
            hump = max( hump, beta )

            #suggested value for the next stepsize ...

            t_new = gamma * t_step * (t_step*tol/err_loc)**xm
            p1 = 10.0**(round( log10( t_new )-SQR1 )-1)
            t_new = trunc( t_new/p1 + 0.55 ) * p1

            err_loc = max( err_loc, rndoff )

            #update the time covered ...

            t_now = t_now + t_step

            #display and keep some information ...

            if verbose:
                  print('integration ', nstep, ' ---------------------------------')
                  #print('scale-square = ', nscale)
                  print('step_size = ', t_step)
                  print('err_loc   = ', err_loc)
                  print('next_step = ', t_new)

            step_min = min( step_min, t_step )
            step_max = max( step_max, t_step )
            s_error = s_error + err_loc
            x_error = max( x_error, err_loc )

            if mxstep == 0 or nstep < mxstep:
                  continue
            iflag = 1
            break

      return w, nstep < mxstep, nstep, ibrkflag==1, mbrkdwn, (x_error, s_error)