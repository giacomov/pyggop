# cython: profile=True

from __future__ import division
import numpy as np
from scipy import integrate as sciint
from scipy import special as scispec
#from math import sqrt, exp

from libc.math cimport sqrt, exp, pow

from pprint import pformat
import warnings

# the various functions need some regularization unfortunately,
# due to simplifications/approximations  in the analytical
# treatments. The following aim at controlling these regularizations
reg={
    'Re':0,
    'Rt':1.e-8,
    }

#def y2x(double y, double m):
#    return (y**(-m-1)-1)/(m+1)

#============================================
#This substitute the scipy version of hyp2f1
#for speed. hygfz_ is defined in specfun.c

cdef extern:
    
    void hygfz_(double *a, double *b, double *c, 
                complex *x, complex *out)

cdef cython_hygfz( double a, double b, double c, complex x):
    
    cdef complex out
    hygfz_( &a, &b, &c, &x, &out)
    
    return out.real

# ==========================================

cdef double fm(double R_t, double x, double m):
    return ( 1 + x*(m+1)*(1-1/R_t) ) / R_t**(m+1)

cdef double inner_tau_integrand(double R_e, double R_t, double x, double b, double m, double a, reg):
    mp1=m+1
    f_m = fm(R_t, x, m)
    #R_e = R_e - 1.e-14 #reg['Re']
    #if R_e==1: print f_m, R_e, R_t, x
    gth_r2 = R_e*(f_m - R_e**(m+1))/((m+1)*(1-R_e))
    gth_r = sqrt(gth_r2);
    gth_t2 = x/R_t**(m+2)
    gth_t = sqrt(gth_t2);
    
    fac = ( (2*mp1)**(1+a) * R_e**(b+a+m*(2+a)/2) * (1-R_e)**a)\
    / ((m+1) * R_e**(m+1) * (1-R_e) + f_m - R_e**(m+1))**(1+a);
    
    if a == 1:
        #H_a = 1 + z/2;
        zma_H_a = (gth_r2+gth_t2)/4.
    elif a == 2:
        #H_a = 1 + z + (3/8)*z**2;
        zma_H_a = (gth_t2**2 + gth_r2**2)/16 + gth_t2*gth_r2/4
    elif a == 3:
        #H_a = 1 + (3/2)*z + (9/8)*z*z + (5/16)*z*z*z;
        zma_H_a = (gth_r2**3+9*gth_r2**2*gth_t2 + 9*gth_t2**2*gth_r2 + gth_t2**3)/64
    else:
        z_m = (1./4.)*(gth_r-gth_t)**2
        
        if z_m==0:
           
           zma_H_a = 0
        
        else:
           z = gth_t*gth_r / z_m
           
           pppp = pow(z_m, a)
           
           zma_H_a = pppp * cython_hygfz( -a, 0.5, 1, -z )
           
           #zma_H_a = pppp *  scispec.hyp2f1(-a,0.5,1,-z)
        
    f = fac * zma_H_a;
    
    return f


cdef double outer_tau_integrand(double rt, double x, double R_0, double DR, double b, double m, double a, reg):
    mp1=m+1
    R_emin = R_0/rt
    f_m = fm(rt, x, m)
    R_emax = min( (R_0+DR)/rt, 1, f_m**(1/mp1) )

    #Is this still necessary? R_emax = max(R_0/rt, R_emax )
    #if R_emax < R_0/rt : return 0.

    cst = rt**(b-2+m*a/2)
    
    args = (rt, x, b, m, a, reg)

    res, err = sciint.quad(inner_tau_integrand, R_emin, R_emax, 
                           args=args, epsrel = 1e-2, epsabs = 0)
        
    return cst*res


def tau_integral(double x, double R0_hat, double DR_hat, double b, double m, double a, reg=reg, verbose=False):

    out1 = 0,0
    if reg['Rt'] != 0:
      out1 = sciint.quad(outer_tau_integrand, 1., 1.+ reg['Rt'],
                args=(x, R0_hat, DR_hat, b, m, a, reg), 
                limit=1000, full_output = 1, epsrel = 1e-2, epsabs = 0)
      #if out1[0]==0.0:print out1[0], out1[1]
      #if out1[-1].__class__==str: print 'Tau integral 1: ',x, R0_hat, DR_hat, out1[0], out1[1], out1[-1]

    R_t_max = np.infty
    out = sciint.quad(outer_tau_integrand, 1.+ reg['Rt'], R_t_max,
              args=(x, R0_hat, DR_hat, b, m, a, reg), 
              limit=50, full_output = 1, epsrel = 1e-2, epsabs = 0)
    
    #if out[-1].__class__==str: print 'Tau integral 2: ',x, R0_hat, DR_hat, out[0], out[1], out[-1]
    return out[0]+out1[0]
