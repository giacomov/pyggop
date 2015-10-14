from __future__ import division
import numpy as np
from scipy import integrate as sciint

#from scipy import special as scispec

from scipy import interpolate as sciinterp
#from math import sqrt, exp, log10, sqrt

from pprint import pformat
import pickle
import sys

from libc.math cimport sqrt, exp, pow, log10

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
plt.ioff()

import functools, multiprocessing

from pyggop.grbod import tau_integral, reg

from pyggop.Tau import Tau

from pyggop.ParallelPool import ParallelPool

cdef double y2x(double y, double m):
    return (y**(-m-1)-1)/(m+1)

cdef double int_F0(double y, double m, double b, double a):
    mp1=m+1
    #eq.94
    res = ( mp1/(m + y**(-mp1)))**(1+a)
    res *= y**(b - 1- m*a/2 );
    return res

class FastFluxComputation(object):
    
    def __init__(self, double m, double b, double a, 
                       double DRbar, double R_0=1.0, 
                       double tau_star=1.0):
      
        self.m = m
        self.mp1 = self.m + 1
        self.b = b
        self.a = a
        self.DRbar = DRbar
        self.R_0 = R_0
        self.tau_star = tau_star
        
        self.prev = 1e10
        self.prevErr = 0
        
        #Pre-load interpolators
        self._setupInterpolator()
    
    def setTbar(self, Tbar):
        
        self.Tbar = Tbar
        
        ToT0 = 1.0 + Tbar
        
        self.ToT0 = ToT0
        
        R0oRL = ( ToT0 )**( -1.0 / self.mp1 ) #R0/RL(T)
        
        self.factor = ToT0**( (2 * self.b - self.m * self.a) / ( 2 * self.mp1 ) )
                
        self.ymin = R0oRL #should always be R0oRL
        
        self.ymax = min( 1, ( 1 + self.DRbar ) * R0oRL )
        
        #xmin = log10( self.ymin * (Tbar+1.0)**(1./(self.m+1)) - 1 )
        #xmax = log10( self.ymax * pow(self.Tbar + 1, 1.0 / self.mp1) - 1.0)
        
        #print("%s < y < %s" %(self.ymin, self.ymax))
        #print("Will ask between %s < X < %s" %(xmin,xmax))
            
    def setEpsilon(self, epsilon):
        
        self.epsilon = epsilon
    
    def _getUniqueName(self):
        
        return "%s-%s-%s-%s-%s" % (self.m, self.b, self.a, self.DRbar, self.tau_star)    
    
    def _setupInterpolator( self ):
        
        self.tau = Tau( self.m, self.b, self.a, 
                        self.DRbar, self.R_0, 
                        self.tau_star )
            
    def flux_integrand2(self, double y):
        
        cdef double mp1 = self.mp1
        cdef double Tbar = self.Tbar
        
        f0 = int_F0( y, self.m, self.b, self.a )
        
        #X = log10( self.Tbar + 1) * (self.mp1)
        
        Rt0oR0 = y * pow(Tbar + 1, 1.0 / mp1)
        
        X = log10( Rt0oR0 - 1.0)
        
        if X < -11:
             
             print("X is %s. Fixing to -11" %(X))
             
             X = -11
        
        x = ( pow( y, - mp1 ) - 1) / mp1
        Y = log10( sqrt(x) )
        
        F_gg = pow(10, self.tau(X, Y) )
        
        if F_gg < -0.1:
            
            print("Tau is %s , less than zero!" % F_gg)
        
        else:
            
            F_gg = max(0, F_gg)
        
        tau_0 = self.tau_star * self.epsilon**(self.a - 1)
        
        k = self.m * self.a * 0.5 + self.b - 1
        
        tau_0 *= pow(self.ToT0, k / mp1 )
        tau_0 *= pow( y , k )
        
        tau_gg = tau_0 * F_gg
        
        f = f0 * exp( - tau_gg )
        
        return f
    
    def getNotAttenuatedFlux(self):
        
        #non attenuated flux
        
        res, err = sciint.quad(int_F0, self.ymin, self.ymax,
                               args = ( self.m, self.b, self.a ), 
                               epsrel = 1.e-10, epsabs = 0, limit = 100)
        
        return res, err, self.factor * res
    
    def getAttenuatedFlux(self):
        
        #print("ymin = %s, ymax = %s" %(self.ymin, self.ymax))
        
        res, err = sciint.quad( self.flux_integrand2, self.ymin, 
                                self.ymax , 
                                limit=500, epsabs = 0, epsrel = 1e-3) 

        self.prev = res
        self.prevErr = res
        
        return res, err, self.factor * res

def worker(Tbar, eps, c):
    
    c.setTbar(Tbar)
    
    thisFluxes = []
    
    floor = False
    
    for epsilon in eps:
        
        if(floor):
          thisFluxes.append(0)
          continue
        
        c.setEpsilon(epsilon)
        
        #ToT0 = 1 + Tbar
        #print ToT0, epsilon
        
        #res1, err1, F0 = c.getNotAttenuatedFlux()
        
        #print "Flux integration without attenuation:", res1,err1, F0
        
        try:
          
          res2, err2, F = c.getAttenuatedFlux()
        
        except:
          
          print("Attenuated flux for (tbar, eps) = (%s,%s) failed!" %(Tbar, epsilon))
          
          res2, err2, F = 0,0,0
          
          raise
          
        thisFluxes.append(F)
        
        if(F < 1e-8):
          #Since the spectrum is monotonically decreasing,
          #let's stop here for this one
          floor = True
        
    pass
    
    return thisFluxes
    
pass #End of worker


def go(double m=0, double b=0, 
       double a=2, double DRbar=0.1,
       double R_0=1.0, double tau_star=1.0,
       int plot=False):
    
    c = FastFluxComputation(m, b, a, DRbar, R_0, tau_star)
    
    Tbars = np.logspace(-6,4,100)
    
    #print("%s" %(Tbars[40]))
    
    eps = np.logspace(-3,7,51)
    
    #Plot transparency
    alphas = np.linspace(0.1,1,Tbars.shape[0])[::-1]
    
    #alphas = alphas[5:]
    #Tbars = Tbars[5:]
    
    #print Tbars
    
    workWrap = functools.partial(worker,eps=eps,c=c)
        
    pool = ParallelPool( )
    
    allSpectra = np.zeros( ( Tbars.shape[0], eps.shape[0] ) )
    
    #res = map(workWrap, Tbars)
    
    for i, res in enumerate( pool.imap( workWrap, Tbars ) ):
     
      sys.stderr.write("\r%03i out of %i completed" %(i+1, Tbars.shape[0]))
      
      #Check that the computation succeeded
      if np.sum( np.isfinite(res) ) != len(res):
          
          raise RuntimeError("Error in computation! Got nan or inf")
            
      allSpectra[i,:] = res
     
      if plot:
         
          plt.plot(eps, res,c='black',alpha=alphas[i])
    
    sys.stderr.write("\n")
    
    pool.close()
    
    #Compute integrated spectrum
    F_time_int = np.zeros_like(eps)
    
    for i in range(F_time_int.shape[0]):
        
        F_time_int[i] = sciint.trapz(allSpectra[:,i], Tbars)
    
    if plot:
        
        plt.plot(eps, F_time_int, c='red')
    
        plt.ylim([1e-8,max( allSpectra.max() * 2, max(F_time_int) * 2)])
        plt.loglog()
        plt.savefig('flux_m%s_a%s_b%s_DR%s.png'%(str(m),str(a),str(b),str(DRbar)))
    
    filename='flux_m%s_a%.2f_b%s_DR%.2g.dat'%(str(m),a,str(b),DRbar)
    
    with open(filename, 'w+') as f:
        
        f.write("#Adim_energy flux\n")
        
        txt = "\n".join(map(lambda x:"%s %s" %x,zip(eps, F_time_int)))
        
        f.write(txt)
        f.write("\n")
        f.close()
    
    return filename
    
