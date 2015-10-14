import numpy as np
from multiprocessing import Pool

from grbod import *

import os, sys
import pickle

import scipy.interpolate
from math import log10

import matplotlib.pyplot as plt

from pyggop.ParallelPool import ParallelPool

#This is the actual computation
def func(DRbar, R_0, b, m, a, xx, yy):
    
    R0_hat = 1.0 / ( 1.0 + 10**xx ) # = R_0/R_t0
    
    DR_hat = DRbar * R0_hat / R_0
    
    x = ( 10**yy )**2
    
    tau = tau_integral(x, R0_hat, DR_hat, 
                       b, m, a,
                       reg={'Re':0,'Rt':1.e-4},
                       verbose=True)
    return tau

def data_stream(DRbar, R_0, b, m, a, XX, YY):
    for idx, xx in np.ndenumerate(XX):
        yield idx, (DRbar, R_0, b, m, a, xx, YY[idx])

def proxy(args):
    return args[0], func(*args[1])

class Tau( object ):
    
    def __init__(self, m, b, a, DRbar, R_0=1.0, tau_star=1.0 ):
        
        self.m = float(m)
        self.b = float(b)
        self.a = float(a)
        self.DRbar = float( DRbar )
        self.R_0 = R_0
        self.tau_star = tau_star
        
        self.name = self._getUniqueName()
                
        self.loadLookupTable()
        
    def _getUniqueName(self):
        
        return "%s-%s-%.2f-%.2g-%s" % (self.m, self.b, self.a, 
                                   self.DRbar, self.tau_star)
    
    def loadLookupTable(self):
        
        if not os.path.exists( '%s.pkl' % self.name ):
            
            #Lookup table does not exists. Create it
            
            self.computeLookupTable()
        
        #Load the lookup table
        results = pickle.load(open('%s.pkl'% self.name,'r'))            
        
        
        #Get the extremes for the interpolation
        
        self.minX,self.maxX = results['X'].min(),results['X'].max()
        self.minY,self.maxY = results['Y'].min(),results['Y'].max()
        
        #print results['comment']
        
        self.tau_interp2 = scipy.interpolate.interp2d( results['X'],
                                            results['Y'],
                                            results['Z'].transpose(),
                                            bounds_error=True)
    
    def compute(self, XX, YY):
        
        result = np.zeros(shape=(XX.shape))
                
        pool = ParallelPool( )
                
        pool_res = pool.map(proxy, data_stream(self.DRbar, self.R_0, 
                                               self.b, self.m, self.a,
                                               XX, YY))
        
        pool.close()
        
        for k,v in pool_res:
        
            result[k] = v
        
        return result
    
    def computeLookupTable( self, plot=True ):
        
        X = np.linspace(-11, 3, 50)#log10(R_t0/R0-1)
        
        Y = np.concatenate((np.arange(-6, -4,1 / 2.0),
                            np.arange(-4, -1,1/ 2.0), 
                            np.arange(-1, -0.04, 0.1/ 4.0),
                            np.arange(-0.04, 0.08, 0.01 / 4.0), 
                            np.arange(0.08, 0.9, 0.1/ 4.0),
                            np.arange(1, 2.2,0.2/ 2.0)))
        
        #Y = np.concatenate((np.arange(-4, -1,1 / 2.0), 
        #                    np.arange(-1, -0.04, 0.1 / 2.0),
        #                    np.arange(-0.04, 0.08, 0.02 / 2.0 ), 
        #                    np.arange(0.08, 0.9, 0.1 / 2.0),
        #                    np.arange(1, 2.2,0.2 / 2.0)))
        
        XX, YY = np.meshgrid(X, Y, indexing='ij', sparse=False, copy=True)
        
        Z = self.compute(XX, YY)
        
        idx = np.isfinite(Z)
        
        Z[~idx] = 1e-27
        
        idx = Z <= 0
        Z[idx] = 1e-27
        
        print("Zmin = %s, Zmax = %s" %(Z.min(),Z.max()))
        
        if plot:
            
            plt.figure(1)
            plt.contourf(X, Y, np.log10(Z.transpose()), 20)
            plt.colorbar()
            plt.savefig("interpolation_data.png")
        
        final = {'X':X, 'Y':Y, 'Z':np.log10(Z)}
        final['comment'] = "X = log10(R_t0/R0-1)\nY = log10(gth_t0) = log10(sqrt(x))\nZ = log10(tau_integral)"
        
        pickle.dump(final, open( '%s.pkl' % self.name, 'w' ))
    
    def __call__(self, X, Y):
        
        try:
            
            val = self.tau_interp2( X, Y )
            #val = log10( func(self.DRbar, self.R_0, self.b, self.m, self.a, X, Y) )
        
        except:
            
            msg = ("Request (X,Y) = (%s, %s) could not be satistfied. " %(X,Y))
            msg += ("Interpolation range is %s < X < %s , %s < Y < %s" %(self.minX,self.maxX,self.minY,self.maxY))
            
            sys.stderr.write(msg)
            sys.stderr.write("\n")
            
            raise ValueError(msg)
            
        return val
            
