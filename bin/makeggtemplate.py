#!/usr/bin/env python

from pyggop import fast_flux_computation
import argparse

desc                          = '''Produce template for gamma-gamma opacity (Granot et al. 2008)'''
parser                        = argparse.ArgumentParser(description=desc)

parser.add_argument("-m",
                    help="",
                    required=False,type=float,default=0)

parser.add_argument("-b",help="",
                    required=False, default=0, type=float)

parser.add_argument("-a",help="Photon index", 
                    required=True, default=2.0, type=float)

parser.add_argument("-d",help="\Delta R / R0", required=True, type=float)

parser.add_argument("-r",help="R0", required=False, type=float, default=1.0)
parser.add_argument("-t",help="tau_star", required=False, type=float, default=1.0)

parser.add_argument("--plot",help="If yes, make a plot with the results",
                    required=False,choices=['yes','no'],default='no')

if __name__=="__main__":
    
    args = parser.parse_args()
    
    if args.plot.lower()=='yes':
        
        plot = True
    
    else:
        
        plot = False
    
    fast_flux_computation.go(args.m, args.b, args.a, args.d, args.r, args.t, plot)
