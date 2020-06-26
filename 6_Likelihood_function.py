#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 29 15:19:36 2020

@author: Francesco Di Lauro
@mail: F.Di-Lauro@sussex.ac.uk
Copyright 2020 Francesco Di Lauro. All Rights Reserved.
See LICENSE file for details

"""

import numpy as np
from FVM3_epsilon import CAPmodel, diffusion, drift,FVM3
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt

def likelihood_base(x_i,x_f,t_f, difflux,driftflux, epsilon):
        spacegrid = np.linspace(0,1,100)
        timegrid_solver = np.linspace(0,t_f,100)
        resvec = FVM3(spacegrid,timegrid_solver,difflux,driftflux,x_i)
        
        f = interp1d(spacegrid, resvec[-1])
        return max(f(x_f), 1e-5)

def loglikelihood(u,times,data, epsilon, gamma =1, N=100):
    dt = np.diff(times)[0]
    x = np.linspace(0,1,100)
    difflux=diffusion(x,N,u,gamma,epsilon)
    driftflux=drift(x,u,gamma,N)
    loglikelihood = 0
    for y in range(1,len(data)):
#        likelihood = likelihood*likelihood_base(data[y-1],data[y], dt, difflux,driftflux, epsilon) 
         logeval = likelihood_base(data[y-1],data[y], dt, difflux,driftflux, epsilon)          
         loglikelihood += np.log(logeval)
         #print(logeval)
    return -loglikelihood

if __name__ =='__main__':        
 
        #tau = 6e-3
        #tau=1.5e-2
        gamma =4.5
        N =1000
        #xi=5e-3
        #xi=0.1
        epsilon = 1e-3
        #u=(tau*N,0,1)

        u = np.loadtxt("E-R_tau=1.000_gamma=4.500_k=10_cap.txt", unpack=True)
        u[0] = u[0]*(N**(2*u[2]))
        data = np.loadtxt("Dataset_ER.out")
        data[:,1] /= N
        #t_grid = np.linspace(0,5,100)
        x_grid = np.linspace(0,1,100)
        
        #cap = CAPmodel(x_grid,u,N)
        
        #difflux = diffusion(x_grid,N,u,gamma,epsilon)
        #driftflux = drift(x_grid,u,gamma,N)
        Cvector = np.linspace(0.8*u[0],1.2*u[0], 5)
        logevaluations=[]
        
        #L-BFGS-B
        #tollerance ~0.01
        #put constraints
        '''
        for C in Cvector:
            u_C = np.array([C,u[1],u[2]])
            
            logevaluations.append(loglikelihood(u_C,data[:,0],data[:,1], epsilon, gamma =gamma, N=N))
        '''
        from scipy.optimize import minimize
        x0 = np.array([8.5, 0.1, 1])
        bounds = [(1, 20),(0,0.5),(0.5,1.5)]
        
        np.random.seed(541994)
       
        res = minimize(loglikelihood, x0, args=(data[:,0],data[:,1], epsilon, gamma, N), bounds = bounds,tol=1e-2, options={'maxiter':100},method='L-BFGS-B')
        minimizer_kwargs = {"method": "L-BFGS-B"}
        #consider using this to avoid local minima
        #res=optimize.basinhopping(nethedge,guess,niter=100,minimizer_kwargs=minimizer_kwargs)
  
        print(res.success)
        print(res.x)
        
        
        fig= plt.figure(figsize=(3,2.6))
        plt.subplots_adjust(left=0.22, bottom=0.2, right=0.94, top=0.95, wspace=0, hspace=0)
        ax0 = fig.add_subplot(111)
        ax0.set_xlim(0, 1)
        ax0.set_ylim(0, 2.5)

        ax0.set_xlabel(r"$k/N$", size=9)
        ax0.set_xticklabels([r"$0$",r"$1$",r"$2$",r"$3$",r"$4$",r"$5$"], size=7)
        
        ax0.set_yticklabels([r"$0.0$",r"$0.5$",r"$1.0$",r"$1.5$",r"$2.0$",r"$2.5$"], size=7)
        
        ax0.set_ylabel(r"$\frac{a_k}{N}$", size=11)
        
        ax0.plot(x_grid,CAPmodel(x_grid,u,N), label='True (C,a,p)')
        ax0.plot(x_grid,CAPmodel(x_grid,res.x,N), color='k', linestyle='--', linewidth=2, label='MLE')        
        ax0.plot(x_grid, CAPmodel(x_grid,x0, N), label='Initial guess')        
        ax0.legend(loc="best")

        plt.savefig("Inference_ER.png",format='png', dpi=400)

        '''
        resvec = FVM3(x_grid,t_grid,difflux,driftflux,xi)        
        plt.plot(x_grid,resvec[-1])
        
        newvector = np.zeros_like(x_grid)
    
        for index,i in enumerate(x_grid):
           newvector[index]=likelihood_base(xi,i,5, difflux,driftflux, epsilon)         
           
        plt.plot(x_grid,newvector,color='r')
        '''
        