#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 08:57:47 2020

@author: ld288
"""

# This file provides a numerical SolveForr using finite differences
# We implement FVM-3 from the paper "Behaviour of different numerical schemes 
#for random genetic", by Shixin Xu et al.
#
# FP equation: du/dt = 1/2*d2(au)/dx2-d(bu)dx
# Dirac delta as initial condition

import numpy as np
from matplotlib import pyplot as plt

def CAPmodel(x,u,N):
    #return u[0]*N**(2*u[2])*((x**u[2])*((1-x)**u[2])*(u[1]*(x - 0.5) +1))
    return u[0]*((x**u[2])*((1-x)**u[2])*(u[1]*(x - 0.5) +1))
def diffusion(x,N,u,gamma,epsilon):    
    diff = (1.0/N)*(CAPmodel(x,u,N) + gamma*x)
    diff[0] = epsilon
    return diff

def drift(x,u,gamma,N):
    drift = CAPmodel(x,u,N) - gamma*x
    return drift




def FVM3(x,t,  difflux, driftflux, xi=0.1):
    '''
    x: vector like - space grid
    t: vector like - time grid
    difflux: vector - diffusion evalued at x (domain [0 - 1])
    driftflux: vector - drift evalued at x (domain [0 - Tf])
    xi: float - initial condition  (in (0,1])
    '''

    def NormSol(x, sol):
        return (sol[1:-1].sum() + (sol[0] + sol[-1])/2)*np.diff(x)[0]    
    dx = np.diff(x)[0]
    dt = np.diff(t)[0]
    solFD = np.zeros((t.size, x.size))
    
    sigma = dx/2
    
    solFD[0] = np.exp(-(x-xi)**2/2/sigma**2)/np.sqrt(2*np.pi)/sigma
    solFD[0] /= NormSol(x, solFD[0])
    
    
    
    #diagonal -1
    Cmat = -dt / dx**2 * (  difflux[:-1]/2)   -dt/dx*driftflux[:-1]/2
    #diagonal
    Bmat = 1 + dt/dx**2 * (difflux)
    #diagonal+1
    Amat =  (dt/dx**2) * (-difflux[1:]/2) + dt/dx *driftflux[1:]/2
    
    M = np.diag(Bmat)
    M += np.diag(Amat, k=1)
    M += np.diag(Cmat, k=-1)
    
    M[0,1] = -dt/(dx**2)*(difflux[1]) + dt/dx*driftflux[1] 
    M[0,0] = 1 + dt/(dx**2) *difflux[0] + dt/dx*driftflux[0]
    M[-1] = 0
    M[-1,-1]=1 +dt/(dx**2)*difflux[-1] - dt/dx*driftflux[-1]
    M[-1,-2] = - dt/(dx**2)*difflux[-2] - dt/dx*driftflux[-2]
    
    solFD[1] = np.linalg.solve(M,  solFD[0])
    #second order in time
    #updated M matrix
    As = np.eye(M.shape[0])
    M = (M-As)*2/3 + As
        
    for i in range(1,t.size-1):
        solFD[i+1] = np.linalg.solve(M,  4*solFD[i]/3-solFD[i-1]/3)
    
    return solFD
    










if __name__ =='__main__':
        fig,ax = plt.subplots(2,5, figsize=(10,8))
        ax = ax.ravel()
        
        t = np.linspace(0,5,100)
        x = np.linspace(0,1,100)        
        #tau = 6e-3
        tau=1.5e-2
        gamma =8
        N =100
        #xi=5e-3
        xi=5e-3
        epsilon = 1e-3
        epsilon = 0
        u=(tau/N,0,1)

        u=regularnetworkcap=np.loadtxt("regularnetwork.txt")
        
        u = u[2]
        N=1000
        difflux = diffusion(x,N,u,gamma,epsilon)
        driftflux = drift(x,u,gamma)
        
        resvec = FVM3(x,t,difflux,driftflux,xi)
        
        
        #Complete network SIS epidemic distribution, run with completegraphtest.py
        
        #distribution = np.loadtxt("distribution_complete_1,5e-2.txt")
        
        for i,sub in enumerate(ax):
            #weights = np.ones_like(distribution[i])/float(len(distribution[i]))
           # if i!=0:
           #     binsize = int((np.max(distribution[i]) - np.min(distribution[i]))*N)-1
           # else:
           #     binsize=99
                
            #sub.hist(distribution[i], bins=binsize, density=True,stacked=True)
            sub.set_xticks([0,0.5,1], ['$0$', '$0.5$', '$1$'])
            #sub.set_xlim(0,1)
            #sub.set_ylim(-0.2,12)
            sub.plot(x,resvec[i*10], color='k')

            sub.set_title('$t=%.2f$'%t[10*i])
 
        for i in range(len(ax)):
            for tick in ax[i].xaxis.get_major_ticks():
                tick.label.set_fontsize(12)
        for i in range(len(ax)):
            for tick in ax[i].yaxis.get_major_ticks():
                tick.label.set_fontsize(12)    