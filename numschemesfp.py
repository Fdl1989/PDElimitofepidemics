#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 10:34:34 2020

Created on Sat Feb 29 15:19:36 2020

@author: Francesco Di Lauro
@mail: F.Di-Lauro@sussex.ac.uk
Copyright 2020 Francesco Di Lauro. All Rights Reserved.
See LICENSE file for details
"""
# This file provides a numerical SolveForr using finite differences
# We implement FVM1-3 from the paper "Behaviour of different numerical schemes 
#for random genetic", by Shixin Xu et al. 
#
# FP equation: du/dt = 1/2*d2(au)/dx2-d(bu)dx
# Dirac delta as initial condition

import numpy as np
from matplotlib import pyplot as plt




#def SolveForFDCC1(x, t, xi, eps=1e-5, numericalscheme='FVM3', leftBC="Dir", rightBC="Dir"):
    # 
    # Implementation from Mohammadi, Borzi, 2015
'''
def PlotSol(x,t,  solFP, num=600, skip=False):
    # Plot the solution
    idx = np.linspace(start=1*skip, stop=solFP.shape[0]-1, endpoint=True, num=num, dtype=int)
    plt.figure()
    for i in idx:
        plt.plot(x, solFP[i], label="t="+str(t[i]))
    plt.legend(loc="best")
    
def fluxcoef(x):
    return x*(1-x)

#there are M+1 points

midPoints = np.hstack((-dx/2, (x[:-1]+x[1:])/2, 1+dx/2))

solFD = np.zeros((t.size, x.size))

sigma = dx

solFD[0] = np.exp(-(x-xi)**2/2/sigma**2)/np.sqrt(2*np.pi)/sigma
solFD[0] /= NormSol(x, solFD[0])

#diagonal -1
Cmat = -dt / dx**2 * (fluxcoef(x[:-1]))
#diagonal
Bmat = 1 + dt/dx**2 * (2*fluxcoef(x))
#diagonal+1
Amat = - (dt/dx**2) * (fluxcoef(x[1:]))

#implicit scheme ftw:
M = np.diag(Bmat)
M += np.diag(Amat, k=1)
M += np.diag(Cmat, k=-1)

M[0,1] = -fluxcoef(x[1])*2*dt/(dx**2)
M[-1,-2] = - 2*dt/dx**2*fluxcoef(x[-2])

    
for i in range(t.size-1):
    solFD[i+1] = np.linalg.solve(M, solFD[i])


#PlotSol(x,t,solFD, 10)



complete graph let's try
'''
def model_cap(u, k, N):
  return u[0] * (k**u[2]) * ((N-k)**u[2]) * (u[1]*(k-N/2.0)+N)
def model_cpq(u, k, N):
  return u[0] *N**(u[1]+u[2]-1)*(k**u[1]) * ((1-k)**u[2]) 


def FVM1(x,t,u,gamma,xi=0.1):
    N=100
    def sigma2(x):
        return model_cpq(u,x,N) + gamma*x
    def dsigma2(x):
        return u[0] *N**(u[1]+u[2]-1) *x**(u[1]-1)*(1-x)**(u[2]-1) *(u[1]*(1-x) -u[2]*x) +gamma
    def mu(x):
        return model_cpq(u,x,N) - gamma*x
        
    def NormSol(x, sol):
        return (sol[1:-1].sum() + (sol[0] + sol[-1])/2)*np.diff(x)[0]     

    dx = np.diff(x)[0]
    dt = np.diff(t)[0]
    solFD = np.zeros((t.size, x.size))
    
    sigma = dx
    
    solFD[0] = np.exp(-(x-xi)**2/2/sigma**2)/np.sqrt(2*np.pi)/sigma
    solFD[0] /= NormSol(x, solFD[0])
    
    #find the turning point for upwind
    xindex = 0
    for i in range(len(x)):
        if mu(x[i])<0 and x[i]>0:
            xindex = i
            break
    
    M = np.zeros((x.size,x.size))
    
    for row in range(x.size):
        if row == 0:
            M[0,1] = -dt/(N*dx**2)*sigma2(x[1]/2) 
            M[0,0] =1+dt/(N*dx**2)*sigma2(x[1]/2) + mu(x[1]/2)*2*dt/dx \
            -dt/(N*dx)*dsigma2(x[1]/2)
        elif row == x.size-1:
            M[-1,-2] = -dt/(N*dx**2) * sigma2((x[-1]+x[-2])/2) 
           
            M[-1,-1] = 1+dt/(N*dx**2) * sigma2((x[-1]+x[-2])/2) +dt/(2*N*dx) * dsigma2((x[-1]+x[-2])/2) - mu((x[-1]+x[-2])/2)*2*dt/dx
        else:
            #i-1
            if row>xindex:
                M[row,row-1]=- dt/(2*N*dx**2) *sigma2((x[row]+x[row-1])/2)
                M[row,row] =1+ dt/(2*N*dx**2) *(sigma2((x[row]+x[row-1])/2)+sigma2((x[row]+x[row+1])/2)) - mu((x[row]+x[row-1])/2)*dt/dx + dt/(2*N*dx)*dsigma2((x[row]+x[row-1])/2)
                M[row,row+1] = - dt/(2*N*dx**2) *sigma2((x[row]+x[row+1])/2) + mu((x[row]+x[row+1])/2)*dt/dx - dt/(2*N*dx)*dsigma2((x[row]+x[row+1])/2)
            else:
                M[row,row-1]=- dt/(2*N*dx**2) *sigma2((x[row]+x[row-1])/2) - mu((x[row]+x[row-1])/2)*dt/dx + dt/(2*N*dx)*dsigma2((x[row]+x[row-1])/2)                
                M[row,row] =1+ dt/(2*N*dx**2) *(sigma2((x[row]+x[row-1])/2)+sigma2((x[row]+x[row+1])/2)) + mu((x[row]+x[row+1])/2)*dt/dx - dt/(2*N*dx)*dsigma2((x[row]+x[row+1])/2)
                M[row,row+1] = - dt/(2*N*dx**2)*sigma2((x[row]+x[row+1])/2) 
     
    print(M[:4,:4])
    for i in range(t.size-1):
        solFD[i+1] = np.linalg.solve(M, solFD[i])
    
    return solFD





def FVM3(x,t, difflux, driftflux, xi=0.1):


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
        gamma =1
        N =100
        #xi=5e-3
        xi=0.1
        
        def diffusion(x):
            return (1.0/N)*(tau*N*x*(1-x) + gamma*x)
        
        def drift(x):
            drift=np.zeros(x.size)
            drift = tau*N*x*(1-x) - gamma*x
            return drift
                

        difflux = diffusion(x)
        driftflux = drift(x)
        resvec = FVM3(x,t,difflux,driftflux,xi)
        
        
        '''
        u=(tau,1,1)
        resvec2 = FVM1(x,t,u,gamma,xi)
        time = t[::10]
                
        
        distribution = np.loadtxt("distribution_complete_1,5e-2.txt")
        
        for i,sub in enumerate(ax):
            #weights = np.ones_like(distribution[i])/float(len(distribution[i]))
            if i!=0:
                binsize = int((np.max(distribution[i]) - np.min(distribution[i]))*N)-1
            else:
                binsize=99
            sub.hist(distribution[i], bins=binsize, density=True,stacked=True)
            sub.set_xticks([0,0.5,1], ['$0$', '$0.5$', '$1$'])
            #sub.set_xlim(0,1)
            #sub.set_ylim(-0.2,12)
            sub.plot(x,resvec[i*10], color='k')
            sub.plot(x,resvec2[i*10], color='r')

            sub.set_title('$t=%.2f$'%t[10*i])
        for i in range(len(ax)):
            for tick in ax[i].xaxis.get_major_ticks():
                tick.label.set_fontsize(12)
        for i in range(len(ax)):
            for tick in ax[i].yaxis.get_major_ticks():
                tick.label.set_fontsize(12)    
        '''