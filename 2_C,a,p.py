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
from matplotlib import pyplot as plt
#from igraph import GraphBase as graph
from pyswarm import pso

def model_cap(u, k, N):
  return u[0]*(k**u[2]) * ((1-k)**u[2]) * (u[1]*(k-1/2.0)+1)
def LS2_cap(u, I, SI, N):
    # Compute continuous log_likelihood ak+1
    ak = model_cap(u, I, N)
    ak_hat = SI
    temp = ak-ak_hat

    dist = np.dot(temp, temp)
    return dist
def LS_cap(I,SI, N,  maxiter=3500):
    # Compute MLE estimator from Uk, Tk for model 2
    swarmsize = int(2*N)
    def temp(v):
        return LS2_cap(v,I,SI,N)
    LB = np.array([1e-1, -2, 0.5])
    UB = np.array([100, 2, 1.5])
    # Constrained optimization
    opt = pso(temp, lb=LB, ub=UB, swarmsize=swarmsize, maxiter=maxiter)

    return opt[0]


if __name__=='__main__':

    ERgamma = [5, 4.5, 7]
    ERtau =[1, 1, 4]
    ERk = [8.0,10.0,7.0]
    taufv = [4,3,0.8]
    '''
    u = np.zeros((7,3))
    for index,N in enumerate([100,500,1000,5000,10000,50000,100000]):
        tau = ERtau[0]
        counts = np.load('txtfiles/%d_E-R_tau=1.000_gamma=5.000_k=8_SI.npy'%N)
        #first guess for the MLE
        I = np.arange(0, counts.size,1)
        SI = counts*tau
        x = I/N
        if N > 1000:
            sample = int(N/1000)
            x = x[::sample]
            SI = SI[::sample]
        C_0,a_0,p_0 = LS_cap(x,SI/N,float(N))
        u[index]= np.array([C_0,a_0,p_0])
        print(C_0/1e-6)
        print(C_0/1e-3)
        print(u[index])
        fig = plt.figure()
        ax = fig.add_subplot(111)
        SI = counts*tau
        ax.plot(I/N,model_cap((C_0,a_0,p_0),I/N, SI.size-1), color='r',linestyle='--',alpha=0.7, label = "Cap, C=%f,a=%f,p=%f"%(C_0,a_0,p_0))
        ax.plot(I/N,SI/N, color='b')
        plt.title("N=%d"%N)
        plt.pause(0.5)
        
    np.savetxt("txtfiles/ER_0.txt",u)
    for index,N in enumerate([100,500,1000,5000,10000,50000,100000]):
        u[index][0] = u[index][0]/N**(2*u[index][2])
    ''' 
    Reggamma = [7,6,8]
    Regtau =[3.5,1,2.5]
    Regk = [8,9,7]
    taufv = [1.5,6,2]        
    u_reg = np.zeros((7,3))
    for i in range(2,3):
        for index,N in enumerate([100,500,1000,5000,10000,50000,100000]):
            #do the plots
            k = Regk[i]
            gamma = Reggamma[i]
            tau = Regtau[i]
            tauf = taufv[i]
            counts = np.load('txtfiles/%d_Reg_tau=2.500_gamma=8.000_k=7_SI.npy'%N)
            
            I = np.arange(0, counts.size,1)
            SI = counts*tau
            x = I/N
            if N > 1000:
                sample = int(N/1000)
                x = x[::sample]
                SI = SI[::sample]
            C_0,a_0,p_0 = LS_cap(x,SI/N,float(N))
            u_reg[index]= np.array([C_0,a_0,p_0])
            print(C_0/1e-6)
            print(C_0/1e-3)
            print(u_reg[index])
            fig = plt.figure()
            ax = fig.add_subplot(111)
            SI = counts*tau
            ax.plot(I/N,model_cap((C_0,a_0,p_0),I/N, SI.size-1), color='r',linestyle='--',alpha=0.7, label = "Cap, C=%f,a=%f,p=%f"%(C_0,a_0,p_0))
            ax.plot(I/N,SI/N, color='b')
            plt.title("N=%d"%N)
            plt.pause(0.5)

    np.savetxt("txtfiles/Reg_0.txt",u_reg)









    