#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 29 15:19:36 2020

@author: Francesco Di Lauro
@mail: F.Di-Lauro@sussex.ac.uk
Copyright 2020 Francesco Di Lauro. All Rights Reserved.
See LICENSE file for details
"""
import numpy as np
import os
import pandas as pd
from matplotlib import pyplot as plt
import random
from pyswarm import pso
from matplotlib import rc
import networkx as nx
import multiprocessing as mp

from fastgillespie import fast_Gillespie
#from igraph import GraphBase as graph
import sys
sys.path.insert(0, "Library")
rc('font',**{'family':'serif'})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)
def model_cap(u, k, N):
  return u[0] * (k**u[2]) * ((N-k)**u[2]) * (u[1]*(k-N/2.0)+N)
def LS2(u, I, SI, N):
    # Compute continuous log_likelihood ak+1
    ak = model_cap(u, I, N)
    ak_hat = SI
    temp = ak-ak_hat

    dist = np.dot(temp, temp)
    return dist
def LS_cap(I,SI, N, swarmsize=2500, maxiter=2500):
    # Compute MLE estimator from Uk, Tk for model 2

    def temp(v):
        return LS2(v,I,SI,N)
    LB = np.array([1.0/(N**2), -2, 0.1])
    UB = np.array([5.0/N, 2, 1.5])
    # Constrained optimization
    opt = pso(temp, lb=LB, ub=UB, swarmsize=swarmsize, maxiter=maxiter)

    return opt[0]

def simulations_SI(G,N,tau, gamma, tauf,seed, procnum):
    '''
    This bit of code is used in parallelization. It runs epidemics on a simple networks and returns everything
    '''
    #to do
    np.random.seed(seed)
    booleancheck = False
    while not booleancheck:
        if procnum < 100:
            model = fast_Gillespie(G,i0 = 5,tau = tau, gamma= gamma,tauf=tauf) # Setup the simulation with given parameters.
            model.run_sim() # Run the simulation.
            if model.I[-1] !=0:
                booleancheck=True
        else:
            model = fast_Gillespie(G,i0 = int(N),tau = tau, gamma= gamma,tauf=1) # Setup the simulation with given parameters.
            model.run_sim() # Run the simulation.
            booleancheck = True
    #print(len(model.I),len(model.SI),len(model.tk))
    return model.SI,model.tk
    #return model.time_grid, model.I
def  rawakandcap(networkchoice, gamma, tau, k,tauf, N, savedestination, number_of_networkgen,number_of_epid, seed_vector):
    #distribution around the steady state

    name = "%s_tau=%.3f_gamma=%.3f_k=%d"%(networkchoice,tau,gamma,k)

    SI_threads=np.zeros(N+1)
    tk_threads=np.zeros(N+1)
    for graph in range(number_of_networkgen):
        if graph%5==0:
            print(".",end = '')
        random.seed(seed_vector[graph*number_of_epid])
        if networkchoice == 'E-R':
            G= nx.fast_gnp_random_graph(N,k/float(N-1.0))
        else:
            G= nx.random_regular_graph(k,N)

        pool = mp.Pool(mp.cpu_count())

        epidemies_object = [pool.apply_async(simulations_SI,args=(G,N,tau, gamma, tauf,seed_vector[procnum], procnum)) for procnum in range(number_of_epid)]
        #I_threads = [return_dict.values()[r][0] for r in range(len(return_dict.values()))]
        pool.close()
        pool.join()
        for r in epidemies_object:
            SI_threads += r.get()[0]
            tk_threads += r.get()[1]
            #plt.plot(r.get()[0], r.get()[1])

        del epidemies_object
        del G
        del pool
    print("")
    tk_threads[np.where(tk_threads==0)] = 1.0
    avg_SI = SI_threads/tk_threads;
    np.save(savedestination+'%s_SI.npy'%name, avg_SI)

    #SI = avg_SI*tau
    #I = np.arange(0,N+1,1)
    #C_0,a_0,p_0 = LS_cap(I,SI,float(N), swarmsize=5000)
    #u= np.array([C_0,a_0,p_0])
    #print(u)
    #np.savetxt(savedestination+"%s_cap.txt"%name, u)

    return True



if __name__=="__main__":

    ERgamma = [5, 4.5, 7]
    ERtau =[1, 1, 4]
    ERk = [8.0,10.0,7.0]
    taufv = [4,3,0.8]
    Reggamma = [7,6,8]
    Regtau =[3.5,1,2.5]
    Regk = [8,9,7]
    
    number_of_networkgen = 50

    number_of_epid = 200


    from datetime import datetime

    startTime = datetime.now()
    #Do the simulations
    '''
    for i in range(1):
        for N in [100,500,1000,5000,10000,50000,100000]:
            k = ERk[i]
            gamma = ERgamma[i]
            tau = ERtau[i]
            tauf = taufv[i]
            R0_er = tau*(k*(N-2)/(N-1.0))/(tau+gamma)
            #print(R0_er)
            #print(tauf)
            networkchoice='E-R'
            seed_vector=np.array([j*(124)+23 for j in range(number_of_networkgen*number_of_epid)])

            rawakandcap(networkchoice, gamma, tau, k,tauf, N, "txtfiles/%d_"%N, number_of_networkgen,number_of_epid, seed_vector)
            print("Size %d done, time used:"%N)
            print(datetime.now()-startTime)
            startTime = datetime.now()
    '''
    fig,ax = plt.subplots(1,2, figsize=(6.2,3.5))
    ax = ax.ravel()
    fig.subplots_adjust(wspace=0.3)
    from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
    axins = zoomed_inset_axes(ax[1], 30, loc='center') # zoom-factor: 2.5, location: upper-le

    for i in range(1):
        for N in [100,500,1000,5000,10000,50000,100000]:
            #do the plots
            k = ERk[i]
            gamma = ERgamma[i]
            tau = ERtau[i]
            tauf = taufv[i]
            SI = np.load('txtfiles/%d_E-R_tau=1.000_gamma=5.000_k=8_SI.npy'%N)
            I = np.arange(0,N+1,1)
            ax[0].plot(I,SI)
            ax[1].plot(I/N,SI/N,label='%d'%N)

            axins.plot(I/N, SI/N)
        x1, x2, y1, y2 = 0.495, 0.515, 1.898, 1.91 # spambulanzeecify the limits
        #axins.plot(I/N, gamma*I/N, color='k')
        #x1, x2, y1, y2 = 0.335, 0.355, 1.72, 1.74 # specify the limits
        
        axins.set_xlim(x1, x2) # apply the x-limits
        axins.set_ylim(y1, y2) # apply the y-limits
        axins.set_xticks([0.5, 0.51])
        axins.set_yticks([1.9, 1.91])
        ax[1].plot(I,gamma/tau*I, color='k', label=r'$\frac{\gamma}{\tau} \frac{k}{N}$')
        ax[0].set_xlabel(r"$k$", labelpad=-1,size=11)
        ax[0].text(5000,195000, r'$a)$', size=11)
        ax[1].text(0.05,1.95000, r'$b)$', size=11)
        ax[0].set_ylabel(r"$a_k$",size=11)
        ax[1].set_xlabel(r"$k/N$",labelpad=-1,size=11)
        ax[1].set_ylabel(r"$\frac{a_k}{N}$", labelpad = -1.5,size=14)
        ax[1].legend(loc='best',ncol=2, fontsize=8)
        ax[0].set_xlim(0,N)
        ax[0].set_ylim(0,1.1*max(SI))
        ax[1].set_xlim(0,1)
        ax[1].set_ylim(0,1.1*max(SI/N))
        from mpl_toolkits.axes_grid1.inset_locator import mark_inset
        mark_inset(ax[1], axins, loc1=1, loc2=2, fc="none", ec="0.5")
        plt.savefig("ER_1_scaling.eps", format='eps')
    

    

    '''
    savedestination = '/home/ld288/Dropbox/Codes/Pde_limit/'
    
    
    seed_vector=np.array([j*(124)+23 for j in range(number_of_networkgen*number_of_epid)])

    for i in range(2,3):
        for N in [100,500,1000,5000,10000,50000,100000]:
            k = Regk[i]
            gamma = Reggamma[i]
            tau = Regtau[i]
            tauf = taufv[i]
            R0_er = tau*(k*(N-2)/(N-1.0))/(tau+gamma)
            #print(R0_er)
            #print(tauf)
            networkchoice='Reg'

            rawakandcap(networkchoice, gamma, tau, k,tauf, N, "txtfiles/%d_"%N, number_of_networkgen,number_of_epid, seed_vector)
            print("Size %d done, time used:"%N)
            print(datetime.now()-startTime)
            startTime = datetime.now()

    '''
    fig,ax = plt.subplots(1,2, figsize=(6.2,3.5))
    ax = ax.ravel()
    fig.subplots_adjust(wspace=0.3)
    from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
    axins = zoomed_inset_axes(ax[1], 30, loc='center') # zoom-factor: 2.5, location: upper-le
    taufv = [1.5,6,2]

    for i in range(2,3):
        for N in [100,500,1000,5000,10000,50000,100000]:
            #do the plots
            k = Regk[i]
            gamma = Reggamma[i]
            tau = Regtau[i]
            tauf = taufv[i]
            SI = np.load('txtfiles/%d_Reg_tau=2.500_gamma=8.000_k=7_SI.npy'%N)
            I = np.arange(0,N+1,1)
            ax[0].plot(I,SI)
            ax[1].plot(I/N,SI/N,label='%d'%N)

            axins.plot(I/N, SI/N)
        x1, x2, y1, y2 = 0.493, 0.513, 1.61, 1.62 # spambulanzeecify the limits
        axins.plot(I/N, gamma/tau*I/N, color='k')
        #x1, x2, y1, y2 = 0.335, 0.355, 1.72, 1.74 # specify the limits
        
        axins.set_xlim(x1, x2) # apply the x-limits
        axins.set_ylim(y1, y2) # apply the y-limits
        axins.set_xticks([0.5, 0.51])
        axins.set_yticks([ 1.61, 1.62])
        ax[1].plot(I,gamma/tau*I, color='k', label=r'$\frac{\gamma}{\tau} \frac{k}{N}$')
        ax[0].set_xlabel(r"$k$", labelpad=-1,size=11)
        ax[0].text(5000,160000, r'$a)$', size=11)
        ax[1].text(0.05,1.6000, r'$b)$', size=11)
        ax[0].set_ylabel(r"$a_k$",size=11)
        ax[1].set_xlabel(r"$k/N$",labelpad=-1,size=11)
        ax[1].set_ylabel(r"$\frac{a_k}{N}$", labelpad = -1.5,size=14)
        ax[1].legend(loc='best',ncol=2, fontsize=8)
        ax[0].set_xlim(0,N)
        ax[0].set_ylim(0,1.1*max(SI))
        ax[1].set_xlim(0,1)
        ax[1].set_ylim(0,1.1*max(SI/N))
        from mpl_toolkits.axes_grid1.inset_locator import mark_inset
        mark_inset(ax[1], axins, loc1=1, loc2=2, fc="none", ec="0.5")
        plt.savefig("Reg_2_scaling.eps", format='eps')

