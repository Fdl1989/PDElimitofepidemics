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
import random
from matplotlib import pyplot as plt
from matplotlib import rc
import networkx as nx
from fastgillespie import fast_Gillespie
#from igraph import GraphBase as graph
import sys
sys.path.insert(0, "Library")
rc('font',**{'family':'serif'})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

if __name__=="__main__":    
    '''
    Regulear
    '''
    
    Reggamma = [7,6,8]
    Regtau =[3.5,1,2.5]
    Regk = [8,9,7]
    taufv = [4,4,4]
    np.random.seed(203)
    random.seed(201)       
    N=1000
    fig,ax = plt.subplots(1,1, figsize=(2.5,2.3))
    fig.subplots_adjust(wspace=0.6, hspace=0.6)

    for i in range(2,3):
        k = Regk[i]
        gamma = Reggamma[i]
        tau = Regtau[i]   
        tauf = taufv[i]
        R0_er = tau*(k*(N-2)/(N-1.0))/(tau+gamma)
        print(R0_er)    
        G= nx.random_regular_graph(k,N)
        
        for i in range(10):
            model = fast_Gillespie(G,i0 = 5,tau = tau, gamma= gamma,tauf=tauf) # Setup the simulation with given parameters.
            model.run_sim() # Run the simulation. 
            if model.I[-2]>10:
                ax.plot(model.time_grid, model.I/1000, color='blue', alpha=0.6)
    ax.set_xlim(0,4)
    ax.set_ylim(0,1)
    ax.set_yticks([0,0.2,0.4,0.6,0.8,1.0])
    ax.set_yticklabels([r"$0.0$",r"$0.2$",r"$0.4$",r"$0.6$",r"$0.8$",r"$1.0$"], size=7)
    ax.set_xticks([0,1,2,3,4])
    ax.set_xticklabels([r"$0$",r"$1$",r"$2$",r"$3$",r"$4$"], size=7)
    ax.set_xlabel(r"$t$", size = 9, labelpad=-0.4)
    ax.set_ylabel(r"$I(t)$", size = 9, labelpad=-3.8)        
    ax.text(0.1,0.91, r"$a)$", size=11)
    ax.tick_params(axis="both", pad=1.2)

    plt.savefig("Images/epiexamples_reg.png",format='png',dpi=250)

    '''
    Erdos Renyi
    '''
    ERgamma = [5, 4.5, 7]
    ERtau =[1, 1, 4]
    ERk = [8.0,10.0,7.0]
    taufv = [4,4,4]

    number_of_networkgen = 50
    
    number_of_epid = 200
    
    
    np.random.seed(203)
    random.seed(201)    
    N=1000
    fig,ax = plt.subplots(1,1, figsize=(2.5,2.3))
    fig.subplots_adjust(wspace=0.6, hspace=0.6)

    for i in range(3):
        k = ERk[i]
        gamma = ERgamma[i]
        tau = ERtau[i]   
        tauf = taufv[i]
        R0_er = tau*(k*(N-2)/(N-1.0))/(tau+gamma)
        print(R0_er)    
        networkchoice='E-R'   
        G = nx.fast_gnp_random_graph(N,k/float(N-1.0))
        
        for i in range(10):
            model = fast_Gillespie(G,i0 = 5,tau = tau, gamma= gamma,tauf=tauf) # Setup the simulation with given parameters.
            model.run_sim() # Run the simulation. 
            if model.I[-2]>10:
                ax.plot(model.time_grid, model.I/1000, color='blue', alpha=0.6)
    ax.set_xlim(0,4)
    ax.set_ylim(0,1)
    ax.set_yticks([0,0.2,0.4,0.6,0.8,1.0])
    ax.set_yticklabels([r"$0.0$",r"$0.2$",r"$0.4$",r"$0.6$",r"$0.8$",r"$1.0$"], size=7)
    ax.set_xticks([0,1,2,3,4])
    ax.set_xticklabels([r"$0$",r"$1$",r"$2$",r"$3$",r"$4$"],  size=7)
    ax.set_xlabel(r"$t$", size = 9, labelpad=-0.4)
    ax.set_ylabel(r"$I(t)$", size = 9, labelpad=-3.8)        
    ax.text(0.1,0.91, r"$b)$", size=11)
    ax.tick_params(axis="both", pad=1.2)
    
    plt.savefig("Images/epiexamples_er.png",format='png',dpi=250)
   