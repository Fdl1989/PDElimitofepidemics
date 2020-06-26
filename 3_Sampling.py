#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 29 15:19:36 2020

@author: Francesco Di Lauro
@mail: F.Di-Lauro@sussex.ac.uk
Copyright 2020 Francesco Di Lauro. All Rights Reserved.
See LICENSE file for details
"""

import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
import random
import multiprocessing as mp
#from igraph import GraphBase as graph

from fastgillespie import fast_Gillespie
#from odes import Mastereq,solutiontomaster,Mastereq_nodeaths,solutiontomaster_noabs
plt.rcParams['text.usetex'] = True
import matplotlib
matplotlib.use('agg')

'''
In this code we do some heuristics about the quasi steady state of different epidemics
on different graphs. In particular we will take Gllespie simulations for different
combinations of $\tau, \gamma, <k>$ and network and different times to which to
sample different realisation of the epidemics.

The distribution we get out of it will be compared to various numerical approximation.
In particular, initially we consider the average number of SI counts to solve the M.equation
and get the distribution out of it. Then we fit SI with the C,a,p function and do the same.
'''


def simulation_steadystatesampling(G,tau, gamma, tauf,N):
    '''
    This bit of code is used in parallelization. It runs epidemics on a simple networks and returns the last point sampled. Used to sample the steady state
    '''


    model = fast_Gillespie(G,i0 = 5,tau = tau, gamma= gamma,tauf=tauf,discretestep=100) # Setup the simulation with given parameters.
    model.run_sim() # Run the simulation.

    return model.I[::10]/float(N)


if __name__ =="__main__":

    '''
    sampling from m different epidemics at the same time
    '''
    np.random.seed(1)
    seed = 10213
    '''
    network and epidemic parameters come from the first of the params chosen for the scaling
    '''

if __name__=="__main__":

    ERgamma = [5, 4.5, 7]
    ERtau =[1, 1, 4]
    ERk = [8.0,10.0,7.0]
    taufv = [4,3,0.8]


    number_of_networkgen = 100

    number_of_epid = 250


    from datetime import datetime

    startTime = datetime.now()
    #Do the simulations
    seed_vector=np.array([j*(124)+23 for j in range(number_of_networkgen*number_of_epid)])

    for i in range(1):
        for N in [100,500,1000,5000]:
            k = ERk[i]
            gamma = ERgamma[i]
            tau = ERtau[i]
            tauf = taufv[i]
            R0_er = tau*(k*(N-2)/(N-1.0))/(tau+gamma)
            #print(R0_er)
            #print(tauf)
            networkchoice='E-R'
            name = "%s_tau=%.3f_gamma=%.3f_k=%d"%(networkchoice,tau,gamma,k)

            distribution = np.zeros( (10,number_of_networkgen*number_of_epid) )
            for graph in range(number_of_networkgen):
                seed = seed_vector[graph*number_of_epid]
                random.seed(seed)
                G= nx.fast_gnp_random_graph(int(N),k/float(N-1.0))

                pool = mp.Pool(mp.cpu_count())
                distribution_objects=[pool.apply_async(simulation_steadystatesampling ,args=(G,tau, gamma, tauf,N)) for epid in range(number_of_epid)]
                distribution[:,graph*number_of_epid:(graph+1)*(number_of_epid)] = np.array([r.get() for r in distribution_objects]).T
                pool.close()
                pool.join()
                del distribution_objects
                del pool
                del G
            np.save("txtfiles/%d_%s_distribution"%(N,name), distribution)

            print("Size %d done, time elapsed since last:"%N)
            print(datetime.now()-startTime)
            startTime = datetime.now()



    savedestination = 'txtfiles'

    Reggamma = [7,6,8]
    Regtau =[3.5,1,2.5]
    Regk = [8,9,7]
    taufv = [1.5,6,2]
    seed_vector=np.array([j*(124)+23 for j in range(number_of_networkgen*number_of_epid)])

    for i in range(2,3):
        for N in [100,500,1000,5000]:
            k = Regk[i]
            gamma = Reggamma[i]
            tau = Regtau[i]
            tauf = taufv[i]
            R0_er = tau*(k*(N-2)/(N-1.0))/(tau+gamma)
            #print(R0_er)
            #print(tauf)
            networkchoice='Reg'
            name = "%s_tau=%.3f_gamma=%.3f_k=%d"%(networkchoice,tau,gamma,k)

            distribution = np.zeros( (10,number_of_networkgen*number_of_epid) )
            for graph in range(number_of_networkgen):
                seed = seed_vector[graph*number_of_epid]
                random.seed(seed)
                G= nx.random_regular_graph(k,N)

                pool = mp.Pool(mp.cpu_count())
                distribution_objects=[pool.apply_async(simulation_steadystatesampling ,args=(G,tau, gamma, tauf,N)) for epid in range(number_of_epid)]
                distribution[:,graph*number_of_epid:(graph+1)*(number_of_epid)] = np.array([r.get() for r in distribution_objects]).T
                pool.close()
                pool.join()
                del distribution_objects
                del pool
                del G
            np.save("txtfiles/%d_%s_distribution"%(N,name), distribution)

            print("Size %d done, time elapsed since last:"%N)
            print(datetime.now()-startTime)
            startTime = datetime.now()



