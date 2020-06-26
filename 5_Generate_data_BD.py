#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Created on Sat Feb 29 15:19:36 2020

@author: Francesco Di Lauro
@mail: F.Di-Lauro@sussex.ac.uk
Copyright 2020 Francesco Di Lauro. All Rights Reserved.
See LICENSE file for details
'''

# Compare BD process with continuous approximations

import random
random.seed(20121989)

import numpy as np
#from BD_functions import *
from matplotlib import pyplot as plt
#from scipy.linalg import expm

def CAPmodel(k,u):
    return u[0]*((k**u[2])*((N-k)**u[2])*(u[1]*(k - 0.5*N) +N))

u = np.loadtxt("E-R_tau=1.000_gamma=4.500_k=10_cap.txt", unpack=True)

N = 1000 # Population size
T = 5 # Final time
t = 0 # Initial time
tau = 2e0
gamma = 4.5
n_I = 5 # Initial infectious nodes
k = np.arange(start=0, stop=N)

# Fully connected graph functions
def a_k(k, tau):
    return tau / N * k * (N - k)

def c_k(k, ganma):
    return gamma * k


# Import the C,a,p values
#u_test = np.loadtxt("u_test_N_SI_all.csv")


# Gillepsie simulation of BD
t=0
SIS_data = []
SIS_data.append((t, n_I))
I = n_I
#ak = a_k(k,tau)
ak = CAPmodel(k,u)

while t < T:
    
    if I == 0:
        break

    w1 = ak[I]
    w2 = gamma*I
    W = w1 + w2
    
    dt = -np.log(np.random.random_sample()) / W
    t = t + dt

    if np.random.random_sample() < w1 / W:
        I = I + 1
    else:
        I = I - 1

    SIS_data.append((t, I))
SIS_data.append((T, I))
sim_BD = np.asarray(SIS_data)

ndata = 30
delta_t = np.linspace(start=0, stop=T, num=ndata)
Data = np.zeros((ndata, 2))

# Extract values of I(t) at delta_t times
for (i, t) in enumerate(delta_t):
    idx = np.where(sim_BD[:,0]-t>=0)[0][0]
    Data[i, 1] = sim_BD[idx, 1]
    Data[i,0] = t
    
fig= plt.figure(figsize=(3,2.6))
plt.subplots_adjust(left=0.2, bottom=0.2, right=0.94, top=0.95, wspace=0, hspace=0)
ax0 = fig.add_subplot(111)
ax0.step(sim_BD[:, 0], sim_BD[:,1]/N, label=r"BD process")
ax0.set_xlim(0, T)
ax0.set_xlabel(r"Time", size=9)
ax0.set_ylim(0, 0.6)
ax0.set_xticklabels([r"$0$",r"$1$",r"$2$",r"$3$",r"$4$",r"$5$"], size=7)

ax0.set_yticklabels([r"$0.0$",r"$0.1$",r"$0.2$",r"$0.3$",r"$0.4$",r"$0.5$",r"$0.6$"], size=7)

ax0.set_ylabel(r"Infected", size=9)
ax0.scatter(Data[:,0], Data[:, 1]/N, color="red", label=r'Data')
ax0.legend(loc="best")

plt.savefig("Data_ER.png",format='png', dpi=400)
np.savetxt("Dataset.out", Data)




