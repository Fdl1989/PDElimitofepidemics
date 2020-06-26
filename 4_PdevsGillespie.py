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
from numschemesfp import FVM3
#from AnalyticalSolution import TPDFBMReflected, TPDFBMAbsorbed
from matplotlib import pyplot as plt

# Test with Dirichlet boundaries
#PDE = FokkerPlanck1D(N=1000,Nx=10000,Nt=200,leftBC="Dir", rightBC="Rob")


u = np.loadtxt('txtfiles/ER_0.txt')
distribution = np.load('txtfiles/1000_E-R_tau=1.000_gamma=5.000_k=8_distribution.npy')
SI = np.load('txtfiles/1000_E-R_tau=1.000_gamma=5.000_k=8_SI.npy')

xi =5e-3
N=1000
for i in range(len(distribution[0])):
    distribution[0][i]=5e-3  
#tau=0.752
ERgamma = [5, 4.5, 7]
ERtau =[1, 1, 4]
ERk = [8.0,10.0,7.0]
taufv = [4,3,0.8]

gamma=ERgamma[0]

tauf = taufv[0]


x = np.linspace(0,1,1001)
t = np.linspace(0,tauf,1000, endpoint=True)
    
def CAPmodel(x):
    return C*(((x)**p)*((1-x)**p)*(a*(x/2.0) +1))

def diffusion(x):
    N=1000
    return (1.0/N)*(CAPmodel(x) + gamma*x)
    
def drift(x):

    drift = CAPmodel(x) - gamma*x
    return drift

def splinediff(x, SI, gamma):
    f2 = interp1d(x,SI/N, kind='cubic')
    return (1/N)*(f2(x)+gamma*x)
def splinedrift(x, SI, gamma):
    f2 = interp1d(x,SI/N, kind='cubic')
    return f2(x) -gamma*x

C,a,p = u[2]


xi =5e-3
N=1000

difflux = diffusion(x)
driftflux = drift(x)
resvec = FVM3(x,t,difflux,driftflux,xi)
fig,ax = plt.subplots(2,5, figsize=(6,4))
ax = ax.ravel()    
fig.subplots_adjust(wspace=0.4,hspace=0.34)

splindiff = splinediff(x,SI,gamma) 
splindrif= splinedrift(x,SI,gamma)
resvecfvm3spline = FVM3(x,t,splindiff,splindrif,xi)

for i,sub in enumerate(ax):
    #print(time[i])
    #weights = np.ones_like(distribution[i])/float(len(distribution[i]))
    if i!=0:
       # binsize = int((np.max(distribution[i]) - np.min(distribution[i]))*N)-1
        binsize2 = int((np.max(distribution[i]) - np.min(distribution[i]))*N)-1

    else:
        binsize2 = 99

    #weights = np.ones_like(distribution[i])/float(len(distribution[i]))
    weights2 = np.ones_like(distribution[i])/float(len(distribution[i]))

   # sub.hist(distribution[i], bins=binsize,  weights=weights,color='b', alpha=0.7, label = 'Sde')
    sub.hist(distribution[i], density=True,color='blue', alpha=0.7, label = 'Gillespie')

    #sub.plot(PDE.x,res/1000.0, color='k', label='Chang-Cooper')
    sub.plot(x,resvec[i*100], color='k', label='FVM3 Cap')
    sub.plot(x,resvecfvm3spline[i*100], linestyle = '--', color='r', label='FVM3 Cap')

    #sub.hist(distributionsde[i], bins=binsizesde, density=True,stacked=True, color='orange')
    sub.set_xticks([0,0.5,1], ['$0$', '$0.5$', '$1$'])
    sub.set_xlim(0,1)
    if i !=0:
        sub.set_ylim(0, 1.2*np.max(resvec[i*100][1:]))
    else:
        sub.set_ylim(0, 1.2*np.max(resvec[i*100]))
    sub.set_title('$t=%.2f$'%t[i*100], size=10)

for i in range(len(ax)):
    for tick in ax[i].xaxis.get_major_ticks():
        tick.label.set_fontsize(9)
for i in range(len(ax)):
    for tick in ax[i].yaxis.get_major_ticks():
        tick.label.set_fontsize(9)      
ax[1].set_ylim(0,40)
ax[2].set_ylim(0,15)
ax[3].set_ylim(0,7)
ax[4].set_ylim(0,7)
ax[5].set_ylim(0,7)
ax[6].set_ylim(0,18)
ax[7].set_ylim(0,18)
ax[8].set_ylim(0,18)
ax[9].set_ylim(0,18)

    
plt.savefig("Images/Pde_ER_0.eps", format='eps')
    
    
    
    
    
import numpy as np
from numschemesfp import FVM3
#from AnalyticalSolution import TPDFBMReflected, TPDFBMAbsorbed
from matplotlib import pyplot as plt

# Test with Dirichlet boundaries
#PDE = FokkerPlanck1D(N=1000,Nx=10000,Nt=200,leftBC="Dir", rightBC="Rob")

def model_cap(u, k, N):
  return u[0]*(k**u[2]) * ((1-k)**u[2]) * (u[1]*(k-1/2.0)+1)
def diffusion(u,x,gamma):
    N=1000
    return (1.0/N)*(model_cap(u,x,len(x)-1) + gamma*x)
    
def drift(u,x,gamma):
    drift = model_cap(u,x,len(x)-1) - gamma*x
    return drift
from scipy.interpolate import interp1d





'''
Regular
'''

uv = np.loadtxt('txtfiles/Reg_0.txt')
distribution = np.load('txtfiles/1000_Reg_tau=2.500_gamma=8.000_k=7_distribution.npy')
counts = np.load('txtfiles/1000_Reg_tau=2.500_gamma=8.000_k=7_SI.npy')


xi =5e-3
N=1000
for i in range(len(distribution[0])):
    distribution[0][i]=5e-3  
#tau=0.752

Reggamma = [7,6,8]
Regtau =[3.5,1,2.5]
Regk = [8,9,7]
taufv = [1.5,6,2]        



gamma=Reggamma[2]

tauf = taufv[2]
tau = Regtau[2]

x = np.linspace(0,1,1001)
t = np.linspace(0,tauf,1000, endpoint=True)

SI = counts*Regtau[2]
'''
plt.plot(x,model_cap(u[2],x, SI.size-1), color='r',linestyle='--',alpha=0.7)
plt.plot(x,SI/N, color='b')
plt.plot(x,gamma*x)
'''



splindiff = splinediff(x,SI,gamma) 
splindrif= splinedrift(x,SI,gamma)
resvecfvm3spline = FVM3(x,t,splindiff,splindrif,xi)

u = uv[2]
difflux = diffusion(u,x,gamma)
driftflux = drift(u,x,gamma)
resvec = FVM3(x,t,difflux,driftflux,xi)

fig,ax = plt.subplots(2,5, figsize=(6,4))
ax = ax.ravel()    
fig.subplots_adjust(wspace=0.4,hspace=0.34)

for i,sub in enumerate(ax):
    #print(time[i])
    #weights = np.ones_like(distribution[i])/float(len(distribution[i]))
    sub.plot(x,resvec[i*100], color='k', label='FVM3 Cap')
    sub.plot(x,resvecfvm3spline[i*100], linestyle = '--',  color='red', label='FVM3 spline')

    if i!=0:
       # binsize = int((np.max(distribution[i]) - np.min(distribution[i]))*N)-1
        binsize = np.arange(np.min(distribution[i]), np.max(distribution[i])+1/N, 1/N )

    else:
        binsize = 99

    #weights = np.ones_like(distribution[i])/float(len(distribution[i]))
    weights = np.ones_like(distribution[i])/float(len(distribution[i]))

   # sub.hist(distribution[i], bins=binsize,  weights=weights,color='b', alpha=0.7, label = 'Sde')
    sub.hist(distribution[i],density=True,  color='blue', alpha=0.7, label = 'Gillespie')
    
  

    #sub.plot(PDE.x,res/1000.0, color='k', label='Chang-Cooper')

    #sub.hist(distributionsde[i], bins=binsizesde, density=True,stacked=True, color='orange')
    sub.set_xticks([0,0.5,1], ['$0$', '$0.5$', '$1$'])
    sub.set_xlim(0,1)

    sub.set_title('$t=%.2f$'%t[i*100], size=10)
for i in range(len(ax)):
    for tick in ax[i].xaxis.get_major_ticks():
        tick.label.set_fontsize(9)
for i in range(len(ax)):
    for tick in ax[i].yaxis.get_major_ticks():
        tick.label.set_fontsize(9)    

ax[1].set_ylim(0,40)
ax[2].set_ylim(0,15)
ax[3].set_ylim(0,7)
ax[4].set_ylim(0,7)
ax[5].set_ylim(0,7)
ax[6].set_ylim(0,18)
ax[7].set_ylim(0,18)
ax[8].set_ylim(0,18)
ax[9].set_ylim(0,18)

plt.savefig("Images/Pde_Reg_2.eps", format='eps')


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    