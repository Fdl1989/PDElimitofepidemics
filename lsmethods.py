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
from pyswarm import pso

    
def model_cpq(u, k, N):
  return u[0] * (k**u[1]) * ((N-k)**u[2])

def model_cpq_scaled(u, k, N):
  return u[0]*N**(u[1]+u[2]-1)*(k**u[1]) * ((1-k)**u[2])


def model_cap(u, k, N):
  return u[0] * (k**u[2]) * ((N-k)**u[2]) * (u[1]*(k-N/2.0)+N)
def LS2_cap(u, I, SI, N):
    # Compute continuous log_likelihood ak+1
    ak = model_cap(u, I, N)
    ak_hat = SI
    temp = ak-ak_hat

    dist = np.dot(temp, temp)
    return dist
def LS_cap(I,SI, N, swarmsize=2500, maxiter=3000):
    # Compute MLE estimator from Uk, Tk for model 2
    swarmsize = int(2*N)
    def temp(v):
        return LS2_cap(v,I,SI,N)
    LB = np.array([1/(N**2), -2, 0.5])
    UB = np.array([1/N, 2, 1.5])
    # Constrained optimization
    opt = pso(temp, lb=LB, ub=UB, swarmsize=swarmsize, maxiter=maxiter)

    return opt[0]


def LS2(u, I, SI, N):
    # Compute continuous log_likelihood ak+1
    ak = model_cpq(u, I, N)
    ak_hat = SI
    temp = ak-ak_hat

    dist = np.dot(temp, temp)
    return dist
def LS_cpq(I,SI, N, swarmsize=2500, maxiter=2500):
    # Compute MLE estimator from Uk, Tk for model 2

    def temp(v):
        return LS2(v,I,SI,N)
    LB = np.array([1.0/(N**2), 0.1, 0.1])
    UB = np.array([1, 2, 2])
    # Constrained optimization
    opt = pso(temp, lb=LB, ub=UB, swarmsize=swarmsize, maxiter=maxiter)

    return opt[0]

def LS2_weight(u, I, SI, weight, N):
    # Compute continuous log_likelihood ak+1
    ak = model_cpq(u, I, N)
    ak_hat = SI
    temp = ak-ak_hat
    dist = np.sum(weight*temp*temp)
    return dist
def LS_cap_weight(I,SI, tk, N, swarmsize=2500, maxiter=2500):
    # Compute MLE estimator from Uk, Tk for model 2
    weight = tk/np.sum(tk)
    print(weight)
    def temp(v):
        return LS2_weight(v,I,SI,weight,N)
    LB = np.array([1.0/(N**3), -2, -2])
    UB = np.array([5.0/N, 2, 2])
    # Constrained optimization
    opt = pso(temp, lb=LB, ub=UB, swarmsize=swarmsize, maxiter=maxiter)

    return opt[0]