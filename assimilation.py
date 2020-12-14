# -*- coding:< utf-8 -*-
"""
Created on Mon Aug  1 11:27:47 2016

@author: Yvonne.Ruckstuhl
"""

#!/usr/bin/env python
# -*- coding: utf-8 -*-
from parameters import *
from scipy import *
import numpy as np
import cvxopt
from cvxopt import matrix, solvers
import pdb
import random
import math
   
def assimilation(model,obspos,obs,method):
    '''
    input
    model: background ensemble of size 3*nx x k
    obspos: array of observation positions (0 means observed, 1 means not observed)
    obs: perturbed observations
    method: data assimilation algorithm (EnKF, QPEns or NN)
    output
    model: analysis ensemble of size 3*nx x k
    an_train (only when training=True): unconstrained analysis ensemble of size 3*nx x k
    '''
    obspos = np.arange(nx*mf)[obspos==0] # position of observations
    nobs=len(obspos) # number of observation
    R = Rdiag[obspos] # observation error covariance matrix   
    P = covP(model) # background error covariance matrix
    if irad > 0:
        P = C*P #localisation
    K = np.dot(P[:,obspos],np.linalg.inv(P[np.ix_(obspos,obspos)]+np.diag(R))) # Kalman Gain
    D = obs[obspos,:]-model[obspos,:] # innovation
    
    if method == "EnKF" or method == "NN":
        model = model +np.dot(K,D)
    
    if method == 'QPEns':
        if training:          
            an_train = model +np.dot(K,D)
        La, Da , V= np.linalg.svd(P, full_matrices=True)
        La =np.dot(La,np.diag(np.sqrt(Da))) # square root of P
        Ya = La[obspos,:] # square root of P in obs space
        ToolR = np.divide(Ya, R[:,None]) 
        G = np.identity(nx*mf)+np.dot(Ya.T,ToolR ) # Hessian of costfunction
        jr = np.arange(nx*(mf-1),nx*mf)  # Used to set constraints r >= 0  
        jh = np.arange(nx*(mf-2),nx*(mf-1))  
        A = np.dot(np.ones(nx) ,np.asmatrix(La[jh,:])) #mass conservation constraint of h 
        sol = np.zeros((nx*mf,k))       
        for i in range(0,k):
            c = np.dot(-ToolR.T, D[:,i])
            solution = solvers.qp(matrix(G),matrix(c),matrix(-La[jr,:]),matrix(model[jr,i]),matrix(A),matrix(np.zeros((1,1)))) 
            sol[:,i] = np.asarray(solution['x']).reshape(-1) # solution of minimisation problem
        model =  model+np.dot(La,sol)
 
    if  method == "QPEns" and training:
        return model, an_train
    else:
        return model

def covP(x):
    '''
    input
    x: background ensemble of size 3*nx x k
    output
    error covariance matrix of size 3*nx x 3*nx
    '''
    rho = 1.0
    x = np.sqrt(rho/(k-1.))*(x - np.mean(x,axis=1).reshape(nx*mf,1))
    return np.dot(x,x.T)


def genobs(r3,r8, truth):
    '''
    input
    r3: seed number for generation of observations
    r8: seed number for generation of the observation perturbations
    truth: array of size 3*nx representing the true state of the atmosphere
    
    output
    obs: observations with Gaussian error for u and h and lognormal error for r of size 3*nx*k
    obs_original: unperturbed observations of size 3*nx
    '''
    obserr=np.zeros((mf*nx,1))
    for i in range(0,nx):
        obserr[i,0]=r3.gauss(mu=0,sigma=gaussobs[0])
        obserr[i+nx,0]=r3.gauss(mu=0,sigma=gaussobs[1])
        obserr[i+2*nx,0]=r3.lognormvariate(mu=mur,sigma=gaussobs[2])
    obs_original=truth+obserr      
    obserr=np.zeros((mf*nx,k))
    for j in range(k):
        for i in range(0,nx):
        
            obserr[i,j]=r8.gauss(mu=0,sigma=gaussobs[0])
            obserr[i+nx,j]=r8.gauss(mu=0,sigma=gaussobs[1])
            obserr[i+2*nx,j]=r8.lognormvariate(mu=mur,sigma=gaussobs[2])
     
    obs = obserr + np.tile(obs_original.reshape(mf*nx,1),(k))
     
      
    return obs, obs_original

def radar(truth,r1):
 
    obspos = np.ones((mf*nx))
    index_rain = np.arange(nx)[truth[2*nx:3*nx,0]>dbz]
    index_norain = np.arange(nx)[truth[2*nx:3*nx,0]<=dbz]
  
    if nu > 0:
        nwind = int(nu/100.*len(index_norain))
        index_wind=r1.sample(list(index_norain),nwind)
        index_wind=sorted(np.concatenate((index_wind,index_rain)))
    else:
        index_wind = index_rain
    obspos[2*nx+index_rain] = 0 
    obspos[nx+index_rain] = 0 
    obspos[index_wind] = 0 
    return obspos
 
