#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 1d shallow water model with reversed buoyancy for convection

from parameters import *
from numpy import *
import numpy as np
import sys
import random
import os.path

def initialize(unoise,k):
    """
    input
    unoise: wind perturbation of size nsub x nx x k
    output
    ens: initial climatological background ensemble of size 3*nx x k
    """
    u = np.zeros((nx,k))+10.
    h = np.zeros((nx,k))+90.
    r = np.zeros((nx,k))
    ens=np.concatenate((u,h,r),axis=0)
    for i in range(int(1000/nsub)):
        ens = sweq(unoise,k,ens)
    return ens
    
def sweq(unoise,k,temp):
    """
    input
    unoise: wind perturbation of size nsub x nx x k
    temp: initial conditions of size 3*nx x k 
    output
    ens: background of size 3*nx x k
    """
    dx = float(500)     # resolution,standard 0.001
    dt = float(1)       # timestep, standard 0.0001    
    g=10   # Gravitational constant
    kh = ku            # diffusion coefficient for h
    phi=np.zeros((1,nx+2,k))  # potential  
    beta_new=np.zeros((nx+2,k))
    u = np.zeros((3,nx+2,k))
    r = np.zeros((3,nx+2,k))
    h = np.zeros((3,nx+2,k))
    
    u[2,1:nx+1,:],u[1,1:nx+1,:],u[0,1:nx+1,:]=temp[0:nx,:],temp[0:nx,:],temp[0:nx,:]
    h[2,1:nx+1,:],h[1,1:nx+1,:],h[0,1:nx+1,:]=temp[nx:2*nx,:],temp[nx:2*nx,:],temp[nx:2*nx,:]
    r[2,1:nx+1,:],r[1,1:nx+1,:],r[0,1:nx+1,:]=temp[2*nx:3*nx,:],temp[2*nx:3*nx,:],temp[2*nx:3*nx,:]
    r = np.where(r<0.,0.,r)
    
    u[2,0,:],u[1,0,:],u[0,0,:]=u[0,nx,:],u[0,nx,:],u[0,nx,:]
    u[2,nx+1,:],u[1,nx+1,:],u[0,nx+1,:]=u[0,1,:],u[0,1,:],u[0,1,:]
    r[2,0,:],r[1,0,:],r[0,0,:]=r[0,nx,:],r[0,nx,:],r[0,nx,:]
    r[2,nx+1,:],r[1,nx+1,:],r[0,nx+1,:]=r[0,1,:],r[0,1,:],r[0,1,:]
    h[2,0,:],h[1,0,:],h[0,0,:]=h[0,nx,:],h[0,nx,:],h[0,nx,:]
    h[2,nx+1,:],h[1,nx+1,:],h[0,nx+1,:]=h[0,1,:],h[0,1,:],h[0,1,:]
    
   
    for it in range(nsub):   # Loop over the given model timesteps
     
      u[1,1:nx+1,:]=u[1,1:nx+1,:]+unoise[it,:,:]

#     transform height into potential phi:
      phi[0,1:nx+1,:] = np.where( h[1,1:nx+1,:] > h_cloud , phic, g*h[1,1:nx+1,:] )  # Replace height
      phi[0,0,:]   = phi[0,nx,:]
      phi[0,nx+1,:] = phi[0,1,:]
      phi[0,:,:]=phi[0,:,:]+gamma*r[1,:,:]  # rain influence is added onto phi

      u[2,1:nx+1,:] = u[0,1:nx+1,:] - (dt/(2*dx))*(u[1,2:nx+2,:]**2 - u[1,0:nx,:]**2) - (2*dt/dx)*(phi[0,1:nx+1,:]-phi[0,0:nx,:])  + (ku/(4*dx*dx))*(u[0,2:nx+2,:] - 2*u[0,1:nx+1,:] + u[0,0:nx,:])*dt
      h[2,1:nx+1,:] = h[0,1:nx+1,:] - (dt/dx)*(u[1,2:nx+2,:]*(h[1,1:nx+1,:]+h[1,2:nx+2,:]) - u[1,1:nx+1,:]*(h[1,0:nx,:]+h[1,1:nx+1,:])) + (kh/(4*dx*dx))*(h[0,2:nx+2,:] - 2*h[0,1:nx+1,:] + h[0,0:nx,:])*dt
      mask = np.logical_and(h[1,1:nx+1,:] > h_rain, u[1,2:nx+2,:]-u[1,1:nx+1,:] < 0)
      beta_new[1:nx+1,:] = np.where( mask, beta , 0 )
      r[2,1:nx+1,:] = r[0,1:nx+1,:] - alpha*dt*2*r[1,1:nx+1,:]-2*beta_new[1:nx+1,:]*(dt/dx)*(u[1,2:nx+2,:]-u[1,1:nx+1,:]) + (kr/(4*dx*dx))*(r[0,2:nx+2,:] - 2*r[0,1:nx+1,:] + r[0,0:nx,:])*dt   # No Advection

    # periodic boundaries

      u[2,0,:]   = u[2,nx,:]
      u[2,nx+1,:] = u[2,1,:]
      h[2,0,:]   = h[2,nx,:]
      h[2,nx+1,:] = h[2,1,:]
      r[2,0,:]   = r[2,nx,:]
      r[2,nx+1,:] = r[2,1,:]
      r[2,:,:]=np.where(r[2,:,:]<0.,0.,r[2,:,:])

      d = .1*.5*(u[2,:,:] - 2.*u[1,:,:] + u[0,:,:])
      u[0,:,:] = u[1,:,:] + 0.53*d
      u[1,:,:] = u[2,:,:] - 0.47*d 
      d = .1*.5*(h[2,:,:] - 2.*h[1,:,:] + h[0,:,:])
      h[0,:,:] = h[1,:,:] + 0.53*d
      h[1,:,:] = h[2,:,:] - 0.47*d
      d = .1*.5*(r[2,:,:] - 2.*r[1,:,:] + r[0,:,:])
      r[0,:,:] = r[1,:,:]+ 0.53*d
      r[1,:,:] = r[2,:,:]- 0.47*d
      
      ens=np.concatenate((u[2,1:nx+1,:],h[2,1:nx+1,:],r[2,1:nx+1,:]),axis=0)
     
    return ens

def noise_u(r4,k):    
    unoise= np.zeros((nsub,2*nx,k))   
    ngauss = 1#r3.poisson(1,1)  # Number of perturbations added  (x,y) x number of gridpoints
    for t in range(nsub):
        for i in range(ngauss):
            for j in range(k):
                pos= r4.randint(0,nx-1)  # the random position at which the noise center is
                unoise[t,pos:pos+nx,j]=unoise[t,pos:pos+nx,j]+zsum

    
    return   unoise[:,0:nx,:]+unoise[:,nx:nx+nx,:] 
        

    
