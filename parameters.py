# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 15:54:40 2018

@author: Yvonne.Ruckstuhl
"""

import numpy as np
import math

datadir='/home/y/Yvonne.Ruckstuhl/python_code/'
SAVEDIR = '/home/y/Yvonne.Ruckstuhl/python_code/test/'
methods = ['EnKF','NN','QPEns']  # options are ['EnKF','QF',QPEns','NN']
rad= 0
training = False
k = 10    # number of ensemble members
nx = 250  # number of gridpoints
cyc = 3#201  # number of 
nsub = 120   # assimilation window in number of modelsteps
nsteps = cyc*nsub
wind = 1   # if u is observed
height = 1  #if h is observed
rain = 1    # if r is observed
mf = 3
# observations errors
sigu = 0.001  
sigh = 0.01
sigr = 1.5
mur = -8.0
dbz = 0.005   
irad = 4 #localization: 0 for no localization
radarint = 1

nu = 10.
rho = 1.0

#model constants
alpha=0.008
beta = 50.0
phic = 899.9
ku = 20000.0
kr = 200.0
h_cloud = 90.005
h_rain = 90.1
gamma = 1.0
dx = float(500)     
dt = float(5) 

def define_field(wind,height,rain):
    fields = np.array([0,1,2])   
    if wind ==  0:
        fields = fields[1:3]
        if height == 0:
            fields = fields[1:2]
            if rain == 0:
                print('Warning, no variables are observed!')
        else:
            if rain == 0:
                fields = fields[0:1]
    else:
        
        if height == 0:
            fields = fields[[0,2]]
            if rain == 0:
                fields = fields[0:1]
        else:
            if rain == 0:
                fields = fields[[0,1]]
    return fields
def LocMat(nx,c):
    LocMat = np.zeros((nx,nx))
    
    for l in range(2*c):
        b = float(c)
        z = l/b
        for j in range(0,nx):
            if l < c:
                if l+j < nx:
                    #pdb.set_trace()
                    LocMat[j+l,j] = -0.25*math.pow(z,5)+0.5*math.pow(z,4)+(5.0/8.0)*math.pow(z,3)-(5.0/3.0)*math.pow(z,2)+1.0
                    LocMat[j,j+l] = LocMat[j+l,j]
                else: 
                    #pdb.set_trace()
                    LocMat[l-nx+j,j] = -0.25*math.pow(z,5)+0.5*math.pow(z,4)+(5.0/8.0)*math.pow(z,3)-(5.0/3.0)*math.pow(z,2)+1.0
                    LocMat[j,l-nx+j] = LocMat[l-nx+j,j]
            else:
                if l+j < nx:
                    #pdb.set_trace()
                    LocMat[j+l,j] = (1.0/12.0)*math.pow(z,5)-0.5*math.pow(z,4)+(5.0/8.0)*math.pow(z,3)+(5.0/3.0)*math.pow(z,2) -5.0*z+4.0-(2.0/3.0)*(b/l)
                    LocMat[j,j+l] = LocMat[j+l,j]
                else: 
                    #pdb.set_trace()
                    LocMat[l-nx+j,j] = (1.0/12.0)*math.pow(z,5)-0.5*math.pow(z,4)+(5.0/8.0)*math.pow(z,3)+(5.0/3.0)*math.pow(z,2) -5.0*z+4.0-(2.0/3.0)*(b/l)
                    LocMat[j,l-nx+j] = LocMat[l-nx+j,j]
    return LocMat
def noiseprep(): # Preparation of the noise field
    mu=float((nx+1)/2)    # Center of the noise
    d=np.array(range(nx+1))  
    sig=float(4)   # Sigma (half width) of the noise field
    amp=float(0.002)  # Amplitude of the added noise field (in m/s)  normal 0.005
    z=(1/(sig*np.sqrt(2.*np.pi)))*np.exp(-0.5*((d-mu)/sig)**2)  
    zsum=z[1:nx+1]-z[0:nx]
    zsum=amp*zsum/max(zsum)  # Normalize the noise to 1 and multiply with the amplitude
    return zsum
fields = define_field(wind,height,rain)
of=len(fields)   # number of different observation fields/fields to assimilate

obs_position = np.tile(np.arange(nx),(mf,1)) 
obserr=np.array([sigu*sigu,sigh*sigh,(np.exp(sigr*sigr)-1)*np.exp(2*mur+sigr*sigr) ])
gaussobs=np.array([sigu,sigh,sigr]) #  error which is added onto the observations
     
mp = 0 
C = LocMat(nx,int(irad))
C = np.tile(C,mf)
C = np.tile(C.transpose(),mf).transpose()

zsum = noiseprep()
Rdiag = np.zeros((mf*nx))
Rdiag[0:nx]=obserr[0]
Rdiag[nx:2*nx]=obserr[1]
Rdiag[2*nx:3*nx]=obserr[2]
