#!/usr/bin/env python
# -*- coding: utf-8 -*-

# $Id: sw_exp.py 93 2013-08-26 09:54:24Z Heiner.Lange $
from parameters import *
from msw_model import sweq,initialize,noise_u
import assimilation as assim
import sys
import numpy as np
import random
import os.path
import math
import numpy.linalg as lin
import numpy.ma as ma
import pickle
from tqdm import tqdm

def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
 
data = {}
data['truth']={}
data['obs_pos']={}

for method in methods:
    data[method]={}   
    data[method]['analysis']={}    
    data[method]['first guess']={}
    if method == 'NN':
        from nn_assim import *        
if training:
    data['train']={}
rd = int(sys.argv[1])
r = [random.Random(10000+s*1000 +rd) for s in range(99)]

datadir = datadir+'nsub'+str(nsub)+'/nu'+str(nu)+'/'+str(k)+'/Data/'+str(rd)+'/'
if not os.path.exists(datadir):
    os.makedirs(datadir)
obs_position = np.ones((mf*nx))
ens = {}
    
#initialize
unoise_t=noise_u(r[0],1)
truth = initialize(unoise_t,1) #truth
    
unoise = noise_u(r[1],k)
ens_train = initialize(unoise,k) #ensemble
    
for method in methods:
    ens[method] = np.copy(ens_train)
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# start cycling
for ti in tqdm(range(cyc-1)): 
    
    # run the model to compute first guesses
    unoise_t=noise_u(r[0],1)
    unoise=noise_u(r[1],k)
    truth = sweq(unoise_t,1,truth) 
    data['truth'][ti] = truth
    for method in methods:
        ens[method] = sweq(unoise,k,ens[method])
        data[method]['first guess'][ti]= ens[method]

    # Generate synthetic observations from the truth, the model state and the observations from the model
    obs,obs_original =assim.genobs(r[2],r[3],np.copy(truth))         
    if radarint==1:
        obs_position = assim.radar(np.copy(truth),r[4])
    data['obs_pos'][ti] = obs_position
    
    # compute initial conditions
    for method in methods:
        if method == 'QPEns' and training:
            ens[method], data['train'][ti] = assim.assimilation(ens[method],obs_position,obs,method)         
        else: 
            ens[method] = assim.assimilation(ens[method],obs_position,obs,method)
         
        if method == 'NN' and ti >= 20:
            print('Start NN')
            ens['NN'] = nn_assim(ens['NN'],obs_position)
            print('Stop NN')
          
        data[method]['analysis'][ti]= ens[method]
        
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# end cycling

# saving data...
save_obj(data['truth'], f'{datadir}truth_data' )
save_obj(data['obs_pos'], f'{datadir}obs_pos' )
for method in methods:
    save_obj(data[method], f'{datadir}{method}_data' )
if training:
    save_obj(data['train'], f'{datadir}train_data' )

