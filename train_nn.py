"""
Script to train a neural network
"""
from fire import Fire   
import numpy as np
import pickle
import keras
import tensorflow as tf
from keras.optimizers import Adam
from keras.layers import *
import os
from networks import *
import numpy.ma as ma
from keras.callbacks import ModelCheckpoint
from parameters import *

def limit_mem():
    """Necessary to stop TF from using all of the GPU"""
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    keras.backend.tensorflow_backend.set_session(tf.Session(config=config))
limit_mem()

def load(fn):
    """Loads the pickle files that the nsw model outputs"""
    with open(fn, 'rb') as f:
        return pickle.load(f, encoding='latin1')   

def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        
def make_256(x,newsize):
    """Rolls the array to x=newsize"""
    a = np.empty((x.shape[0], newsize, x.shape[2]), 'float32')
    dsize = int((newsize-nx)/2)
    a[:, dsize:-dsize] = x
    a[:, :dsize] = x[:, -dsize:]
    a[:, -dsize:] = x[:, :dsize]
    return a

def get_data(path, exps=50, time_range = [0],ens=10,split=0.5,newsize=256):
    """Loads experiments and does some reshaping"""
    
    n = len(exps)*ens*len(time_range)
    X = np.zeros((n,nx,4))
    Y = np.zeros((n,nx,3))
   
    index = []
    for iexp, exp in enumerate(exps):
       
        Y_temp = load(f'{path}{exp}/QPEns_data.pkl')['analysis']
        X_temp = {}      
        X_temp['an'] = load(f'{path}{exp}/train_data.pkl')  
        X_temp['obspos'] = load(f'{path}{exp}/obs_pos.pkl')
        for ti,t in enumerate(time_range):        
            Y[iexp*ens*len(time_range)+ti*ens:iexp*ens*len(time_range)+(ti+1)*ens,...] = Y_temp[t][...,:ens].reshape(3,nx,-1).T
            X[iexp*ens*len(time_range)+ti*ens:iexp*ens*len(time_range)+(ti+1)*ens,:,:3] = X_temp['an'][t].reshape(3,nx,-1).T    
            X[iexp*ens*len(time_range)+ti*ens:iexp*ens*len(time_range)+(ti+1)*ens,:,3] = np.tile(X_temp['obspos'][t][2*nx:],(ens,1))

    mean_temp = np.average(np.average(X[...,0:3],axis = 0),axis=0)
    mean = [mean_temp[0],mean_temp[1],mean_temp[2]]
    mean[2] = 0
    
    std_temp = np.average(np.var(X[...,0:3],axis = 0),axis=0)
    std = [std_temp[0]**0.5, std_temp[1]**0.5,std_temp[2]**0.5]
    for i in range(3):
        X[...,i] = (X[...,i]-mean[i])/std[i]
        Y[...,i] = (Y[...,i]-mean[i])/std[i]
   
    X = make_256(X,newsize)
    
    split2 = int(X.shape[0]*split)
    X_train = X[:split2,...]
    X_valid = X[split2:,...]
    
    Y_train = Y[:split2,...]
    Y_valid = Y[split2:,...]
   
    return X_train, X_valid, Y_train, Y_valid, mean, std  

    
def scale(x, m, s): return (x - m) / s
def unscale(x, m, s): return x * s + m


def save(mean,std,name): 
    np.save(name+'mean.npy',mean)
    np.save(name+'std.npy',std)


    
def main(datadir, keras_save_fn, name, exps, nn_args, split=0.5, time_range=[20,499], ens=10,
         loss='mse', lr=1e-3, epochs=100, bs=96, bias_loss_axs=1, bias_loss_betas=1):
   
    if not os.path.exists(keras_save_fn):
        os.makedirs(keras_save_fn) 
    if not os.path.exists(keras_save_fn+name):
        os.makedirs(keras_save_fn+name)   
    
    nn_args['inn'] = 4
    newsize = 0
    for i in range(len(nn_args['kernels'])):
        newsize = newsize + nn_args['kernels'][i]
    newsize = int((newsize-len(nn_args['kernels'])) + nx)    
    if type(exps) is tuple:
        exps = list(range(exps[0], exps[1]))
    time_range = range(time_range[0],time_range[1])  
    n_exps = len(exps)
     
    X_train,X_valid, Y_train,Y_valid, mean , std = get_data(datadir, exps, time_range,ens,split,newsize)
    save(mean,std,f'{keras_save_fn}{name}/')
    
    model = fully_convolutional(**nn_args)
    loss = combined_bias_loss(bias_loss_axs, bias_loss_betas) 
    model.compile(loss=loss,optimizer=Adam(lr)) 
    filepath=keras_save_fn+name+'/weights.best.hdf5'
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False, save_weights_only=False, mode='auto')
    callbacks_list = [checkpoint]
    histo = model.fit(X_train, Y_train, batch_size=bs, epochs=epochs, validation_data=(X_valid, Y_valid), shuffle=True,callbacks=callbacks_list)
    hist = histo.history
    save_obj(hist, f'{keras_save_fn}{name}/hist' )
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False, save_weights_only=False, mode='auto')
    callbacks_list = [checkpoint]
    model.save(f'{keras_save_fn}{name}/model.h5')
           
if __name__ == '__main__':
    Fire(main)
