"""
The neural networks are defined in this file.
"""

from keras.layers import *
from keras.models import Model

def fully_convolutional(filters, kernels, activation='selu', positive_r=True, inn = 3):
    insize = 0 
    for i in range(len(kernels)):
        insize = insize + kernels[i]
    insize = int((insize-len(kernels)) + 250)
    inp = Input(shape=(insize, inn))
    x = Conv1D(filters[0], kernel_size=kernels[0], padding='valid', activation=activation,
               input_shape=(insize, inn,))(inp)  
    for f, k in zip(filters[1:-1], kernels[1:-1]):
        x = Conv1D(f, kernel_size=k, padding='valid', activation=activation)(x)         
    if positive_r:
        out = Conv1D_positive_r(x, kernel_size=kernels[-1])
    else:
        out = Conv1D(filters[-1], kernel_size=kernels[-1], padding='valid', activation='linear')(x)
    return Model(inp, out)

def Conv1D_positive_r(x, kernel_size):
    """index of r is hard-coded to 2!"""
    out1 = Conv1D(1, kernel_size=kernel_size, padding='valid', activation='linear')(x)
    out2 = Conv1D(1, kernel_size=kernel_size, padding='valid', activation='linear')(x)
    out3 = Conv1D(1, kernel_size=kernel_size, padding='valid', activation='relu')(x)
    return Concatenate()([out1, out2, out3])

def bias_loss(y_true, y_pred, ax):
    true_sum = K.mean(y_true, 1)[:, ax]
    pred_sum = K.mean(y_pred, 1)[:, ax]
    bias = K.mean(K.sqrt(K.square(true_sum - pred_sum)))
    return bias

def combined_bias_loss(axs, betas):
    if type(axs) is not list: axs = [axs]; betas = [betas]

    def combined_loss(y_true, y_pred):
        loss = K.mean(K.sqrt(K.mean(K.square(y_true - y_pred),axis=1)),axis=-1)
        for ax, beta in zip(axs, betas):
            loss += beta * bias_loss(y_true, y_pred, ax)
        return loss
    return combined_loss

 


