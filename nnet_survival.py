#nnet_survival_examples.py
#Functions used in neural network survival model.
#Author: Michael Gensheimer, Stanford University, michael.gensheimer@gmail.com
#Tested in Python version 2.7.12, with numpy version 1.13.3, TensorFlow version 1.1.0, Keras version 2.0.4

from __future__ import print_function
import numpy as np
import keras.backend as K

halflife=365.25*2
breaks=-np.log(1-np.arange(0.0,0.96,0.05))*halflife/np.log(2) #this defines the time intervals for the discrete-time survival model
n_intervals=len(breaks)-1
timegap = breaks[1:] - breaks[:-1]

def surv_likelihood(y_true, y_pred):
  """Custom Keras loss function for neural network survival model.
  Arguments
      y_true: Tensor.
        First half of the values is 1 if individual survived that interval, 0 if not.
        Second half of the values is for individuals who failed, and is 1 for time interval during which failure occured, 0 for other intervals.
        See make_surv_array function.
      y_pred: Tensor, predicted survival probability (1-hazard probability) for each time interval.
  Returns
      Vector of losses for this minibatch.
  """
  cens_uncens = 1. + y_true[:,0:n_intervals] * (y_pred-1.) #component for all individuals
  uncens = 1. - y_true[:,n_intervals:2*n_intervals] * y_pred #component for only uncensored individuals
  return K.sum(-K.log(K.clip(K.concatenate((cens_uncens,uncens)),K.epsilon(),None)),axis=-1) #return -log likelihood

def surv_likelihood_rnn(y_true, y_pred):
  """Custom Keras loss function for neural network survival model. Used for recurrent neural networks with time-distributed output.
       This function is very similar to surv_likelihood but deals with the extra dimension of y_true and y_pred that exists because of the time-distributed output.
  """
  cens_uncens = 1. + y_true[0,:,0:n_intervals] * (y_pred-1.) #component for all patients
  uncens = 1. - y_true[0,:,n_intervals:2*n_intervals] * y_pred #component for only uncensored patients
  return K.sum(-K.log(K.clip(K.concatenate((cens_uncens,uncens)),K.epsilon(),None)),axis=-1) #return -log likelihood
 
def make_surv_array(t,f):
  """Transforms censored survival data into vector format that can be used in Keras.
    Arguments
        t: Array of failure/censoring times.
        f: Censoring indicator. 1 if failed, 0 if censored.
    Returns
        Two-dimensional array of survival data, dimensions are number of individuals X number of time intervals*2
  """
  n_samples=t.shape[0]
  breaks_midpoint = breaks[:-1] + 0.5*timegap
  y_train = np.zeros((n_samples,n_intervals*2))
  for i in range(n_samples):
    if f[i]: #if failed (not censored)
      y_train[i,0:n_intervals] = 1.0*(t[i]>=breaks[1:]) #give credit for surviving each time interval where failure time >= upper limit
      if t[i]<breaks[-1]: #if failure time is greater than end of last time interval, no time interval will have failure marked
        y_train[i,n_intervals+np.where(t[i]<breaks[1:])[0][0]]=1 #mark failure at first bin where survival time < upper break-point
    else: #if censored
      y_train[i,0:n_intervals] = 1.0*(t[i]>=breaks_midpoint) #if censored and lived more than half-way through interval, give credit for surviving the interval.
  return y_train
