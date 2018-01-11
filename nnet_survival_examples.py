#nnet_survival_examples.py
#Examples of neural network survival model with simulated data
#Author: Michael Gensheimer, Stanford University, michael.gensheimer@gmail.com
#Tested in Python version 2.7.12, with numpy version 1.13.3, TensorFlow version 1.1.0, Keras version 2.0.4

from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing import sequence
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, Activation, LSTM, GRU, Embedding, Concatenate, Conv1D, GlobalMaxPooling1D, MaxPooling1D, GlobalAveragePooling1D, BatchNormalization, TimeDistributed
from keras import optimizers, layers, regularizers
import keras.backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model
import math
from lifelines import KaplanMeierFitter
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index

import nnet_survival

#all pts with same exponential survival distribution, no censoring
halflife1 = 365.25
n_samples=1000
np.random.seed(seed=0)
t = np.random.exponential(scale=1 / (np.log(2)/halflife1), size=n_samples)
f=np.ones(n_samples) #all patients failed (none censored)
y_train=nnet_survival.make_surv_array(t,f)
x_train = np.zeros(n_samples)

model = Sequential()
model.add(Dense(nnet_survival.n_intervals,input_dim=1))
model.add(Activation('sigmoid'))
model.compile(loss=nnet_survival.surv_likelihood, optimizer=optimizers.RMSprop())
early_stopping = EarlyStopping(monitor='loss', patience=2)
model.fit(x_train, y_train, batch_size=256, epochs=100000, callbacks=[early_stopping])
y_pred=model.predict_proba(x_train,verbose=0)

kmf = KaplanMeierFitter()
kmf.fit(t, event_observed=f)
plt.plot(nnet_survival.breaks,np.concatenate(([1],np.cumprod(y_pred[0,:]))),'bo-')
plt.plot(kmf.survival_function_.index.values, kmf.survival_function_.KM_estimate,color='k')
plt.xlabel('Follow-up time (days)')
plt.ylabel('Proportion surviving')
plt.title('All patients from same survival distribution, no censoring. Actual=black, predicted=blue.')
plt.show()

#all pts with same exponential survival distribution, some patients censored
halflife1 = 365.25*2
n_samples=1000
np.random.seed(seed=0)
time = np.random.exponential(scale=1 / (np.log(2)/halflife1), size=n_samples)
censtime = np.random.exponential(scale=1 / (np.log(2)/(1.0*halflife1)), size=n_samples)
f = time<censtime
t = time
t[~f] = censtime[~f]
y_train=nnet_survival.make_surv_array(t,f)
x_train = np.zeros(n_samples)

model = Sequential()
model.add(Dense(nnet_survival.n_intervals,input_dim=1))
model.add(Activation('sigmoid'))
model.compile(loss=nnet_survival.surv_likelihood, optimizer=optimizers.RMSprop())
early_stopping = EarlyStopping(monitor='loss', patience=2)
model.fit(x_train, y_train, batch_size=256, epochs=10000, callbacks=[early_stopping])
y_pred=model.predict_proba(x_train,verbose=0)

kmf = KaplanMeierFitter()
kmf.fit(t, event_observed=f)
plt.plot(nnet_survival.breaks,np.concatenate(([1],np.cumprod(y_pred[0,:]))),'bo-')
plt.plot(kmf.survival_function_.index.values, kmf.survival_function_.KM_estimate,color='k')
plt.xlabel('Follow-up time (days)')
plt.ylabel('Proportion surviving')
plt.title('All patients from same survival distribution, some censored. Actual=black, predicted=blue.')
plt.show()

#some patients censored, one predictor variable
halflife1 = 200
halflife2 = 400
halflife_cens = 400
n_samples=5000
np.random.seed(seed=0)
t1 = np.random.exponential(scale=1 / (np.log(2)/halflife1), size=int(n_samples/2))
t2 = np.random.exponential(scale=1 / (np.log(2)/halflife2), size=int(n_samples/2))
t=np.concatenate((t1, t2))
censtime = np.random.exponential(scale=1 / (np.log(2)/(halflife_cens)), size=n_samples)
f = t<censtime
t[~f] = censtime[~f]

y_train=nnet_survival.make_surv_array(t,f)
x_train = np.zeros(n_samples)
x_train[int(n_samples/2):]=1

model = Sequential()
model.add(Dense(nnet_survival.n_intervals,input_dim=1))
model.add(Activation('sigmoid'))
model.compile(loss=nnet_survival.surv_likelihood, optimizer=optimizers.RMSprop())
early_stopping = EarlyStopping(monitor='loss', patience=2)
model.fit(x_train, y_train, batch_size=256, epochs=1000, callbacks=[early_stopping])
y_pred=model.predict_proba(x_train,verbose=0)

kmf = KaplanMeierFitter()
kmf.fit(t[0:int(n_samples/2)], event_observed=f[0:int(n_samples/2)])
plt.plot(nnet_survival.breaks,np.concatenate(([1],np.cumprod(y_pred[0,:]))),'bo-')
plt.plot(kmf.survival_function_.index.values, kmf.survival_function_.KM_estimate,color='k')
kmf.fit(t[int(n_samples/2)+1:], event_observed=f[int(n_samples/2)+1:])
plt.plot(nnet_survival.breaks,np.concatenate(([1],np.cumprod(y_pred[-1,:]))),'ro-')
plt.plot(kmf.survival_function_.index.values, kmf.survival_function_.KM_estimate,color='k')
plt.xticks(np.arange(0, 2000.0001, 200))
plt.yticks(np.arange(0, 1.0001, 0.125))
plt.xlim([0,2000])
plt.ylim([0,1])
plt.xlabel('Follow-up time (days)')
plt.ylabel('Proportion surviving')
plt.title('One covariate. Actual=black, predicted=blue/red.')
plt.show()
