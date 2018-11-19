#support_study.py

#Results of nnet-survival and baseline models (Cox prop. hazards model, cox-nnet) on
#SUPPORT study data (publicly available courtesy of Vanderbilt Dep't of Biostatistics)
#Prospective study survival data on 9105 hospitalized patients
#Data: http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/support2csv.zip
#Description: http://biostat.mc.vanderbilt.edu/wiki/Main/SupportDesc
#The data have been cleaned and missing values have been imputed.

#Author: Michael Gensheimer, Stanford University, michael.gensheimer@gmail.com
#Tested with Python version 3.6, Keras version 2 (using TensorFlow backend)


running_time_test = 0
if running_time_test: #disable GPU, set Keras to use only 1 CPU core
	import os
	os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
	import tensorflow as tf
	import keras.backend as K
	config = tf.ConfigProto(intra_op_parallelism_threads=1,\
	        inter_op_parallelism_threads=1, allow_soft_placement=True,\
	        device_count = {'CPU' : 1, 'GPU' : 0})
	session = tf.Session(config=config)
	K.set_session(session)
else:
	import keras.backend as K

from __future__ import print_function
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from keras.preprocessing import sequence
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, Activation, LSTM, GRU, Embedding, Concatenate, Conv1D, GlobalMaxPooling1D, MaxPooling1D, GlobalAveragePooling1D, BatchNormalization, TimeDistributed
from keras import optimizers, layers, regularizers
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model
import math
from lifelines import KaplanMeierFitter
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sklearn.preprocessing import StandardScaler
from scipy import stats
import time

import nnet_survival
import other_code.cox_nnet as cox_nnet #for cox-nnet baseline model

CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
	'#f781bf', '#a65628', '#984ea3',
	'#999999', '#e41a1c', '#dede00']

def cox_basehaz(lp, time, dead):
#Find baseline hazard for Cox model using Breslow method
#Adapted from https://stats.stackexchange.com/questions/46532/cox-baseline-hazard
#Inputs are Numpy arrays.
#lp=Cox model linear predictor values
#time=vector of failure/censoring times
#dead=boolean, did patient fail/die
#
#Returns:
#1: unique failure times
#2: baseline hazard function at these times
	time=pd.Series(time)
	dead=pd.Series(dead)
	prediction = np.expand_dims(np.exp(lp),1)
	failed_times = time.loc[dead==1]
	y = failed_times.value_counts().sort_index().index.values #ordered distinct event times
	d = failed_times.value_counts().sort_index().values #number of events
	h0 = np.zeros(len(y))
	for l in range(len(y)):
		h0[l] = d[l] / np.sum(prediction[time >= y[l]])
	H0 = np.cumsum(h0)
	#surv_baseline = np.exp(-H0)
	return (y, H0)

def cox_pred_surv(lp, H0):
#Predicted survival curves from Cox model
#Inputs are Numpy arrays.
#lp=Cox model linear predictor values
#H0=basline hazard function
#
#Returns: predicted survival rate at each follow-up time
	prediction = np.expand_dims(np.exp(lp),1)
	return np.exp(-np.matmul(prediction, np.expand_dims(H0,0)))

def calib_plot(fu_time, n_bins, pred_surv, time, dead, color, label, error_bars=0,alpha=1., markersize=1., markertype='o'):
	cuts = np.concatenate((np.array([-1e6]),np.percentile(pred_surv, np.arange(100/n_bins,100,100/n_bins)),np.array([1e6])))
	bin = pd.cut(pred_surv,cuts,labels=False)
	kmf = KaplanMeierFitter()
	est = []
	ci_upper = []
	ci_lower = []
	mean_pred_surv = []
	for which_bin in range(max(bin)+1):
		kmf.fit(time[bin==which_bin], event_observed=dead[bin==which_bin])
		est.append(np.interp(fu_time, kmf.survival_function_.index.values, kmf.survival_function_.KM_estimate))
		ci_upper.append(np.interp(fu_time, kmf.survival_function_.index.values, kmf.confidence_interval_.loc[:,'KM_estimate_upper_0.95']))
		ci_lower.append(np.interp(fu_time, kmf.survival_function_.index.values, kmf.confidence_interval_.loc[:,'KM_estimate_lower_0.95']))
		mean_pred_surv.append(np.mean(pred_surv[bin==which_bin]))
	est = np.array(est)
	ci_upper = np.array(ci_upper)
	ci_lower = np.array(ci_lower)
	if error_bars:
		plt.errorbar(mean_pred_surv, est, yerr = np.transpose(np.column_stack((est-ci_lower,ci_upper-est))), fmt='o',c=color,label=label)
	else:
		plt.plot(mean_pred_surv, est, markertype, c=color,label=label, alpha=alpha, markersize=markersize)
	return (mean_pred_surv, est)


data_support = pd.read_csv('data/support_parsed.csv')
train_prop = 0.7 #proportion of patients to place in training set
np.random.seed(0)
train_indices = np.random.choice(len(data_support),int(train_prop*len(data_support)),replace=False)
test_indices = np.setdiff1d(np.arange(len(data_support)), train_indices)
data_train = data_support.iloc[train_indices]
data_test = data_support.iloc[test_indices]

x_train = data_train.drop(["time", "dead"], axis=1).as_matrix()
x_test = data_test.drop(["time", "dead"], axis=1).as_matrix()
scaler = StandardScaler().fit(x_train)
x_train = scaler.transform(x_train) #Standardize each predictor variable
x_test = scaler.transform(x_test)

########################################
#Standard Cox proportional hazards model

from lifelines import CoxPHFitter
cph = CoxPHFitter()
cph.fit(data_train, duration_col='time', event_col='dead')
#cph.print_summary()

#Cox model discrimination train set
prediction = cph.predict_partial_hazard(data_train)
print(concordance_index(data_train.time,-prediction,data_train.dead)) #0.735

#Cox model discrimination test set
prediction = cph.predict_partial_hazard(data_test)
print(concordance_index(data_test.time,-prediction,data_test.dead)) #0.735


################################
#Nnet-survival / Our model (flexible version to
#allow non-proportional hazards)

halflife=365.*1.4
breaks=-np.log(1-np.arange(0.0,0.96,0.05))*halflife/np.log(2) 
#breaks=-np.log(1-np.arange(0.0,1,0.099))*halflife/np.log(2) 
n_intervals=len(breaks)-1
timegap = breaks[1:] - breaks[:-1]

y_train = nnet_survival.make_surv_array(data_train.time.values,data_train.dead.values,breaks)
y_test = nnet_survival.make_surv_array(data_test.time.values,data_test.dead.values,breaks)

hidden_layers_sizes = 7 #Using single hidden layer, with this many neurons

##############################################################
#Our model cross-validation to pick L2 regularization strength

from sklearn.model_selection import KFold
n_folds = 10
kf=KFold(n_splits=n_folds, shuffle=True, random_state=0)
early_stopping = EarlyStopping(monitor='loss', patience=20)

#l2_array = np.concatenate(([0.],np.power(10.,np.arange(-6,-2))))
l2_array = np.power(10.,np.arange(-4,1))
grid_search_train = np.zeros((len(l2_array),n_folds))
grid_search_test = np.zeros((len(l2_array),n_folds))
for i in range(len(l2_array)):
	print(str(i+1) + '/' + str(len(l2_array)))
	j=0
	cv_folds = kf.split(x_train)
	for traincv, testcv in cv_folds:
		x_train_cv = x_train[traincv]
		y_train_cv = y_train[traincv]
		x_test_cv = x_train[testcv]
		y_test_cv = y_train[testcv]
		model = Sequential()
		#model.add(Dense(n_intervals,input_dim=x_train.shape[1],bias_initializer='zeros',kernel_regularizer=regularizers.l2(l2_array[i])))
		model.add(Dense(hidden_layers_sizes, input_dim=x_train.shape[1],bias_initializer='zeros', activation='relu', kernel_regularizer=regularizers.l2(l2_array[i])))
		model.add(Dense(n_intervals))
		model.add(Activation('sigmoid'))
		model.compile(loss=nnet_survival.surv_likelihood(n_intervals), optimizer=optimizers.RMSprop()) #lr=0.0001))
		history=model.fit(x_train_cv, y_train_cv, batch_size=256, epochs=100000, callbacks=[early_stopping],verbose=0)
		grid_search_train[i,j] = model.evaluate(x_train_cv,y_train_cv,verbose=0)
		grid_search_test[i,j] = model.evaluate(x_test_cv,y_test_cv,verbose=0)
		j=j+1

print(np.average(grid_search_train,axis=1))
print(np.average(grid_search_test,axis=1))
l2_final = l2_array[np.argmax(-np.average(grid_search_test,axis=1))]

############################
#Our model: train final model
l2_final=0.1

from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

model = Sequential()
model.add(Dense(hidden_layers_sizes, input_dim=x_train.shape[1],bias_initializer='zeros', kernel_regularizer=regularizers.l2(l2_final)))
model.add(Activation('relu'))
model.add(Dense(n_intervals))
model.add(Activation('sigmoid'))
model.compile(loss=nnet_survival.surv_likelihood(n_intervals), optimizer=optimizers.RMSprop())
early_stopping = EarlyStopping(monitor='loss', patience=20)
history=model.fit(x_train, y_train, batch_size=256, epochs=100000, callbacks=[early_stopping],verbose=0)

#Discrimination performance
y_pred=model.predict_proba(x_train,verbose=0)
oneyr_surv=np.cumprod(y_pred[:,0:np.nonzero(breaks>365)[0][0]], axis=1)[:,-1]
print(concordance_index(data_train.time,oneyr_surv,data_train.dead)) #0.723

y_pred=model.predict_proba(x_test,verbose=0)
oneyr_surv=np.cumprod(y_pred[:,0:np.nonzero(breaks>365)[0][0]], axis=1)[:,-1]
print(concordance_index(data_test.time,oneyr_surv,data_test.dead)) #0.723


#########
#cox-nnet
#https://github.com/traversc/cox-nnet/

#cross validation on training set to pick L2 regularization strength
model_params = dict(node_map = None, input_split = None)
search_params = dict(method = "nesterov", learning_rate=0.01, momentum=0.9, 
	max_iter=10000, stop_threshold=0.995, patience=1000, patience_incr=2, 
	rand_seed = 123, eval_step=23, lr_decay = 0.9, lr_growth = 1.0)
cv_params = dict(L2_range = np.arange(-6,2.1))

likelihoods, L2_reg_params, mean_cvpl = cox_nnet.L2CVProfile(x_train,data_train.time.as_matrix(),data_train.dead.as_matrix(),
    model_params, search_params, cv_params, verbose=False)

L2_reg = L2_reg_params[np.argmax(mean_cvpl)] #Best L2_reg is -5

#train final model
L2_reg = -5.
model_params = dict(node_map = None, input_split = None, L2_reg=np.exp(L2_reg))
cox_nnet_model, cox_nnet_cost_iter = cox_nnet.trainCoxMlp(x_train, data_train.time.as_matrix(),data_train.dead.as_matrix(), model_params, search_params, verbose=False)
cox_nnet_theta_train = cox_nnet_model.predictNewData(x_train)
cox_nnet_theta_test = cox_nnet_model.predictNewData(x_test)

#discrimination on train, test sets
print(concordance_index(data_train.time,-cox_nnet_theta_train,data_train.dead))
print(concordance_index(data_test.time,-cox_nnet_theta_test,data_test.dead))


#######################################
#Calibration plot comparing all methods

n_bins = 10
my_alpha = 0.7
my_markersize = 5.
fu_time_array = np.array([0.5, 1, 3])*365.
fu_time_label_array = ['6 months', '1 year', '3 years']
#mse_array = np.zeros((4,len(fu_time_array)))
for fu_time_i in range(len(fu_time_array)):
	fu_time = fu_time_array[fu_time_i]
	plt.subplot(3, 1, 1+fu_time_i)
	#plt.figure()
	plt.plot([0,1], [0,1], ls="--", c=".7")
	pred_surv = nnet_pred_surv(model.predict_proba(x_test,verbose=0), breaks, fu_time)
	(pred, actual)=calib_plot(fu_time, n_bins, pred_surv,data_test.time.as_matrix(), data_test.dead.as_matrix(),
		CB_color_cycle[1],'Nnet-survival', alpha=my_alpha, markersize=my_markersize, markertype='o')
	#mse_array[0, fu_time_i] = ((pred-actual)**2).mean()
	times, H0 = cox_basehaz(cox_nnet_theta_train, data_train.time.values, data_train.dead.values)
	y_pred = cox_pred_surv(cox_nnet_theta_test, H0)
	pred_surv = []
	for i in range(y_pred.shape[0]):
		pred_surv.append(np.interp(fu_time,times,y_pred[i,:]))
	pred_surv = np.array(pred_surv)
	(pred, actual)=calib_plot(fu_time, n_bins, pred_surv, data_test.time.as_matrix(), data_test.dead.as_matrix(),
		CB_color_cycle[0],'Cox-nnet', alpha=my_alpha, markersize=my_markersize, markertype='^')
	#mse_array[1, fu_time_i] = ((pred-actual)**2).mean()
	deepsurv_lp_train = np.genfromtxt('results/deepsurv_train_prediction.txt')
	deepsurv_lp_test = np.genfromtxt('results/deepsurv_test_prediction.txt')
	times, H0 = cox_basehaz(deepsurv_lp_train, data_train.time.values, data_train.dead.values)
	y_pred = cox_pred_surv(deepsurv_lp_test, H0)
	pred_surv = []
	for i in range(y_pred.shape[0]):
		pred_surv.append(np.interp(fu_time,times,y_pred[i,:]))
	pred_surv = np.array(pred_surv)
	(pred, actual)=calib_plot(fu_time, n_bins, pred_surv, data_test.time.as_matrix(), data_test.dead.as_matrix(),
		CB_color_cycle[3],'Deepsurv', alpha=my_alpha, markersize=my_markersize, markertype='s')
	#mse_array[2, fu_time_i] = ((pred-actual)**2).mean()
	y_pred=cph.predict_survival_function(data_test)
	times=y_pred.index.values.astype('float64')
	y_pred=y_pred.as_matrix().transpose()
	pred_surv = []
	for i in range(y_pred.shape[0]):
		pred_surv.append(np.interp(fu_time,times,y_pred[i,:]))
	pred_surv = np.array(pred_surv)
	(pred, actual)=calib_plot(fu_time, n_bins, pred_surv, data_test.time.as_matrix(), data_test.dead.as_matrix(),
		CB_color_cycle[2],'Cox PH model', alpha=my_alpha, markersize=my_markersize, markertype='D')
	#mse_array[3, fu_time_i] = ((pred-actual)**2).mean()
	plt.xlim([0,1])
	plt.ylim([0,1])
	plt.legend()
	plt.xlabel('Predicted survival rate')
	plt.ylabel('Actual survival rate')
	plt.title(fu_time_label_array[fu_time_i])

plt.show()
#plt.savefig("results/support_study_calib.pdf", bbox_inches='tight')

#mean squared error for each method
#print(mse_array.mean(axis=1))

#######################################
#Export predictions for processing in R
deepsurv_lp_train = np.genfromtxt('results/deepsurv_train_prediction.txt')
deepsurv_lp_test = np.genfromtxt('results/deepsurv_test_prediction.txt')
	
fu_time_label_array = ['6 months', '1 year', '3 years']
pred_array = np.zeros((4,x_test.shape[0], len(fu_time_array)))
for fu_time_i in range(len(fu_time_array)):
	fu_time = fu_time_array[fu_time_i]
	y_pred=model.predict_proba(x_test,verbose=0)
	y_pred=np.cumprod(y_pred, axis=1)
	for i in range(y_pred.shape[0]):
		pred_array[0,i,fu_time_i] = np.interp(fu_time,breaks[1:],y_pred[i,:])
	times, H0 = cox_basehaz(cox_nnet_theta_train, data_train.time.values, data_train.dead.values)
	y_pred = cox_pred_surv(cox_nnet_theta_test, H0)
	for i in range(y_pred.shape[0]):
		pred_array[1,i,fu_time_i] = np.interp(fu_time,times,y_pred[i,:])
	times, H0 = cox_basehaz(deepsurv_lp_train, data_train.time.values, data_train.dead.values)
	y_pred = cox_pred_surv(deepsurv_lp_test, H0)
	for i in range(y_pred.shape[0]):
		pred_array[2,i,fu_time_i] = np.interp(fu_time,times,y_pred[i,:])
	y_pred=cph.predict_survival_function(data_test)
	times=y_pred.index.values.astype('float64')
	y_pred=y_pred.as_matrix().transpose()
	for i in range(y_pred.shape[0]):
		pred_array[3,i,fu_time_i] = np.interp(fu_time,times,y_pred[i,:])

np.save("results/preds_nnet-survival.npy", pred_array[0,:,:])
np.save("results/preds_cox_nnet.npy", pred_array[1,:,:])
np.save("results/preds_deepsurv.npy", pred_array[2,:,:])
np.save("results/preds_cph.npy", pred_array[3,:,:])
export_data=data_test.loc[:,['time','dead']]
export_data.to_csv('results/test_surv.csv',index=False)



###################
#Running time tests

n_reps=3
epochs = 1000
sample_size_array = np.power(10,np.arange(3,6.1,0.5)).astype(np.int64)

#Our model (Nnet-survival)
time_array = np.zeros((len(sample_size_array),n_reps))
for i in range(len(sample_size_array)):
	for j in range(n_reps):
		data_train = data_support.sample(n=sample_size_array[i],replace=True,random_state=i*10+j)
		x_train = data_train.drop(["time", "dead"], axis=1).values
		y_train = nnet_survival.make_surv_array(data_train.time.values,data_train.dead.values,breaks)
		scaler = StandardScaler().fit(x_train)
		x_train = scaler.transform(x_train)
		start = time.time()
		model = Sequential()
		model.add(Dense(hidden_layers_sizes, input_dim=x_train.shape[1],bias_initializer='zeros', activation='relu'))
		model.add(Dense(n_intervals))
		model.add(Activation('sigmoid'))
		model.compile(loss=nnet_survival.surv_likelihood(n_intervals), optimizer=optimizers.RMSprop())
		history=model.fit(x_train, y_train, batch_size=256, epochs=epochs, verbose=0)
		y_pred=model.predict_proba(x_train,verbose=0)
		end = time.time()
		time_array[i,j]=end-start
		print(sample_size_array[i],j,time_array[i,j])

time_data=pd.DataFrame({'n' : sample_size_array, 'running_time' : np.average(time_array,axis=1)}) 
time_data.to_csv('results/running_time_nnet_survival.csv',index=False)

#Cox-nnet model
time_array = np.zeros((len(sample_size_array),n_reps))
model_params = dict(node_map = None, input_split = None, L2_reg=np.exp(-3))
search_params = dict(method = "nesterov", learning_rate=0.01, momentum=0.9, 
    max_iter=epochs, stop_threshold=0.995, patience=1000000, patience_incr=2, 
    rand_seed = 123, eval_step=23, lr_decay = 0.9, lr_growth = 1.0)
for i in range(len(sample_size_array)):
	for j in range(n_reps):
		data_train = data_support.sample(n=sample_size_array[i],replace=True,random_state=i*10+j)
		x_train = data_train.drop(["time", "dead"], axis=1).values
		scaler = StandardScaler().fit(x_train)
		x_train = scaler.transform(x_train)
		start = time.time()
		cox_nnet_model, cox_nnet_cost_iter = cox_nnet.trainCoxMlp(x_train, data_train.time.as_matrix(),data_train.dead.as_matrix(), model_params, search_params, verbose=False)
		cox_nnet_theta_train = cox_nnet_model.predictNewData(x_train)
		times, H0 = cox_basehaz(cox_nnet_theta_train, data_train.time.values, data_train.dead.values)
		end = time.time()
		time_array[i,j]=end-start
		print(sample_size_array[i],j,time_array[i,j])

time_data=pd.DataFrame({'n' : sample_size_array, 'running_time' : np.average(time_array,axis=1)}) 
time_data.to_csv('results/running_time_coxnnet.csv',index=False)

#Cox PH
time_array = np.zeros((len(sample_size_array),n_reps))
for i in range(len(sample_size_array)):
	for j in range(n_reps):
		data_train = data_support.sample(n=sample_size_array[i],replace=True,random_state=i*10+j)
		start = time.time()
		cph = CoxPHFitter()
		cph.fit(data_train, duration_col='time', event_col='dead')
		y_pred=cph.predict_survival_function(data_train)
		end = time.time()
		time_array[i,j]=end-start
		print(sample_size_array[i],j,time_array[i,j])

time_data=pd.DataFrame({'n' : sample_size_array, 'running_time' : np.average(time_array,axis=1)}) 
time_data.to_csv('results/running_time_coxph.csv',index=False)



###############################
##Extra code: calibration plots

#Cox model calibration train set
y_pred=cph.predict_survival_function(data_train)
times=y_pred.index.values
y_pred=y_pred.as_matrix().transpose()
cuts = np.concatenate((np.array([-1e6]),np.percentile(y_pred[:,1], [25, 50, 75]),np.array([1e6])))
bin = pd.cut(y_pred[:,1],cuts,labels=False)

kmf = KaplanMeierFitter()
for which_bin in range(max(bin)+1):
  kmf.fit(data_train.time.iloc[bin==which_bin], event_observed=data_train.dead.iloc[bin==which_bin])
  plt.plot(kmf.survival_function_.index.values, kmf.survival_function_.KM_estimate,color='k')
  pred_surv=np.mean(y_pred[bin==which_bin,:],axis=0)
  plt.plot(times,pred_surv,'b-')

plt.xticks(np.arange(0, 365*5, 365))
plt.yticks(np.arange(0, 1.0001, 0.125))
plt.xlim([0,365.25*5])
plt.ylim([0,1])
plt.gca().set_position([0.1, 0.1, .8, .8])
plt.show()

#Cox model calibration test set
y_pred=cph.predict_survival_function(data_test)
times=y_pred.index.values
y_pred=y_pred.as_matrix().transpose()
cuts = np.concatenate((np.array([-1e6]),np.percentile(y_pred[:,1], [25, 50, 75]),np.array([1e6])))
bin = pd.cut(y_pred[:,1],cuts,labels=False)

kmf = KaplanMeierFitter()
for which_bin in range(max(bin)+1):
  kmf.fit(data_test.time.iloc[bin==which_bin], event_observed=data_test.dead.iloc[bin==which_bin])
  plt.plot(kmf.survival_function_.index.values, kmf.survival_function_.KM_estimate,color='k')
  pred_surv=np.mean(y_pred[bin==which_bin,:],axis=0)
  plt.plot(times,pred_surv,'b-')

plt.xticks(np.arange(0, 365*5, 365))
plt.yticks(np.arange(0, 1.0001, 0.125))
plt.xlim([0,365.25*5])
plt.ylim([0,1])
plt.gca().set_position([0.1, 0.1, .8, .8])
plt.show()

#Our model: train set calibration

#cuts = np.concatenate((np.array([-1e6]),np.percentile(oneyr_surv, [25, 50, 75]),np.array([1e6])))
cuts = np.concatenate((np.array([-1e6]),np.percentile(oneyr_surv, np.arange(25,100,25)),np.array([1e6])))
bin = pd.cut(oneyr_surv,cuts,labels=False)

kmf = KaplanMeierFitter()
for which_bin in range(max(bin)+1):
  kmf.fit(data_train.time.iloc[bin==which_bin], event_observed=data_train.dead.iloc[bin==which_bin])
  plt.plot(kmf.survival_function_.index.values, kmf.survival_function_.KM_estimate,color='k')
  pred_surv=np.mean(np.cumprod(y_pred[bin==which_bin,:], axis=1),axis=0)
  plt.plot(breaks,np.concatenate(([1],pred_surv)),'b-')
  
plt.xticks(np.arange(0, 365*5, 365))
plt.yticks(np.arange(0, 1.0001, 0.125))
plt.xlim([0,365.25*5])
plt.ylim([0,1])
plt.gca().set_position([0.1, 0.1, .8, .8])
plt.show()

####
#calibration plot at specific f/u time
fu_time = 365./2
n_bins = 10

plt.figure()
plt.plot([0,1], [0,1], ls="--", c=".7")

y_pred=cph.predict_survival_function(data_train)
times=y_pred.index.values.astype(float)
y_pred=y_pred.as_matrix().transpose()
pred_surv = []
for i in range(y_pred.shape[0]):
	pred_surv.append(np.interp(fu_time,times,y_pred[i,:]))

pred_surv = np.array(pred_surv)

calib_plot(fu_time, n_bins, pred_surv, data_train.time.as_matrix(), data_train.dead.as_matrix(), 'g','Cox')

y_pred=model.predict_proba(x_train,verbose=0)
y_pred=np.cumprod(y_pred, axis=1)
pred_surv = []
for i in range(y_pred.shape[0]):
	pred_surv.append(np.interp(fu_time,breaks[1:],y_pred[i,:]))

pred_surv = np.array(pred_surv)

calib_plot(fu_time, n_bins, pred_surv, data_train.time.as_matrix(), data_train.dead.as_matrix(), 'b','Nnet-survival')

plt.xlim([0,1])
plt.ylim([0,1])
plt.legend()
plt.show()

#test set calibration

#cuts = np.concatenate((np.array([-1e6]),np.percentile(oneyr_surv, [25, 50, 75]),np.array([1e6])))
cuts = np.concatenate((np.array([-1e6]),np.percentile(oneyr_surv, np.arange(25,100,25)),np.array([1e6])))
bin = pd.cut(oneyr_surv,cuts,labels=False)

kmf = KaplanMeierFitter()
for which_bin in range(max(bin)+1):
  kmf.fit(data_test.time.iloc[bin==which_bin], event_observed=data_test.dead.iloc[bin==which_bin])
  plt.plot(kmf.survival_function_.index.values, kmf.survival_function_.KM_estimate,color='k')
  pred_surv=np.mean(np.cumprod(y_pred[bin==which_bin,:], axis=1),axis=0)
  plt.plot(breaks,np.concatenate(([1],pred_surv)),'b-')
  
plt.xticks(np.arange(0, 365*5, 365))
plt.yticks(np.arange(0, 1.0001, 0.125))
plt.xlim([0,365.25*5])
plt.ylim([0,1])
plt.gca().set_position([0.1, 0.1, .8, .8])
plt.show()


###
#test set calib plot at specific f/u times

fu_time = 365./2
n_bins = 5

plt.figure()
plt.plot([0,1], [0,1], ls="--", c=".7")

y_pred=cph.predict_survival_function(data_test)
times=y_pred.index.values.astype(float)
y_pred=y_pred.as_matrix().transpose()
pred_surv = []
for i in range(y_pred.shape[0]):
	pred_surv.append(np.interp(fu_time,times,y_pred[i,:]))

pred_surv = np.array(pred_surv)

calib_plot(fu_time, n_bins, pred_surv, data_test.time.as_matrix(), data_test.dead.as_matrix(), 'g','Cox')

y_pred=model.predict_proba(x_test,verbose=0)
y_pred=np.cumprod(y_pred, axis=1)
pred_surv = []
for i in range(y_pred.shape[0]):
	pred_surv.append(np.interp(fu_time,breaks[1:],y_pred[i,:]))

pred_surv = np.array(pred_surv)

calib_plot(fu_time, n_bins, pred_surv, data_test.time.as_matrix(), data_test.dead.as_matrix(), 'b','Nnet-survival')

plt.xlim([0,1])
plt.ylim([0,1])
plt.legend()
plt.show()
