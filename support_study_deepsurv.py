#Tested with Python 2.7.15 (DeepSurv gave errors with Python 3.6)
#Install lasagne using instructions at: https://github.com/Lasagne/Lasagne
#Then, install DeepSurv using instructions
#at: https://github.com/jaredleekatzman/DeepSurv/blob/master/deepsurv/deep_surv.py

from __future__ import print_function
import lasagne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import math
from lifelines import KaplanMeierFitter
from lifelines.utils import concordance_index
from sklearn.preprocessing import StandardScaler
import time

import deepsurv

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



#############################################################
#SUPPORT study data (publicly available courtesy of Vanderbilt Dep't of Biostatistics)
#Prospective study survival data on 9105 hospitalized patients
#Data: http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/support2csv.zip
#Description: http://biostat.mc.vanderbilt.edu/wiki/Main/SupportDesc
#The data have been cleaned and missing values have been imputed.

train_prop = 0.7 #proportion of patients to place in training set
learning_rate = 1e-2
hidden_layers_sizes = 7 #Using single hidden layer, with this many neurons
n_in = 39 # input dimensions
n_epochs = 5000

data_support = pd.read_csv('data/support_parsed.csv')
np.random.seed(0)
train_indices = np.random.choice(len(data_support),int(train_prop*len(data_support)),replace=False)
test_indices = np.setdiff1d(np.arange(len(data_support)), train_indices)
data_train = data_support.iloc[train_indices]
data_test = data_support.iloc[test_indices]
data_train = data_train.reset_index(drop=True)
data_test = data_test.reset_index(drop=True)

x_train = data_train.drop(["time", "dead"], axis=1).values
x_test = data_test.drop(["time", "dead"], axis=1).values
scaler = StandardScaler().fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#########
#DeepSurv
train_data_deepsurv = {
	'x':x_train.astype('float32'),
	't':data_train.time.values.astype('float32'),
	'e':data_train.dead.values.astype('int32')}
test_data_deepsurv = {
	'x':x_test.astype('float32'),
	't':data_test.time.values.astype('float32'),
	'e':data_test.dead.values.astype('int32')}

#################################################
#Cross-validation to pick regularization strength

from sklearn.model_selection import KFold
n_folds = 10
kf=KFold(n_splits=n_folds, shuffle=True, random_state=0)

l2_array = np.power(10.,np.arange(-5,1))
grid_search_test = np.zeros((len(l2_array),n_folds))
for i in range(len(l2_array)):
	print(str(i+1) + '/' + str(len(l2_array)))
	j=0
	cv_folds = kf.split(x_train)
	for traincv, testcv in cv_folds:
		train_data_deepsurv_cv = {
			'x':x_train[traincv].astype('float32'),
			't':data_train.loc[traincv].time.values.astype('float32'),
			'e':data_train.loc[traincv].dead.values.astype('int32')}
		test_data_deepsurv_cv = {
			'x':x_train[testcv].astype('float32'),
			't':data_train.loc[testcv].time.values.astype('float32'),
			'e':data_train.loc[testcv].dead.values.astype('int32')}
		hyperparams = {'n_in': n_in, 'learning_rate': learning_rate, 'L2_reg': l2_array[i],'hidden_layers_sizes': [hidden_layers_sizes]}
		network = deepsurv.DeepSurv(**hyperparams)
		log = network.train(train_data_deepsurv_cv, test_data_deepsurv_cv, n_epochs=n_epochs)
		grid_search_test[i,j] = log['best_valid_loss']
		j=j+1

print(np.average(grid_search_test,axis=1))
l2_final = l2_array[np.argmax(-np.average(grid_search_test,axis=1))]
l2_final = 1e-3

hyperparams = {
            'n_in': n_in,
            'learning_rate': learning_rate,
            'L2_reg': l2_final,
            'hidden_layers_sizes': [hidden_layers_sizes]
        }

network = deepsurv.DeepSurv(**hyperparams)
log = network.train(train_data_deepsurv, n_epochs=n_epochs)
print('DeepSurv C-index training set:')
print(network.get_concordance_index(**train_data_deepsurv))
print('DeepSurv C-index test set:')
print(network.get_concordance_index(**test_data_deepsurv))

#save predictions
train_prediction = network.predict_risk(x_train.astype('float32'))
np.savetxt('results/deepsurv_train_prediction.txt',train_prediction)
test_prediction = network.predict_risk(x_test.astype('float32'))
np.savetxt('results/deepsurv_test_prediction.txt',test_prediction)

#calibration plot for test set
times, H0 = cox_basehaz(network.predict_risk(x_train.astype('float32')).flatten(), data_train.time.values, data_train.dead.values)
y_pred = cox_pred_surv(network.predict_risk(x_test.astype('float32')).flatten(), H0)
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

##################################
#Running time experiment: DeepSurv
#To restrict Theano to one core, before starting Python, execute:
#export OMP_NUM_THREADS=1


n_reps=3
epochs=1000
l2_final = 1e-3
sample_size_array = np.power(10,np.arange(3,6.1,0.5)).astype(np.int64)
#sample_size_array = np.power(10,np.arange(5,6.1,0.5)).astype(np.int64)
time_array = np.zeros((len(sample_size_array),n_reps))

for i in range(len(sample_size_array)):
	for j in range(n_reps):
		data_train = data_support.sample(n=sample_size_array[i],replace=True,random_state=i*10+j)
		x_train = data_train.drop(["time", "dead"], axis=1).values
		scaler = StandardScaler().fit(x_train)
		x_train = scaler.transform(x_train)
		train_data_deepsurv = {
		'x':x_train.astype('float32'),
			't':data_train.time.values.astype('float32'),
			'e':data_train.dead.values.astype('int32')}
		hyperparams = {
		            'n_in': n_in,
		            'learning_rate': learning_rate,
		            'L2_reg': l2_final,
		            'hidden_layers_sizes': [hidden_layers_sizes]
		        }
		start = time.time()
		network = deepsurv.DeepSurv(**hyperparams)
		log = network.train(train_data_deepsurv, n_epochs=epochs)
		times, H0 = cox_basehaz(network.predict_risk(x_train.astype('float32')).flatten(), data_train.time.values, data_train.dead.values)
		end = time.time()
		time_array[i,j]=end-start
		print(sample_size_array[i],j,time_array[i,j])

time_data=pd.DataFrame({'n' : sample_size_array, 'running_time' : np.average(time_array,axis=1)}) 
time_data.to_csv('results/running_time_deepsurv.csv',index=False)

#print(np.average(time_array,axis=1))
#print(network.get_concordance_index(**train_data_deepsurv))

##########################
#Optimize number of epochs
n_epochs_array=[100, 500, 1000, 2000, 5000, 10000] #5000 produces best test set loss
for n_epochs in n_epochs_array:
	network = deepsurv.DeepSurv(**hyperparams)
	log = network.train(train_data_deepsurv, test_data_deepsurv, n_epochs=n_epochs)
	print(n_epochs)
	print(network.get_concordance_index(**train_data_deepsurv))
	print(network.get_concordance_index(**test_data_deepsurv))
	print(log['best_valid_loss'])
