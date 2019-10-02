#!/usr/bin/python

#Ted Thompson, Sept 2018

"""
	Predicts probabilities of a test_data.csv

	Sample usage from terminal:
            python test_data.csv

        Args:
			test data (csv) : python test_data.csv

        Returns:
            test data predicted () : The id and install_prob column for test_data.csv
"""
#load packages
import missingno as msno
import pandas as pd
import numpy as np
import datetime as dt
import time

import matplotlib.pyplot as plt
from tqdm import tqdm, trange
import sys
import logging
import pickle
from sklearn.externals import joblib

from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import log_loss, accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA

# LOGGING

level = 2
path = "logs/predict_prob_{}.log".format(round(time.time()))

# Determine desired logging level
level = int(str(level) + "0")

# Create a logging instance
logger = logging.getLogger()
logger.setLevel(level)

# Setup logging file
logger_handler = logging.FileHandler(path)
logger_handler.setLevel(level)

# Formatting:
logger_formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')

# Put them together
logger_handler.setFormatter(logger_formatter)

logger.addHandler(logger_handler)
logger.info("Logging successfully configured!")



test_data = sys.argv[1]

logger.info("Loading test_data.csv....")
print("Loading test_data.csv....")
dat_test = pd.read_csv(test_data, sep=';', engine='c', iterator=True, chunksize=1870068)

#processing half of the test data at a time due to resources restrictions
f = [1,2]
for j,dat in zip(tqdm(f),dat_test):
	### set column types
	coltype = {'id': 'category', 'softwareVersion': 'category', 'deviceType': 'str','campaignId':'category', 'platform':'category',
	       'sourceGameId':np.int64, 'country':'category', 'startCount':np.int64, 'viewCount':np.int64, 'clickCount':np.int64,
	       'installCount':np.int64,  'startCount1d':np.int64, 'startCount7d':np.int64,
	       'connectionType':'category'}

	logger.info("setting column type...")
	print("setting column type...")

	for i in coltype:
	    dat[i]=dat[i].astype(coltype[i])

	logger.info("setting time columns...")
	print("setting time columns...")

	#set time column type
	dat['timestamp'] = pd.to_datetime(dat['timestamp'])
	dat['lastStart'] = pd.to_datetime(dat['lastStart'])
	# dates need to be converted for model
	dat['date'] = dat['timestamp'].astype(np.int64)

	dat['lastStartHours'] = (dat['timestamp']-dat['lastStart'])/np.timedelta64(1,'h')
	dat['lastStartHours'].fillna(0,inplace=True)
	dat['month'] = dat['timestamp'].dt.month
	dat['day'] = dat['timestamp'].dt.dayofweek
	dat['hour'] = dat['timestamp'].dt.hour

	#dropping columns I'm not going to need
	#all observations take place in january
	X = dat.drop(columns=['lastStart', 'month'])

	logger.info("setting dummies...")
	print("setting time dummies...")

	#one hot encoding on category type
	X = pd.get_dummies(X, columns=['connectionType', 'platform']) #campainId excluded because not enough resources for cardinality

	X['country_n'] = X.country.astype('object')
	X['country_n'] = X['country_n'].fillna('0')

	PCA_cols = list()
	L_cols = list()
	for i in X.columns:
	    if (X[i].unique().shape[0]>100) and (X[i].dtype not in [np.int64,np.int8,np.datetime64,np.float64]):
	        L_cols.append(i)
	    if (X[i].unique().shape[0]>100) and (X[i].dtype not in [np.int64,np.int8,np.datetime64,np.float64]) and i not in ['id','campaignId','timestamp', 'sourceGameId', 'deviceType']:
	        PCA_cols.append(i)

	# my device doesn't have the resources to compute to complete one hot encoding and pca with softwareVersion and deviceType so they will also be excuded


	#cardinaliry to high for my computer for the following columns

	logger.info("setting PCA + dummies...")
	print("setting PCA + dummies...")

	# list of # of pca for each column
	n_pc=[5,12]

	for i,j in zip(tqdm(PCA_cols),n_pc):
	    x = pd.get_dummies(X[i])
	    pca = PCA(n_components=j)
	    Z = pca.fit_transform(x)
	    print(i+':', pca.components_.shape)
	    Z_dat = pd.DataFrame(Z, columns=[i+'_'+str(n) for n in np.arange(1,pca.components_.shape[0]+1)])
	    X[Z_dat.columns] = Z_dat

	logger.info("setting LabelEncoder")
	print("setting LabelEncoder")

	L = ['campaignId','softwareVersion', 'deviceType', 'country_n']
	for i in tqdm(L):
	    encoder = LabelEncoder()
	    X[i] = encoder.fit_transform(X[i])

	df = X.drop(columns=['country'])
	idx = df[df.isnull().any(axis=1)].index.values
	X.drop(index=idx, inplace=True)

	# set id as index
	X.set_index('id', inplace=True)

	X.drop(columns=['country','softwareVersion', 'timestamp'], inplace=True)
	x = pd.DataFrame(index=X.index)

	logger.info("setting predictions")
	print("setting predictions")

	#load model
	model = joblib.load('rf_model.pkl')

	x['install_prob'] = model.predict_proba(X)[:,1]

	logger.info("saving as test_data_predicted_{j}.csv".format(j))
	print("saving as test_data_predicted_{j}.csv".format(j))

	x.to_csv('results/test_data_predicted_{j}.csv'.format(j))
