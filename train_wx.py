import os
import gzip 
import csv
import numpy as np

from rdkit import Chem
from rdkit import RDConfig

from rdkit.Chem import ChemicalFeatures
from rdkit.Chem import AllChem
from rdkit.Chem import MACCSkeys
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit.Chem.rdMolChemicalFeatures import *

from sklearn import linear_model
from sklearn import svm

'''
Linear Regression
'''
# clf = linear_model.LinearRegresion()
# clf.fit([X], [y])
# clf.coef_

'''
Ridge Regression
'''
# clf = linear_model.Ridge(alpha = .5)
# clf.fit([X], [y])
# w is represented as below:
# clf.coef_ 

'''
Ridge Regression with Cross Validation
'''
#clf = linear_model.RidgeCV(alphas=[lam])
#clf.fit([X], [y])
# The optimal alpha/lam value is represented by below
#clf.alpha_

'''
Lasso Regression
'''
#clf = linear_model.Lasso(alpha=0.1)
#clf.fit([X], [y])

'''
ElasticNet
'''
# clf = linear_model.ElasticNetCV(alpha, l1_ratio)

'''
BayesianRidge
Default:
alpha_1 = 1**-6 
alpha_2 = 1**-6 
lambda_1 = 1**-6 
lambda_2 = 1**-6 
'''
# X = [[0,0], [1,1]]
# Y = [0,1,2,...]
# clf = linear_model.BayesianRidge()
# clf.fit(X, Y)
# clf.predict([X])
# clf.coef_

'''
Support Vector Regression
'''
clf = svm.SVR()
clf.fit(X,y)
clf.predict([[]])

def read_data(train_csv, N):
	smiles = []
	gaps = []
	all_feats = []
	# Load the data.
	for row in train_csv:
		# Grab the smile data
		smile = row[0]
		m = Chem.MolFromSmiles(smile)
		#Morgan Fingerprint --> optimal: 1, 1024 bits 
		feats = np.array(AllChem.GetMorganFingerprintAsBitVect(m, 1, nBits=1024, useFeatures=True))
		all_feats.append(feats)

		#Grab the gap data 
		gap = float(row[257])
		gaps.append(gap)
		N -= 1
		if N%1000 == 0:
			print N
		if N == 0: break

	return (gaps, all_feats)

def rmse(train_csv, w, gaps, features):
	Y = np.vstack(np.array(gaps)).T
	X = (np.vstack((np.ones(N), (np.vstack(tuple(features))).T))).T
	Y_hat = np.dot(X, w).T
	return np.sqrt((np.sum(np.square(Y - Y_hat))) / N)
	
def cross_validation(all_gaps, all_features, N):
  lam = .5
  min_lam_err = float("inf")
  while True:
	print "start"
	lam_err = 0
	for i in xrange(0,5):
	  test_gaps = []
	  test_features = []
	  train_gaps = []
	  train_features = []

	  for j in xrange(0,N):
		if j % 5 != i:
		  train_gaps.append(all_gaps[j])
		  train_features.append(all_features[j])
		else:
		  test_gaps.append(all_gaps[j])
		  test_features.append(all_features[j])

	  Y = np.vstack(np.array(train_gaps))
	  X = (np.vstack((np.ones(4*N/5), (np.vstack(tuple(train_features))).T))).T
	  w = np.linalg.solve(np.dot(X.T, X) + lam * np.identity(1025), np.dot(X.T, Y))

	  test_Y = np.vstack(np.array(test_gaps)).T
	  test_X = (np.vstack((np.ones(N/5), (np.vstack(tuple(test_features))).T))).T
	  Y_hat = np.dot(test_X, w).T
	  lam_err += np.sum(np.square(test_Y - Y_hat))
	
	if lam_err/N < min_lam_err:
	  min_lam_err = lam_err/N
	  lam -= .07
	  print min_lam_err
	else:
	  break

  return lam

def ridge_plus_validation(N):
	train_filename = 'train.csv.gz'

	# Load the training file
	with gzip.open(train_filename, 'r') as train_fh:
		# Parse it as a CSV file.
		train_csv = csv.reader(train_fh, delimiter=',', quotechar='"')
		
		# Skip the header row.
		next(train_csv, None)

		data = read_data(train_csv, N)
		gaps = data[0]
		features = data[1]

		lam = cross_validation(gaps, features, N)
		Y = np.vstack(np.array(gaps))
		X = (np.vstack((np.ones(N), (np.vstack(tuple(features))).T))).T
		w = np.linalg.solve(np.dot(X.T, X) + lam * np.identity(1025), np.dot(X.T, Y))
		return w

def predict(N, w):
	test_filename = 'final_features.csv.gz'
	all_feats = []
	index = 1
	# Load the test file
	with gzip.open(test_filename, 'r') as test_fh:
		# Parse it as a CSV file.

		test_csv = csv.reader(test_fh, delimiter=',', quotechar='"')
		#for x in range(0, 800000):
			#next(test_csv, None)
		with open('results.csv', 'w') as results_file:
		# Grab the data.
			results_model = csv.writer(results_file, delimiter=',', quotechar='"')
			results_model.writerow(["Id"] + ["Prediction"])
			next(test_csv, None)
			for row in test_csv:
				#Grab Features 
				features = np.array([float(x) for x in row[1:1025]])
				#X = np.vstack(((1), (tuple(features)))).T
				#X = (np.vstack((np.ones(1), (np.vstack(tuple(features))).T))).T
				temp = np.vstack(tuple(features))
				X = (np.vstack(((1), temp))).T
				Y_hat = np.dot(X, w)[0][0]
				results_model = csv.writer(results_file, delimiter=',', quotechar='"')
				results_model.writerow([index] + [Y_hat])
				index+=1
				N -= 1
				if N % 10000 == 0:
					print N
				if N == 0: break

		return 0

if __name__ == "__main__":
	w = ridge_plus_validation(50000)
	predict(900000, w)
