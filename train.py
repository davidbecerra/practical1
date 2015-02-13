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

def read_data(train_csv, N):
	smiles = []
	gaps = []
	all_feats = []
	# Load the data.
	for row in train_csv:
		# Grab the smile data
		smile = row[0]

		#Morgan Fingerprint --> optimal: 1, 1024 bits 
		feats = np.array([float(x) for x in row[1:1025]])
		all_feats.append(feats)

		#Grab the gap data 
		gap = float(row[1025])
		gaps.append(gap)
		
		N -= 1
		if N == 0: break

	return (gaps, all_feats)

def rmse(train_csv, w, gaps, features):
	Y = np.vstack(np.array(gaps)).T
	X = (np.vstack((np.ones(N), (np.vstack(tuple(features))).T))).T
	Y_hat = np.dot(X, w).T
	return np.sqrt((np.sum(np.square(Y - Y_hat))) / N)
	
def cross_validation(all_gaps, all_features, N):
	lam = 1
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
			lam_err += np.sum(np.square(Y - Y_hat))
		
		if lam_err/N < min_lam_err:
			min_lam_err = lam_err/N
			lam += 150
			print lam
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
		return (w, lam)

if __name__ == "__main__":
	print ridge_plus_validation(100000)
	#read_data(10000)

def predict(N, w, lam):
	test_filename = 'final_features.csv.gz'
	all_feats = []
	w = 0.01
	index = 1
	# Load the test file
	with gzip.open(test_filename, 'r') as test_fh:
		# Parse it as a CSV file.
		test_csv = csv.reader(test_fh, delimiter=',', quotechar='"')
		with open('results.csv', 'w') as results_file:
		# Grab the data.
			for row in test_csv:
				#Grab Features 
				features = np.array([float(x) for x in row[1:1025]])
				X = (np.vstack((np.ones(N), (np.vstack(tuple(features))).T))).T
				Y_hat = np.dot(X, w).T
				
				results_model = csv.writer(results_file, delimiter=',', quotechar='"')
				results_model.writerow([index] + [Y_hat])
				index+1
				N -= 1
				if N == 0: break

		return 0

if __name__ == "__main__":
	w, lam = ridge_plus_validation(100000)
	predict(800000, w, lam)
