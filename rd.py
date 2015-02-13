from rdkit import Chem
from rdkit.Chem import ChemicalFeatures
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit import RDConfig
import os
import gzip 
import csv
import numpy as np
from rdkit.Chem.rdMolChemicalFeatures import *
from rdkit.Chem import AllChem
from rdkit.Chem import MACCSkeys


# fdefName = os.path.join(RDConfig.RDDataDir,'BaseFeatures.fdef')
# factory = ChemicalFeatures.BuildFeatureFactory(fdefName)
# m = Chem.MolFromSmiles('OCc1ccccc1CN')
# feats = factory.GetFeaturesForMol(m)
# print feats[0].GetFamily()

# inf = gzip.open('test1.csv.gz')
# gzsuppl = Chem.ForwardSDMolSupplier(inf)
# print gzsuppl
# ms = [x for x in gzsuppl]
# print len(ms)
def read_data(train_csv, N):
	smiles = []
	gaps = []
	all_feats = []
	fdefName = os.path.join(RDConfig.RDDataDir,'BaseFeatures.fdef')
	factory = ChemicalFeatures.BuildFeatureFactory(fdefName)
	# Load the data.
	for row in train_csv:
		# Grab the smile data
		smile = row[0]
		m = Chem.MolFromSmiles(smile)
		#Pharmacophores --> core dump/doesnt work
		#feats = np.array(factory.GetFeaturesForMol(m))

		#Topological fingerprint --> unmatched matrix size/doesnt work
		#feats = np.array(FingerprintMols.GetRDKFingerprint(m))

		#MACCS --> 167 bits and yields high rmse
		#feats = np.array(MACCSkeys.GenMACCSKeys(m))

		#Morgan Fingerprint --> optimal: 2, 4096 bits 
		feats = np.array(AllChem.GetMorganFingerprintAsBitVect(m, 1, nBits=4096, useFeatures=True))
		
		#Hashed Atom Pair Finger -< high rmse aroun 0.4
		#feats = np.array(AllChem.GetHashedAtomPairFingerprintAsBitVect(m, nBits=2048, includeChirality=True))

		#Hashed Topological Torsion fingerprint
		#feats = np.array(AllChem.GetHashedTopologicalTorsionFingerprintAsBitVect(m, nBits=4096, includeChirality=True))
		all_feats.append(feats)


		#Grab the gap data 
		gap = float(row[257])
		gaps.append(gap)

		N -= 1
		if N == 0: break

	return (gaps, all_feats)

def rmse(train_csv, w, N):
	gaps, features = read_data(train_csv, N)
	Y = np.vstack(np.array(gaps)).T
	X = (np.vstack((np.ones(N), (np.vstack(tuple(features))).T))).T
	Y_hat = np.dot(X, w).T
	return np.sqrt((np.sum(np.square(Y - Y_hat))) / N)
	
def smile_extract(N):
	train_filename = 'train.csv.gz'
	test_filename  = 'test.csv.gz'
	pred_filename  = 'example_mean.csv'

	# Load the training file
	with gzip.open(train_filename, 'r') as train_fh:
		# Parse it as a CSV file.
		train_csv = csv.reader(train_fh, delimiter=',', quotechar='"')
		
		# Skip the header row.
		next(train_csv, None)

		data = read_data(train_csv, N)
		gaps = data[0]
		all_feats = data[1]
		# Compute the ridge regression parameters
		while True:
			lam = 0.001
			Y = np.vstack(np.array(gaps)) # N x 1
			X = (np.vstack((np.ones(N), (np.vstack(tuple(all_feats))).T))).T
			# X = (np.vstack(tuple(features))) # N x J (where J = # features)
			w = np.linalg.solve(np.dot(X.T, X) + lam * np.identity(4097), np.dot(X.T, Y))
			# w0 = np.sum(Y) / Y.shape[0] # compute w0 as sample mean of training data
			# w = np.vstack(np.insert(w, 0, w0)) # (J+1) x 1
			print rmse(train_csv, w, N)
		return rmse(train_csv, w, N)

smile_extract(1000)

