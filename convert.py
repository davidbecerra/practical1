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


def read_data(N):
	smiles = []
	gaps = []
	all_feats = []
	test_filename  = 'train.csv.gz'
	# Load the training file
	with gzip.open(test_filename, 'r') as test_fh:
	# Parse it as a CSV file.
		test_csv = csv.reader(test_fh, delimiter=',', quotechar='"')
		
		next(test_csv, None)
		# Load the data.
		with open('train_features.csv', 'w') as feats_file:
			for row in test_csv:
				# Grab the smile data
				smile = row[0]
				m = Chem.MolFromSmiles(smile)
				feats = np.array(AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=4096, useFeatures=True))

				feats_model = csv.writer(feats_file, delimiter=',', quotechar='"')
				feats_model.writerow([smile] + list(feats))
				N -= 1
				if N % 10000 == 0:
					print N
				if N == 0: 
					print "done"
					break

	return 0

if __name__ == "__main__":
  read_data(500000)
