import os
import gzip 
import csv
import numpy as np

	
def predict(N):
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
	predict(800000)

