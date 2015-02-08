import csv
import gzip
import numpy as np
import math

train_filename = 'train.csv.gz'
test_filename  = 'test.csv.gz'
pred_filename  = 'example_mean.csv'
N = 1000
N_IN = N
N_OUT = N
N_ERROR = N
lam = 0.5
# Load the training file.
train_data = []
#Number of data points
with gzip.open(train_filename, 'r') as train_fh:

    # Parse it as a CSV file.
    train_csv = csv.reader(train_fh, delimiter=',', quotechar='"')
    
    # Skip the header row.
    next(train_csv, None)

    # Load the data.
    for row in train_csv:
        smiles   = row[0]
        features = np.array([float(x) for x in row[1:257]])
        gap      = float(row[257])
        
        train_data.append({ 'smiles':   smiles,
                            'features': features,
                            'gap':      gap })
        N_IN-=1
        if N_IN == 0:
            break

# Compute the mean of the gaps in the training data.
gaps = np.array([datum['gap'] for datum in train_data])
mean_gap = np.mean(gaps)
features = np.array([datum['features'] for datum in train_data])
Temp = np.vstack((np.ones(features.shape[0]), features.T))
X = Temp.T

w = np.linalg.solve(np.dot(X.T, X)+lam*np.identity(X.T.shape[0]), np.dot(X.T, gaps))
# Calculate the root square means

total_error = 0
error = (gaps - np.dot(X, w))**2
for x in error:
    total_error += x
rmse = (total_error/N)**0.5
print rmse   



    

# print total_error

# Load the test file.
test_data = []
with gzip.open(test_filename, 'r') as test_fh:

    # Parse it as a CSV file.
    test_csv = csv.reader(test_fh, delimiter=',', quotechar='"')
    
    # Skip the header row.
    next(test_csv, None)

    # Load the data.
    for row in test_csv:
        id       = row[0]
        smiles   = row[1]
        features = np.array([float(x) for x in row[2:258]])
        
        test_data.append({ 'id':       id,
                           'smiles':   smiles,
                           'features': features })
        N_OUT -=1
        if N_OUT == 0:
            break
# Write a prediction file.
with open(pred_filename, 'w') as pred_fh:

    # Produce a CSV file.
    pred_csv = csv.writer(pred_fh, delimiter=',', quotechar='"')

    # Write the header row.
    pred_csv.writerow(['Id', 'Prediction'])

    for datum in test_data:
        pred_csv.writerow([datum['id'], mean_gap])