import csv
import gzip
import numpy as np


def read_data(train_csv, N):
  gaps = []
  all_features = []
  # Load the data.
  for row in train_csv:
    if N == 1000: print row[0]
    # Grab the feature data
    features = np.array([float(x) for x in row[1:257]])
    all_features.append(features)
    # Grab the Homo-Lumo gap data
    gap = float(row[257])
    gaps.append(gap)
    N -= 1
    if N == 0: break
  return (gaps, all_features)

def rmse(train_csv, w, N):
  gaps, features = read_data(train_csv, N)
  Y = np.vstack(np.array(gaps)).T
  X = (np.vstack(tuple(features)))
  print X.shape
  Y_hat = np.dot(X, w)
  # print np.sqrt((np.sum(np.square(Y - Y_hat))) / 1000)

def ridge_regression(N):
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
    features = data[1]
    # Compute the ridge regression parameters
    lam = 100
    Y = np.vstack(np.array(gaps)) # N x 1
    X = (np.vstack(tuple(features))) # N x J (where J = # features)
    w = np.linalg.solve(np.dot(X.T, X) + lam * np.identity(256), np.dot(X.T, Y))
    w0 = np.sum(Y) / Y.shape[0] # compute w0 as sample mean of training data
    w = np.vstack(np.insert(w, 0, w0)) # (J+1) x 1
    rmse(train_csv, w, N)

if __name__ == "__main__":
  ridge_regression(1000)