import csv
import gzip
import numpy as np

def read_data(train_csv, N):
  gaps = []
  all_features = []
  # Load the data.
  for row in train_csv:
    # Grab the feature data
    features = np.array([float(x) for x in row[1:257]])
    all_features.append(features)
    # Grab the Homo-Lumo gap data
    gap = float(row[257])
    gaps.append(gap)
    N -= 1
    if N == 0: break
  return (gaps, all_features)

def read_data_test(train_csv, N):
  gaps = []
  all_features = []
  old_N = N
  # Load the data.
  for row in train_csv:
    if N <= (old_N/2):
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
  X = (np.vstack((np.ones(N), (np.vstack(tuple(features))).T))).T
  Y_hat = np.dot(X, w).T
  return np.sqrt((np.sum(np.square(Y - Y_hat))) / N)

def rmse_new_data(train_csv, w, N):
  gaps, features = read_data_test(train_csv, 2*N)
  Y = np.vstack(np.array(gaps)).T
  X = (np.vstack((np.ones(N), (np.vstack(tuple(features))).T))).T
  Y_hat = np.dot(X, w).T
  return np.sqrt((np.sum(np.square(Y - Y_hat))) / N)

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
    Y = np.vstack(np.array(gaps)) # N x 1
    X = (np.vstack((np.ones(N), (np.vstack(tuple(features))).T))).T
    # X = (np.vstack(tuple(features))) # N x J (where J = # features)
    w = np.linalg.solve(np.dot(X.T, X) + lam * np.identity(257), np.dot(X.T, Y))
    # w0 = np.sum(Y) / Y.shape[0] # compute w0 as sample mean of training data
    # w = np.vstack(np.insert(w, 0, w0)) # (J+1) x 1
    return rmse(train_csv, w, N)

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
      w = np.linalg.solve(np.dot(X.T, X) + lam * np.identity(257), np.dot(X.T, Y))

      test_Y = np.vstack(np.array(test_gaps)).T
      test_X = (np.vstack((np.ones(N/5), (np.vstack(tuple(test_features))).T))).T
      Y_hat = np.dot(test_X, w).T
      lam_err += np.sum(np.square(test_Y - Y_hat))
    
    if lam_err/N < min_lam_err:
      min_lam_err = lam_err/N
      print lam
      print min_lam_err
      lam -= .07
    else:
      break

  return lam

def ridge_plus_validation(N):
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
    lam = cross_validation(gaps, features, N)
    Y = np.vstack(np.array(gaps))
    X = (np.vstack((np.ones(N), (np.vstack(tuple(features))).T))).T
    w = np.linalg.solve(np.dot(X.T, X) + lam * np.identity(257), np.dot(X.T, Y))
    return rmse_new_data(train_csv, w, N)

if __name__ == "__main__":
  print ridge_plus_validation(40000)