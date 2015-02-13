import csv
import gzip
import numpy as np
from sklearn.linear_model import Lasso

train_filename = 'train.csv.gz'
test_filename  = 'test.csv.gz'
pred_filename  = 'example_mean.csv'

# Reads N rows from train_csv and outputs gaps and features
def read_data(train_csv, N):
  gaps = [] # list of HOMO-LUMO gap for N rows of data
  all_features = [] # list of feature vector for N rows of data
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

def rmse(train_csv, w, N):
  gaps, features = read_data(train_csv, N)
  Y = np.vstack(np.array(gaps)).T
  X = (np.vstack((np.ones(N), (np.vstack(tuple(features))).T))).T
  Y_hat = np.dot(X, w).T
  return np.sqrt((np.sum(np.square(Y - Y_hat))) / N)

# Least Angle Regression method
def LARS(X, y):
  w = np.zeros((256, 1))
  y_hat = np.dot(X, w)
  r = y - y_hat
  # A = [] # array indices of included parameter coefficients
  # xs = []
  for i in xrange(1):
    c = np.dot(X.T, y - y_hat) # current correlation vector
    C = np.amax(np.absolute(c)) # max correlation
    # j = np.argmax(np.absolute(c))
    indices = (np.argwhere(np.absolute(c) == C).T)[0]
    j = 0
    for i in indices:
      if i not in A:
        A.append(i)
        j = i
        break
    sj = np.sign(c[j])
    
    # xs.append(sj * X[:, j])
    # X_A = np.vstack(tuple(xs)).T
    # G_A = np.dot(X_A.T, X_A)
    # G_inv = np.linalg.inv(G_A)
    # one_a = np.ones((len(xs), 1))
    # A_A = np.power(np.dot(one_a.T, np.dot(G_inv, one_a)), -0.5)
    # w_A = A_A * np.dot(G_inv, one_a)
    # u_A = np.dot(X_A, w_A)
    # a = np.dot(X.T, u_A)
    # gammas = []
    # for i in xrange(256):
    #   if i not in A:
    #     gammas.append((C - c[i]) / (A_A - a[i]))
    #     gammas.append((C + c[i]) / (A_A + a[i]))
    # gammas = [x for x in gammas if x >= 0]
    # if not gammas: break
    # gamma = min(gammas)
    # mu = mu + gamma * u_A
  # w = np.dot(np.linalg.inv(X), mu)
  print mu
  return w

def forward_stagewise(X, y):
  epsilon = 0.01
  w = np.zeros((257, 1))
  r = y
  # mu = np.dot(X, w)
  for i in xrange(6000):
    # r = y - mu# residual
    # c =  np.dot(X.T, y - mu) # current correlations
    c =  np.dot(X.T, r) # current correlations
    j = np.argmax(c)
    sj = np.sign(c[j])
    delta_j = epsilon * sj
    # mu = mu + delta_j * 
    w[j] = w[j] + delta_j
    r = r - delta_j * np.vstack(X[:,j])
  return w

def lasso(N):
  # Load the training file
  with gzip.open(train_filename, 'r') as train_fh:
    # Parse it as a CSV file.
    train_csv = csv.reader(train_fh, delimiter=',', quotechar='"')
    
    # Skip the header row.
    next(train_csv, None)
    data = read_data(train_csv, N)
    gaps = data[0]
    features = data[1]
    Y = np.vstack(np.array(gaps)) # N x 1
    X = (np.vstack((np.ones(N), (np.vstack(tuple(features))).T))).T # N x (J+1 = 257)

    # X = np.vstack(tuple(features))
    # LARS(X, Y)
    w = forward_stagewise(X, Y)
    return rmse(train_csv, w, N)

def real_lasso(N):
  # Load the training file
  with gzip.open(train_filename, 'r') as train_fh:
    # Parse it as a CSV file.
    train_csv = csv.reader(train_fh, delimiter=',', quotechar='"')
    
    # Skip the header row.
    next(train_csv, None)
    data = read_data(train_csv, N)
    gaps = data[0]
    features = data[1]
    Y = np.vstack(np.array(gaps)) # N x 1
    X = (np.vstack((np.ones(N), (np.vstack(tuple(features))).T))).T # N x (J+1 = 257)

    # Get test data
    data = read_data(train_csv, N)
    X_test = (np.vstack((np.ones(N), (np.vstack(tuple(data[1]))).T))).T
    Y_test = np.vstack(np.array(data[0]))

    lasso_reg = Lasso(alpha = 0.01)
    Y_hat = np.vstack(lasso_reg.fit(X, Y).predict(X_test))
    return np.sqrt((np.sum(np.square(Y_test - Y_hat))) / N)

if __name__ == "__main__":
      # lasso_reg = Lasso(alpha = 0.4)
    # data = read_data(train_csv, N)
    # X_predict = (np.vstack((np.ones(N), (np.vstack(tuple(data[1]))).T))).T
    # Y = np.vstack(np.array(data[0]))
    # Y_hat = lasso_reg.fit(X, Y).predict(X_predict)
    # return np.sqrt((np.sum(np.square(Y - Y_hat))) / N)
  print real_lasso(1000)
  # print lasso(1000)



# def LARS(X, y):
#   w = np.zeros((256, 1))
#   mu = np.dot(X, w)
#   A = [] # array indices of included parameter coefficients
#   xs = []
#   for i in xrange(4):
#     c = np.dot(X.T, y - mu) # current correlation vector
#     C = np.amax(np.absolute(c)) # max correlation
#     # j = np.argmax(np.absolute(c))
#     indices = (np.argwhere(np.absolute(c) == C).T)[0]
#     j = 0
#     for i in indices:
#       if i not in A:
#         A.append(i)
#         j = i
#         break
#     sj = np.sign(c[j])
#     xs.append(sj * X[:, j])
#     X_A = np.vstack(tuple(xs)).T
#     G_A = np.dot(X_A.T, X_A)
#     G_inv = np.linalg.inv(G_A)
#     one_a = np.ones((len(xs), 1))
#     A_A = np.power(np.dot(one_a.T, np.dot(G_inv, one_a)), -0.5)
#     w_A = A_A * np.dot(G_inv, one_a)
#     u_A = np.dot(X_A, w_A)
#     a = np.dot(X.T, u_A)
#     gammas = []
#     for i in xrange(256):
#       if i not in A:
#         gammas.append((C - c[i]) / (A_A - a[i]))
#         gammas.append((C + c[i]) / (A_A + a[i]))
#     gammas = [x for x in gammas if x >= 0]
#     if not gammas: break
#     gamma = min(gammas)
#     mu = mu + gamma * u_A
#   # w = np.dot(np.linalg.inv(X), mu)
#   print mu
#   return w
