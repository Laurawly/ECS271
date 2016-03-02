#
try:
    import numpy as np
except:
    print "This implementation requires the numpy module."
    exit(0)

import csv
import scipy.sparse as sparse
import os
from sklearn.utils.extmath import randomized_svd
from sklearn import cross_validation

def SVD(R, K):
    U, s, VT = randomized_svd(R, n_components = K, n_iter=6, random_state=3)	
    S = np.diag(s)
    return U, S, VT
##############################################################################


if __name__ == "__main__":

    X = np.loadtxt('train.csv', delimiter = ',', usecols = (0,1,2), skiprows = 1)
    Y = np.loadtxt('test.csv', delimiter = ',', usecols = (0,1), skiprows = 1, dtype=int)

    shape = tuple(X.max(axis=0)[:2]+1)
    R = sparse.coo_matrix((X[:,2], (X[:,0], X[:,1])), shape = shape, dtype = float)
    R = R.todense()
    R = np.array(R)
    mask = np.divide(np.array(R, dtype=int), np.array(R, dtype=int))
    N = len(R)
    M = len(R[0])
    K = 23
    mean = np.mean(R)+3.
    r = R + mean -np.multiply(mask,3)#- np.multiply(mask, Z+mean)
    U, S, VT = SVD(r, K)
    result = np.around(np.array(np.dot(U, np.dot(S,VT))))

    Z = result[Y[:,0],Y[:,1]]
    np.savetxt('svd_test.csv', Z, delimiter = '\n')
