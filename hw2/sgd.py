#
try:
    import numpy as np
except:
    print "This implementation requires the numpy module."
    exit(0)

import csv
import scipy.sparse as sparse
import os
os.environ['CUDARRAY_BACKEND'] = 'cuda'
import cudarray as ca

###############################################################################

"""
@INPUT:
    R     : a matrix to be factorized, dimension N x M
    P     : an initial matrix of dimension N x K
    Q     : an initial matrix of dimension M x K
    K     : the number of latent features
    steps : the maximum number of steps to perform the optimisation
    alpha : the learning rate
    beta  : the regularization parameter
@OUTPUT:
    the final matrices P and Q
"""
def matrix_factorization(R, P, Q, mask, steps=200000000, alpha=0.00005, beta=0.02):
    Q = ca.transpose(Q)
    for step in xrange(steps):
 	E = ca.subtract(R, ca.multiply(ca.dot(P,Q), mask))

	rmse = ca.sqrt(ca.sum(ca.power(E,2)) / ca.sum(mask))
	rmse = np.array(rmse)[0]

 	print 'step: %i RMSE: %f' % (step, rmse)
        if rmse < 0.65:
            break
	P = ca.add(ca.multiply(P,(1-alpha*beta)),ca.multiply(ca.dot(E,ca.transpose(Q)), 2*alpha))
	Q = ca.add(ca.multiply(Q,(1-alpha*beta)),ca.multiply(ca.dot(ca.transpose(P),E),2*alpha))

    return P, Q



if __name__ == "__main__":
    X = np.loadtxt('train.csv', delimiter = ',', usecols = (0,1,2), skiprows = 1)
    Y = np.loadtxt('test.csv', delimiter = ',', usecols = (0,1), skiprows = 1, dtype=int)
    shape = tuple(X.max(axis=0)[:2]+1)
    R = sparse.coo_matrix((X[:,2], (X[:,0], X[:,1])), shape = shape, dtype = X.dtype)
    R = R.todense()
    R = np.array(R)
    mask = np.divide(np.array(R, dtype=int), np.array(R, dtype=int))
    d_R = ca.array(R)
    d_M = ca.array(mask)
    N = len(R)
    M = len(R[0])
    K = 23

    P = np.random.rand(N,K)
    Q = np.random.rand(M,K)

    d_P = ca.array(P)
    d_Q = ca.array(Q)

    d_nP, d_nQ = matrix_factorization(d_R, d_P, d_Q, d_M)

    d_nR = ca.dot(d_nP, d_nQ)
    nR = np.around(np.array(d_nR))
    Z = nR[Y[:,0],Y[:,1]]
    np.savetxt('sgd_test.csv', Z, delimiter = '\n')
