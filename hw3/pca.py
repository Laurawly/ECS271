import numpy as np
import matplotlib.pyplot as plt
import pylab
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
import glob


X = np.array([np.array(Image.open(im)).flatten() for im in glob.glob('Images/*.pgm')], 'f')
n_samples, n_features = X.shape

# PCA computation
X -= np.mean(X, axis = 0)
cov = np.dot(X.T, X) / X.shape[0] # get the data covariance matrix
U,S,V = np.linalg.svd(cov)
Xrot_reduced = np.dot(X, U[:,:3]) # Xrot_reduced becomes [N x 100]
fig = pylab.figure()
ax = Axes3D(fig)
sequence_x_vals = Xrot_reduced[:,0]
sequence_y_vals = Xrot_reduced[:,1]
sequence_z_vals = Xrot_reduced[:,2]
ax.scatter(sequence_x_vals, sequence_y_vals, sequence_z_vals)
np.savetxt('pca_eigen_vector.txt', U[:,:3].T, delimiter = '\n\n')
plt.show()

