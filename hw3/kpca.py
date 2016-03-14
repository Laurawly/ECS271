import numpy as np
import matplotlib.pyplot as plt
import pylab
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
import glob
from sklearn.decomposition import PCA, KernelPCA


X = np.array([np.array(Image.open(im)).flatten() for im in glob.glob('Images/*.pgm')], 'f')
n_samples, n_features = X.shape


# PCA computation
pca = PCA(n_components = 3)
X_pca = pca.fit_transform(X)
cov = pca.get_covariance()
U,S,V = np.linalg.svd(cov)
# Kernel PCA computation
kpca = KernelPCA(n_components=3)
X_kpca = kpca.fit_transform(X)


fig1 = pylab.figure()
ax = Axes3D(fig1)
sequence_x_vals = X_kpca[:,0]
sequence_y_vals = X_kpca[:,1]
sequence_z_vals = X_kpca[:,2]
ax.scatter(sequence_x_vals, sequence_y_vals, sequence_z_vals)
plt.title('Kernel PCA')
plt.show()
fig2 = pylab.figure()
ax = Axes3D(fig2)
sequence1_x_vals = X_pca[:,0]
sequence1_y_vals = X_pca[:,1]
sequence1_z_vals = X_pca[:,2]
ax.scatter(sequence1_x_vals, sequence1_y_vals, sequence1_z_vals)
np.savetxt('pca_eigen_vector1.txt', U[:,:3].T, delimiter = '\n')
plt.title('PCA')
plt.show()

