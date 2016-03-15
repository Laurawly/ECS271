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
X -= np.mean(X, axis=0)
pca = PCA(n_components = 3)
X_pca = pca.fit_transform(X)
cov = pca.get_covariance()
print(cov.shape)
U,S,V = np.linalg.svd(cov)

#w,h = 64, 60
#data = U[:, 2].reshape((60,64))
#rescaled = (255.0 / data.max() * (data - data.min())).astype(np.uint8)
#img = Image.fromarray(rescaled)
#img.show()

# Kernel PCA computation
#kpca = KernelPCA(kernel = "rbf", n_components=3, gamma = 1e-4 * 0.3)
#X_kpca = kpca.fit_transform(X)

#fig1 = pylab.figure()
#ax = Axes3D(fig1)
#sequence_x_vals = X_kpca[:,0]
#sequence_y_vals = X_kpca[:,1]
#sequence_z_vals = X_kpca[:,2]
#ax.scatter(sequence_x_vals, sequence_y_vals, sequence_z_vals)
#plt.title('Kernel PCA')


fig2 = pylab.figure()
ax = Axes3D(fig2)
sequence1_x_vals = X_pca[:,0]
sequence1_y_vals = X_pca[:,1]
sequence1_z_vals = X_pca[:,2]
ax.scatter(sequence1_x_vals, sequence1_y_vals, sequence1_z_vals)
#np.savetxt('pca_eigen_vector1.txt', U[:,:3].T, delimiter = '\n')
plt.title('PCA')

#fig3 = plt.figure()
#ax = fig3.add_subplot(111)
#x = np.arange(10)
#ax.scatter(x,S[:10])
#plt.title('eigen values')

width = 1

fig4 = plt.figure()
eigen1 = np.dot(X, U[:,0])
x = np.arange(eigen1.shape[0])
ax = fig4.add_subplot(111)
rects = ax.bar(x,eigen1, width)
plt.title('pca eigen vector 1')

fig5 = plt.figure()
eigen2 = np.dot(X, U[:,1])
x = np.arange(eigen2.shape[0])
ax = fig5.add_subplot(111)
rects = ax.bar(x,eigen2, width)
plt.title('pca eigen vector 2')

fig6 = plt.figure()
eigen3 = np.dot(X, U[:,2])
x = np.arange(eigen3.shape[0])
ax = fig6.add_subplot(111)
rects = ax.bar(x,eigen3, width)
plt.title('pca eigen vector 3')

plt.show()


