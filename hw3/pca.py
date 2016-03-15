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
X -= X.mean(axis=1).reshape(n_samples,-1)
cov = np.dot(X.T, X) / X.shape[0] # get the data covariance matrix
U,S,V = np.linalg.svd(cov)
print(U.shape)
Xrot_reduced = np.dot(X, U[:,:3]) # Xrot_reduced becomes [N x 100]
fig = pylab.figure()
ax = Axes3D(fig)
sequence_x_vals = Xrot_reduced[:,0]
sequence_y_vals = Xrot_reduced[:,1]
sequence_z_vals = Xrot_reduced[:,2]
ax.scatter(sequence_x_vals, sequence_y_vals, sequence_z_vals)
plt.title('pca projection')
#np.savetxt('pca_eigen_vector.txt', U[:,:3].T, delimiter = '\n\n')
#plt.show()

#w,h = 64, 60
#data = U[:, 2].reshape((60,64))
#rescaled = (255.0 / data.max() * (data - data.min())).astype(np.uint8)
#img = Image.fromarray(rescaled)
#img.save('fig2.png')

width = 1

fig1 = plt.figure()
eigen1 = np.dot(X, U[:,0])
x = np.arange(eigen1.shape[0])
ax = fig1.add_subplot(111)
rects = ax.bar(x, eigen1.T, width)
plt.title('pca eigen vector 1')

fig2 = plt.figure()
eigen2 = np.dot(X, U[:,1])
x = np.arange(eigen2.shape[0])
ax = fig2.add_subplot(111)
rects = ax.bar(x, eigen2.T, width)
plt.title('pca eigen vector 2')

fig3 = plt.figure()
eigen3 = np.dot(X, U[:,2])
x = np.arange(eigen3.shape[0])
ax = fig3.add_subplot(111)
rects = ax.bar(x, eigen3.T, width)
plt.title('pca eigen vector 3')

plt.show()
