import numpy as np
import matplotlib.pyplot as plt
import pylab
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
import glob
from sklearn.decomposition import PCA, KernelPCA
from sklearn import manifold, datasets


X = np.array([np.array(Image.open(im)).flatten() for im in glob.glob('Images/*.pgm')], 'f')
n_samples, n_features = X.shape


# lle
lle = manifold.LocallyLinearEmbedding(n_neighbors=50,n_components=3) 
X_r0 = lle.fit_transform(X)
fig1 = pylab.figure()
ax = Axes3D(fig1)
sequence0_x_vals = X_r0[:,0]
sequence0_y_vals = X_r0[:,1]
sequence0_z_vals = X_r0[:,2]
ax.scatter(sequence0_x_vals, sequence0_y_vals, sequence0_z_vals)
plt.title('Local Linear Embedding_50')

lle = manifold.LocallyLinearEmbedding(n_neighbors=5,n_components=3) 
X_r1 = lle.fit_transform(X)
fig5 = pylab.figure()
ax = Axes3D(fig5)
sequence1_x_vals = X_r1[:,0]
sequence1_y_vals = X_r1[:,1]
sequence1_z_vals = X_r1[:,2]
ax.scatter(sequence1_x_vals, sequence1_y_vals, sequence1_z_vals)
plt.title('Local Linear Embedding_5')

lle = manifold.LocallyLinearEmbedding(n_neighbors=30,n_components=3) 
X_r = lle.fit_transform(X)
fig6 = pylab.figure()
ax = Axes3D(fig6)
sequence_x_vals = X_r[:,0]
sequence_y_vals = X_r[:,1]
sequence_z_vals = X_r[:,2]
ax.scatter(sequence_x_vals, sequence_y_vals, sequence_z_vals)
plt.title('Local Linear Embedding_30')

width = 1e-7

fig2 = plt.figure()
ax = fig2.add_subplot(111)
x = np.arange(sequence_x_vals.shape[0])
rects = ax.bar(x, sequence_x_vals, width)
plt.title('lle eigen vector 1')

fig3 = plt.figure()
ax = fig3.add_subplot(111)
x = np.arange(sequence_y_vals.shape[0])
rects = ax.bar(x, sequence_y_vals, width)
plt.title('lle eigen vector 2')

fig4 = plt.figure()
ax = fig4.add_subplot(111)
x = np.arange(sequence_z_vals.shape[0])
rects = ax.bar(x, sequence_z_vals, width)
plt.title('lle eigen vector 3')

plt.show()

