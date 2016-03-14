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
lle = manifold.LocallyLinearEmbedding(n_neighbors=10,n_components=3) 
X_r = lle.fit_transform(X)
fig1 = pylab.figure()
ax = Axes3D(fig1)
sequence_x_vals = X_r[:,0]
sequence_y_vals = X_r[:,1]
sequence_z_vals = X_r[:,2]
ax.scatter(sequence_x_vals, sequence_y_vals, sequence_z_vals)
plt.title('Local Linear Embedding')
plt.show()

