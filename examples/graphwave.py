import sys
sys.path.append(".")

import numpy as np
import pygsp as gsp
import sgw_tools as sgw
import matplotlib.pyplot as plt

G = gsp.graphs.Comet(10, 3)
G.compute_fourier_basis()

g = sgw.GWHeat(G)

ts = np.linspace(0, 100, 25)
emb = sgw.embedding(g, ts)
data_z = emb.reshape(emb.shape[0], emb.shape[1]*emb.shape[2])
data = np.c_[data_z.real, data_z.imag]

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

pca = PCA(n_components=3)
pca_data = pca.fit_transform(StandardScaler().fit_transform(data))
km = KMeans(n_clusters=3)
labels = km.fit_predict(pca_data)
sgw.plotGraph(G)
sgw.plotEmbedding(emb)
print(labels)
plt.show()
