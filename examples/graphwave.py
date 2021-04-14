import sys
sys.path.append(".")

import numpy as np
import pygsp as gsp
import sgw_tools as sgw
import matplotlib.pyplot as plt

G = gsp.graphs.Comet(20, 11)
G.compute_fourier_basis()

g = sgw.GWHeat(G, Nf=2)

sgw.plotTig(g)

ts = np.linspace(0, 100, 25)
gw = sgw.graphWave(G, g, ts)
data_z = gw.reshape(gw.shape[0], gw.shape[1]*gw.shape[2])
data = np.c_[data_z.real, data_z.imag]

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

pca = PCA(n_components=2)
pca_data = pca.fit_transform(StandardScaler().fit_transform(data))
km = KMeans(n_clusters=3)
labels = km.fit_predict(pca_data)
sgw.plotGraph(G)
sgw.plotGraphWave(gw)
plt.figure()
plt.title('GraphWave PCA')
plt.scatter(pca_data[:,0], pca_data[:,1], s=20)
for i in range(G.N):
    plt.gca().annotate(str(i), (pca_data[i,0], pca_data[i,1]))
print(labels)
plt.show()
