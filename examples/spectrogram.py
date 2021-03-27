import sys
sys.path.append(".")

import numpy as np
import pygsp as gsp
import sgw_tools as sgw
import matplotlib.pyplot as plt

G = gsp.graphs.Comet(20, 11)
G.estimate_lmax()

g = sgw.GaussianFilter(G, Nf=100)
data = sgw.spectrogram(G, g)

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

pca = PCA(n_components=2)
pca_data = pca.fit_transform(StandardScaler().fit_transform(data))
km = KMeans(n_clusters=3)
labels = km.fit_predict(pca_data)
sgw.plotGraph(G)
gsp.plotting.plot_spectrogram(G)
plt.figure()
plt.scatter(pca_data[:,0], pca_data[:,1], s=20)
for i in range(G.N):
    plt.gca().annotate(str(i), (pca_data[i,0], pca_data[i,1]))
print(labels)
plt.show()
