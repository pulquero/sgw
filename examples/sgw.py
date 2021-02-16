import sys
sys.path.append(".")

import numpy as np
import pygsp as gsp
import sgw_tools as sgw
import matplotlib.pyplot as plt

G = gsp.graphs.Comet(10, 3)
G.compute_fourier_basis()
g = gsp.filters.MexicanHat(G, 3, lpfactor=10)
sig = sgw.signature(g)
codebook = sgw.codebook(sig, 3)
code = sgw.code(sig, codebook, 0.8)

from sklearn.decomposition import PCA

pca = PCA(n_components=2)
sig_pca = pca.fit_transform(sig)
codebook_pca = pca.fit_transform(codebook)
plt.scatter(sig_pca[:,0], sig_pca[:,1])
plt.scatter(codebook_pca[:,0], codebook_pca[:,1])
plt.show()
