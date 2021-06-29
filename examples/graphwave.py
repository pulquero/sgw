import sys
sys.path.append(".")

import numpy as np
import scipy.sparse as sparse
import pygsp as gsp
import sgw_tools as sgw
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


def pca(data, clusters):
    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(data)
    print('Variance', pca.explained_variance_ratio_)
    km = KMeans(n_clusters=clusters)
    labels = km.fit_predict(pca_data)
    print('Clusters', labels)
    plt.figure('PCA')
    plt.title('GraphWave PCA')
    plt.scatter(pca_data[:,0], pca_data[:,1], s=20)
    xlim = plt.gca().get_xlim()
    min_r = 0.1*(xlim[1]-xlim[0])/plt.gcf().get_figwidth()
    for i, center in enumerate(km.cluster_centers_):
        radius = np.max([np.linalg.norm(np.subtract(pt, center)) for pt in pca_data[labels==i]])
        plt.gca().add_patch(plt.Circle((center[0], center[1]), max(min_r,radius), color='grey', linestyle=':', fill=False))
    for i in range(len(data)):
        plt.gca().text(pca_data[i,0], pca_data[i,1], str(i))
    plt.show()


def simple_demo():
    print("*** Simple demo ***")
    G = gsp.graphs.Comet(20, 11)

    G.compute_fourier_basis()
    g = sgw.GWHeat(G, Nf=2, gamma=0.95, eta=0.85)
    ts = np.linspace(0, 100, 25)
    gw = sgw.graphWave(G, g, ts)
    data = sgw.gw_flatten(gw)

    sgw.plotGraph(G)
    sgw.plotTig(g)
    sgw.plotGraphWave(gw)
    pca(data, 3)


def build_undirected_graph():
    W = np.zeros((31, 31))
    W[0,1] = 1
    W[1,0] = 1
    for i in range(2,13):
        W[i-2,i] = 1
        W[i,i-2] = 1
    clique1 = [11]
    clique1.extend(range(13,22))
    for i, u in enumerate(clique1):
        for v in clique1[i+1:]:
            W[u,v] = 1
            W[v,u] = 1
    clique2 = [12]
    clique2.extend(range(22,31))
    for i, u in enumerate(clique2):
        for v in clique2[i+1:]:
            W[u,v] = 1
            W[v,u] = 1
    return sparse.csr_matrix(W)


def undirected_demo():
    print("*** Undirected demo ***")
    W = build_undirected_graph()
    G = sgw.BigGraph(W)

    G.compute_fourier_basis()
    g = sgw.GWHeat(G, Nf=2, gamma=0.95, eta=0.4)
    ts = np.linspace(0, 100, 50)
    gw = sgw.graphWave(G, g, ts)
    data = sgw.gw_flatten(gw)

    sgw.plotGraph(G)
    pca(data, 8)


def build_directed_graph():
    W = np.zeros((21, 21))
    W[0,1] = 1
    for i in range(2,13):
        W[i-2,i] = 1
    for i in range(13,17):
        W[11,i] = 1
    for i in range(17,21):
        W[i,12] = 1
    return sparse.csr_matrix(W)


def directed_demo():
    print("*** Directed demo ***")
    W = build_directed_graph()
    G = sgw.BigGraph(W, q=0.02)
    
    G.compute_fourier_basis()
    g = sgw.GWHeat(G, Nf=10, gamma=0.68, eta=0.021)
    ts = np.linspace(1, 10, 10)
    gw = sgw.graphWave(G, g, ts, method='exact')
    data = sgw.gw_flatten(gw)
    
    sgw.plotGraph(G)
    pca(data, 9)

simple_demo()
undirected_demo()
directed_demo()
